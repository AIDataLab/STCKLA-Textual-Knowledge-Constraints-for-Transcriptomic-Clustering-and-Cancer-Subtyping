"""llm.py definitions moved here without algorithm changes."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from ..logging_utils import diagnostic_print


def _brca_default_subtype_map() -> Dict[int, str]:
    return {0: "LumA", 1: "Her2", 2: "LumB", 3: "Normal", 4: "Basal"}


def load_subtype_map(path: str, dataset_key: str = "") -> Dict[int, str]:
    if path and Path(path).exists():
        try:
            if str(path).lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            cols_low = {str(c).strip().lower(): c for c in df.columns}
            id_col = None
            name_col = None
            for cand in ["label", "label_id", "class_id", "id", "num", "number"]:
                if cand in cols_low:
                    id_col = cols_low[cand]
                    break
            for cand in ["subtype", "subtype_name", "name", "celltype", "cell_type", "type"]:
                if cand in cols_low:
                    name_col = cols_low[cand]
                    break
            if id_col is None or name_col is None:
                id_col = df.columns[0]
                name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            out = {}
            for _, row in df.iterrows():
                try:
                    k = int(row[id_col])
                    v = str(row[name_col]).strip()
                    if v and v.lower() != "nan":
                        out[k] = v
                except Exception:
                    continue
            if out:
                diagnostic_print(f"[LABEL] subtype_map loaded from {path}: {out}", flush=True)
                return out
        except Exception as e:
            print(f"[LABEL][WARN] failed to read subtype_map_path={path}: {e}", flush=True)

    key = str(dataset_key).lower()
    if "brca" in key or "breast" in key:
        m = _brca_default_subtype_map()
        print(f"[LABEL][WARN] subtype_map_path not found or invalid: {path}; use built-in BRCA map: {m}", flush=True)
        return m
    return {}


def load_label_series(label_path: str, sample_index: pd.Index) -> pd.Series:
    if not label_path:
        raise ValueError("--label_path is required")
    p = Path(label_path)
    if not p.exists():
        raise FileNotFoundError(f"label_path not found: {label_path}")
    if str(p).lower().endswith(".npy"):
        arr = np.load(p, allow_pickle=True)
        s = pd.Series(arr, index=sample_index)
    else:
        df = pd.read_csv(p, index_col=0)
        if df.shape[1] >= 1:
            s = df.iloc[:, 0]
        else:
            s = pd.Series(df.index, index=df.index)
        if set(map(str, sample_index)).issubset(set(map(str, s.index))):
            s.index = s.index.map(str)
            s = s.reindex(sample_index.map(str))
            s.index = sample_index
        elif len(s) == len(sample_index):
            s = pd.Series(s.to_numpy(), index=sample_index)
        else:
            raise ValueError(f"label length/index mismatch: labels={len(s)}, samples={len(sample_index)}")
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any():
        raise ValueError("label_path contains non-numeric or missing labels after alignment")
    return s.astype(int)


def _extract_indices_from_obj(obj: Any, n: int, sample_index: pd.Index) -> np.ndarray:
    if isinstance(obj, dict):
        for k in ["exposed_rows", "rows", "indices", "exposed_indices", "labeled_indices", "ids", "exposed_ids"]:
            if k in obj:
                return _extract_indices_from_obj(obj[k], n, sample_index)
    arr = np.asarray(obj)
    if arr.dtype == bool and arr.size == n:
        return np.where(arr)[0]
    vals = arr.tolist()
    if not isinstance(vals, list):
        vals = [vals]
    out = []
    sample_str_to_i = {str(x): i for i, x in enumerate(sample_index)}
    for v in vals:
        if isinstance(v, (int, np.integer)):
            if 0 <= int(v) < n:
                out.append(int(v))
        else:
            sv = str(v)
            if sv in sample_str_to_i:
                out.append(sample_str_to_i[sv])
            else:
                try:
                    iv = int(float(sv))
                    if 0 <= iv < n:
                        out.append(iv)
                except Exception:
                    pass
    return np.asarray(sorted(set(out)), dtype=int)


def load_exposed_mask(split_dir: str, exposed_index_path: str, n: int, sample_index: pd.Index) -> Tuple[np.ndarray, str]:
    paths = []
    if exposed_index_path:
        paths.append(Path(exposed_index_path))
    if split_dir:
        sd = Path(split_dir)
        paths.extend([
            sd / "exposed_rows.npy",
            sd / "exposed_mask.npy",
            sd / "exposed_ids.json",
            sd / "exposed_rows.json",
            sd / "exposed_indices.json",
            sd / "split.json",
        ])
    for p in paths:
        if not p.exists():
            continue
        try:
            if str(p).lower().endswith(".npy"):
                obj = np.load(p, allow_pickle=True)
            elif str(p).lower().endswith(".json"):
                obj = json.load(open(p, "r", encoding="utf-8"))
            else:
                obj = pd.read_csv(p, header=None).iloc[:, 0].tolist()
            idx = _extract_indices_from_obj(obj, n, sample_index)
            mask = np.zeros(n, dtype=bool)
            mask[idx[(idx >= 0) & (idx < n)]] = True
            if mask.sum() > 0:
                print(f"[LABEL] exposed split loaded from {p} | exposed_n={int(mask.sum())}/{n}", flush=True)
                return mask, str(p)
        except Exception as e:
            print(f"[LABEL][WARN] failed to parse exposed split {p}: {e}", flush=True)
    raise FileNotFoundError("No valid exposed split found. Provide --split_dir or --exposed_index_path.")


def save_exposed_ids_qwen(
    save_dir: str,
    sample_index: pd.Index,
    exposed_mask: np.ndarray,
    *,
    seed: int,
    labeled_ratio: float,
    stratified: bool,
    label_path: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    exposed_mask = np.asarray(exposed_mask, dtype=bool)
    if exposed_mask.ndim != 1:
        raise RuntimeError(f"[EXPOSED] exposed_mask must be 1D, got shape={exposed_mask.shape}")

    n_total = int(len(sample_index))
    if n_total != len(exposed_mask):
        raise RuntimeError(f"[EXPOSED] sample_index len {n_total} != exposed_mask len {len(exposed_mask)}")

    rows = np.where(exposed_mask)[0].astype(np.int64)
    exposed_rows = rows.tolist()

    exposed_ids = [str(i) for i in exposed_rows]

    with open(os.path.join(save_dir, "exposed_ids.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "exposed_ids": exposed_ids,
                "exposed_rows": exposed_rows,
                "id_type": "row_index",
                "n_exposed": int(len(exposed_ids)),
                "n_total": n_total,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.save(os.path.join(save_dir, "exposed_rows.npy"), rows)
    np.save(os.path.join(save_dir, "exposed_mask.npy"), exposed_mask.astype(np.bool_))

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "save_dir": str(save_dir),
        "label_path": str(label_path),
        "seed": int(seed),
        "labeled_ratio": float(labeled_ratio),
        "stratified_sample": bool(stratified),
        "sampling_mode": "class-balanced random sampling" if bool(stratified) else "global random sampling",
        "id_type": "row_index",
        "n_total": n_total,
        "n_exposed": int(len(exposed_ids)),
        "files": ["exposed_ids.json", "exposed_rows.npy", "exposed_mask.npy", "exposed_meta.json"],
        "created_by": "testqwen_16_0629_1.py",
    }

    with open(os.path.join(save_dir, "exposed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[EXPOSED] saved exposed split to: {save_dir}", flush=True)
    print(f"[EXPOSED] n_exposed={meta['n_exposed']} / n_total={meta['n_total']} | id_type=row_index", flush=True)


def pick_labeled_indices_qwen(
    y_id: np.ndarray,
    ratio: float,
    seed: int = 0,
    stratified: bool = True,
) -> np.ndarray:
    y_id = np.asarray(y_id, dtype=np.int64).reshape(-1)
    n = int(len(y_id))
    rng = np.random.RandomState(int(seed))

    ratio = float(ratio)
    ratio = min(max(ratio, 0.0), 1.0)

    m = int(round(n * ratio))
    m = max(1, m) if ratio > 0 else 0

    exposed_mask = np.zeros(n, dtype=bool)
    if m <= 0:
        return exposed_mask

    classes = np.unique(y_id)
    num_classes = len(classes)

    if (not stratified) or (num_classes <= 1):
        idx = rng.permutation(n)[:m]
        exposed_mask[idx] = True
        return exposed_mask

    class_to_indices = {int(c): np.where(y_id == c)[0] for c in classes}
    capacities = {int(c): len(class_to_indices[int(c)]) for c in classes}

    if m < num_classes:
        chosen_classes = rng.permutation(classes)[:m]
        for c in chosen_classes:
            idx_c = class_to_indices[int(c)]
            pick = rng.permutation(idx_c)[:1]
            exposed_mask[pick] = True
        return exposed_mask

    quotas = {int(c): 0 for c in classes}
    base = m // num_classes
    remainder = m % num_classes

    for c in classes:
        quotas[int(c)] = min(base, capacities[int(c)])

    assigned = sum(quotas.values())
    leftover = m - assigned

    if remainder > 0 and leftover > 0:
        eligible = [int(c) for c in rng.permutation(classes) if quotas[int(c)] < capacities[int(c)]]
        for c in eligible[:min(remainder, leftover)]:
            quotas[c] += 1
            leftover -= 1

    while leftover > 0:
        eligible = [int(c) for c in classes if quotas[int(c)] < capacities[int(c)]]
        if len(eligible) == 0:
            break

        rng.shuffle(eligible)
        progressed = False

        for c in eligible:
            if leftover <= 0:
                break
            if quotas[c] < capacities[c]:
                quotas[c] += 1
                leftover -= 1
                progressed = True

        if not progressed:
            break

    picks = []
    for c in classes:
        c = int(c)
        k = int(quotas[c])
        if k <= 0:
            continue
        idx_c = class_to_indices[c]
        pick = rng.permutation(idx_c)[:k]
        picks.append(pick)

    picks = np.concatenate(picks) if len(picks) else np.array([], dtype=int)

    if len(picks) < m:
        rest = np.setdiff1d(np.arange(n), picks, assume_unique=False)
        add = rng.permutation(rest)[: (m - len(picks))]
        picks = np.concatenate([picks, add])
    elif len(picks) > m:
        picks = rng.permutation(picks)[:m]

    exposed_mask[picks] = True
    return exposed_mask


def create_and_save_exposed_mask_qwen(
    labels: pd.Series,
    sample_index: pd.Index,
    save_dir: str,
    *,
    ratio: float,
    seed: int,
    stratified: bool,
    label_path: str,
) -> Tuple[np.ndarray, str]:
    if not save_dir:
        raise ValueError("[EXPOSED] --split_dir is required when creating exposed split")

    y_id = labels.to_numpy(dtype=np.int64)
    exposed_mask = pick_labeled_indices_qwen(
        y_id,
        ratio=ratio,
        seed=seed,
        stratified=stratified,
    )

    if int(exposed_mask.sum()) <= 0:
        raise RuntimeError("[EXPOSED] created exposed_mask has 0 samples")

    save_exposed_ids_qwen(
        save_dir=save_dir,
        sample_index=sample_index,
        exposed_mask=exposed_mask,
        seed=seed,
        labeled_ratio=ratio,
        stratified=stratified,
        label_path=label_path,
    )

    return exposed_mask, os.path.join(save_dir, "exposed_mask.npy")
