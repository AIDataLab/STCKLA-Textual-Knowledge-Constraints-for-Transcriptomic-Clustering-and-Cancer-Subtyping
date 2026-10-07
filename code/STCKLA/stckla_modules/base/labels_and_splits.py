"""base.py definitions moved here without algorithm changes."""

import os
import json
import time
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from ..logging_utils import diagnostic_print


def save_exposed_ids(
    save_dir: str,
    sample_index: pd.Index,
    labeled_mask: np.ndarray,
    *,
    seed: int,
    labeled_ratio: float,
    stratified: bool,
    label_path: str,
):
    os.makedirs(save_dir, exist_ok=True)

    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    if labeled_mask.ndim != 1:
        raise RuntimeError(f"[EXPOSED] labeled_mask must be 1D, got shape={labeled_mask.shape}")

    n_total = int(len(sample_index))
    if n_total != len(labeled_mask):
        raise RuntimeError(f"[EXPOSED] sample_index len {n_total} != labeled_mask len {len(labeled_mask)}")

    rows = np.where(labeled_mask)[0].astype(np.int64)
    exposed_rows = rows.tolist()
    # Use row indices to keep sample IDs consistent across file formats.
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
    np.save(os.path.join(save_dir, "exposed_mask.npy"), labeled_mask.astype(np.bool_))

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
    }
    with open(os.path.join(save_dir, "exposed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[EXPOSED] saved exposed set to: {save_dir}")
    print(f"          n_exposed={meta['n_exposed']} / n_total={meta['n_total']} | id_type=row_index")


def load_fixed_exposed_mask_if_exists(save_dir: str, n_samples: int) -> Optional[np.ndarray]:
    path = os.path.join(save_dir, "exposed_mask.npy")
    if not os.path.isfile(path):
        return None
    try:
        mask = np.load(path)
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != int(n_samples):
            print(f"[EXPOSED][WARN] existing exposed_mask.npy shape mismatch: got {mask.shape}, expect ({n_samples},)")
            return None
        print(f"[EXPOSED] loaded fixed exposed_mask from {path}, labeled_cnt={int(mask.sum())}/{len(mask)}")
        return mask
    except Exception as e:
        print("[EXPOSED][WARN] failed to load exposed_mask.npy:", repr(e))
        return None


def load_exposed_sample_ids(train_meta_dir: str) -> Optional[List[str]]:
    if not train_meta_dir:
        return None
    cand = [
        os.path.join(train_meta_dir, "exposed_ids.json"),
        os.path.join(train_meta_dir, "exposed_samples.json"),
        os.path.join(train_meta_dir, "exposed_sample_ids.json"),
    ]
    for p in cand:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if isinstance(meta, dict):
            # Prefer row-index IDs.
            for k in ["exposed_rows", "rows", "exposed_row_indices"]:
                if k in meta and isinstance(meta[k], list):
                    ids = [str(int(x)) for x in meta[k]]
                    print(f"[EXPOSED] loaded {len(ids)} exposed ids from {p} (key={k}, interpreted=row_index)")
                    return ids
            for k in ["exposed_sample_ids", "exposed_ids", "exposed"]:
                if k in meta and isinstance(meta[k], list):
                    ids = [str(x) for x in meta[k]]
                    print(f"[EXPOSED] loaded {len(ids)} exposed ids from {p} (key={k})")
                    return ids
        elif isinstance(meta, list):
            ids = [str(x) for x in meta]
            print(f"[EXPOSED] loaded {len(ids)} exposed ids(list) from {p}")
            return ids
    print(f"[EXPOSED][WARN] no exposed ids json found in: {train_meta_dir}")
    return None


def build_keep_mask_from_exposed(sample_ids: List[str], exposed_ids: List[str]) -> np.ndarray:
    sample_ids_str = [str(x) for x in sample_ids]
    exposed_set = set(str(x) for x in exposed_ids)
    hit = sum(sid in exposed_set for sid in sample_ids_str)
    keep_mask = np.array([sid not in exposed_set for sid in sample_ids_str], dtype=bool)
    diagnostic_print(f"[EXPOSED][CHECK] sample_ids={len(sample_ids_str)} exposed_ids={len(exposed_set)} hit={hit} keep={int(keep_mask.sum())}")
    return keep_mask


def save_eval_split(save_dir: str, sample_ids: List[str], keep_mask: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    eval_ids = [str(sample_ids[i]) for i in np.where(keep_mask)[0].tolist()]
    exposed_ids = [str(sample_ids[i]) for i in np.where(~keep_mask)[0].tolist()]
    with open(os.path.join(save_dir, "eval_split_used.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total": int(len(sample_ids)),
                "n_eval_keep": int(len(eval_ids)),
                "n_exposed_filtered": int(len(exposed_ids)),
                "eval_keep_sample_ids": eval_ids[:5000],
                "exposed_filtered_sample_ids": exposed_ids[:5000]
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    pd.DataFrame({
        "sample_id": [str(x) for x in sample_ids],
        "row_id": np.arange(len(sample_ids), dtype=np.int64),
        "is_eval_keep": keep_mask.astype(int)
    }).to_csv(os.path.join(save_dir, "eval_split_used.csv"), index=False)


def load_labels(label_path: str, sample_index: Optional[pd.Index] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    df = pd.read_csv(label_path)
    if df.shape[1] < 1:
        raise RuntimeError(f"[LABEL] label csv has no columns: {label_path}")

    y_raw = df.iloc[:, 0].astype(str).values

    if sample_index is not None:
        candidates = []
        for c in df.columns[1:]:
            lc = str(c).lower()
            if ("sample" in lc) or (lc in ["id", "sid", "sample_id", "case_id"]):
                candidates.append(c)
        if df.shape[1] >= 2:
            candidates = [df.columns[1]] + candidates

        used = None
        for c in candidates:
            try:
                sids = df[c].astype(str)
                if len(set(sids)) == len(sids):
                    tmp = df.copy()
                    tmp.index = sids.values
                    inter = len(set(tmp.index) & set(sample_index.astype(str)))
                    if inter >= int(0.8 * len(sample_index)):
                        tmp = tmp.reindex(sample_index.astype(str))
                        if tmp.iloc[:, 0].isna().any():
                            continue
                        y_raw = tmp.iloc[:, 0].astype(str).values
                        used = c
                        break
            except Exception:
                continue
        if used is not None:
            print(f"[LABEL] aligned by sample id column: {used}")
        else:
            print("[LABEL] fallback: align labels by row order (requires same length)")

    if sample_index is not None and len(y_raw) != len(sample_index):
        raise RuntimeError(f"[LABEL] label length {len(y_raw)} != n_samples {len(sample_index)}.")

    uniq = sorted(list({str(x) for x in y_raw}))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    y_id = np.array([label2id[str(x)] for x in y_raw], dtype=np.int64)

    print(f"[LABEL] loaded labels: n={len(y_id)} num_classes={len(label2id)} head={list(label2id.items())[:5]}")
    return y_id, label2id


def pick_labeled_indices(y_id: np.ndarray, ratio: float, seed: int = 0, stratified: bool = True) -> np.ndarray:
    n = int(len(y_id))
    rng = np.random.RandomState(seed)
    ratio = float(ratio)
    ratio = min(max(ratio, 0.0), 1.0)

    m = int(round(n * ratio))
    m = max(1, m) if ratio > 0 else 0
    labeled_mask = np.zeros(n, dtype=bool)

    if m <= 0:
        return labeled_mask

    classes = np.unique(y_id)
    num_classes = len(classes)

    if (not stratified) or (num_classes <= 1):
        idx = rng.permutation(n)[:m]
        labeled_mask[idx] = True
        return labeled_mask

    class_to_indices = {int(c): np.where(y_id == c)[0] for c in classes}
    capacities = {int(c): len(class_to_indices[int(c)]) for c in classes}

    if m < num_classes:
        chosen_classes = rng.permutation(classes)[:m]
        for c in chosen_classes:
            idx_c = class_to_indices[int(c)]
            pick = rng.permutation(idx_c)[:1]
            labeled_mask[pick] = True
        return labeled_mask

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

    labeled_mask[picks] = True
    return labeled_mask


def load_or_create_fixed_labeled_mask(
    y_id: np.ndarray,
    sample_index: pd.Index,
    save_dir: str,
    *,
    ratio: float,
    seed: int,
    stratified: bool,
    label_path: str,
) -> np.ndarray:
    fixed = load_fixed_exposed_mask_if_exists(save_dir, n_samples=len(sample_index))
    if fixed is not None:
        return fixed

    labeled_mask = pick_labeled_indices(
        y_id,
        ratio=ratio,
        seed=seed,
        stratified=stratified,
    )
    save_exposed_ids(
        save_dir=save_dir,
        sample_index=sample_index,
        labeled_mask=labeled_mask,
        seed=seed,
        labeled_ratio=ratio,
        stratified=stratified,
        label_path=label_path,
    )
    return labeled_mask


def load_labels_1col_numeric(label_path: str) -> np.ndarray:
    df = pd.read_csv(label_path)
    if df.shape[1] == 0:
        raise RuntimeError(f"[LABEL] no columns in {label_path}")
    col = None
    for cand in ["Label", "label", "LABEL"]:
        if cand in df.columns:
            col = cand
            break
    y = df[col].to_numpy() if col is not None else df.iloc[:, 0].to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    if np.isnan(y).any():
        bad = np.where(np.isnan(y))[0][:10]
        raise RuntimeError(f"[LABEL] NaN found in labels at rows {bad.tolist()} (check label csv)")
    return y.astype(np.int64).reshape(-1)


def remap_labels_to_0k(y: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[int, int]]]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    uniq = np.unique(y)
    if uniq.min() == 0 and np.array_equal(uniq, np.arange(len(uniq), dtype=np.int64)):
        return y, None
    mapping = {int(old): int(new) for new, old in enumerate(uniq.tolist())}
    y2 = np.vectorize(lambda v: mapping[int(v)])(y).astype(np.int64)
    return y2, mapping
