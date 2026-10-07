

import argparse
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import scanpy as sc
except Exception:
    sc = None



# Basic helpers


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def norm_gene(g: Any) -> str:
    if g is None:
        return ""
    g = str(g).strip().upper().replace('"', '').replace("'", "")
    g = re.sub(r"\s+", "", g)
    return g


def gene_family_key(g: Any) -> str:
    g = norm_gene(g)
    m = re.match(r"^([A-Z]{3,8})(?:\d+[A-Z]?)$", g)
    return m.group(1) if m else g


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def parse_float_list(s: Any, default: List[float]) -> List[float]:
    if s is None:
        return list(default)
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals if vals else list(default)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def matrix_stats(df: pd.DataFrame) -> Dict[str, Any]:
    arr = df.to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(arr)
    valid = arr[finite]
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "finite_fraction": float(finite.mean()) if arr.size else 1.0,
        "nonzero_fraction": float((np.abs(arr) > 0).mean()) if arr.size else 0.0,
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean()) if valid.size else None,
        "std": float(valid.std()) if valid.size else None,
    }


def compare_matrices(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "same_shape": tuple(before.shape) == tuple(after.shape),
        "same_index": before.index.equals(after.index),
        "same_columns": before.columns.equals(after.columns),
        "changed_fraction": 0.0,
        "mean_abs_diff": 0.0,
        "max_abs_diff": 0.0,
        "top_changed_genes": [],
    }
    if not out["same_shape"]:
        return out
    b = before.to_numpy(dtype=np.float32, copy=False)
    a = after.to_numpy(dtype=np.float32, copy=False)
    d = np.abs(a - b)
    out["changed_fraction"] = float((d > 1e-12).mean())
    out["mean_abs_diff"] = float(d.mean())
    out["max_abs_diff"] = float(d.max())
    per_gene = d.mean(axis=0)
    order = np.argsort(-per_gene)[:50]
    rows = []
    for j in order:
        if per_gene[j] <= 0:
            continue
        rows.append({
            "gene": str(before.columns[j]),
            "mean_abs_diff": float(per_gene[j]),
            "max_abs_diff": float(d[:, j].max()),
            "changed_samples": int((d[:, j] > 1e-12).sum()),
        })
    out["top_changed_genes"] = rows
    return out


# Data loading and alignment

def load_gene_list(path: str) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    if "gene_name" in df.columns:
        genes = df["gene_name"].tolist()
    else:
        genes = df.iloc[:, 0].tolist()
    out, seen = [], set()
    for g in genes:
        g = str(g).strip()
        if g and g not in seen:
            out.append(g)
            seen.add(g)
    return out


def read_expression(path: str) -> pd.DataFrame:
    path = str(path)
    low = path.lower()
    if low.endswith(".npz"):
        mat = scipy.sparse.load_npz(path)
        return pd.DataFrame(mat.toarray())
    if low.endswith(".npy"):
        return pd.DataFrame(np.load(path))
    if low.endswith(".h5ad"):
        if sc is None:
            raise RuntimeError("scanpy is not available, cannot read .h5ad")
        ad = sc.read_h5ad(path)
        idx = ad.obs_names.tolist()
        if "gene_name" in ad.var.columns:
            cols = ad.var["gene_name"].astype(str).tolist()
        else:
            cols = ad.var_names.astype(str).tolist()
        mat = ad.X.toarray() if issparse(ad.X) else ad.X
        return pd.DataFrame(mat, index=idx, columns=cols)
    return pd.read_csv(path, index_col=0)


def infer_orientation(df: pd.DataFrame, gene_list: List[str], overlap_min: int, ratio: float) -> pd.DataFrame:
    cols = set(str(x).strip() for x in df.columns)
    idxs = set(str(x).strip() for x in df.index)
    gl = set(str(x).strip() for x in gene_list)
    overlap_cols = len(cols & gl)
    overlap_idx = len(idxs & gl)
    print(f"[ALIGN] overlap(columns,gene_list)={overlap_cols} | overlap(index,gene_list)={overlap_idx}", flush=True)
    if overlap_cols >= overlap_min or overlap_idx >= overlap_min:
        if overlap_cols > overlap_idx * ratio:
            print("[ALIGN] use input as samples x genes", flush=True)
            return df
        if overlap_idx > overlap_cols * ratio:
            print("[ALIGN] transpose genes x samples -> samples x genes", flush=True)
            return df.T
    print("[ALIGN][WARN] cannot confidently infer orientation; keep input as-is", flush=True)
    return df


def dedup_columns(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(str)
    if not df.columns.duplicated().any():
        return df
    if policy == "first":
        return df.loc[:, ~df.columns.duplicated(keep="first")]
    if policy == "mean":
        return df.T.groupby(df.columns).mean().T
    return df.T.groupby(df.columns).sum().T


def clean_expression(df: pd.DataFrame, input_clip: float, dedup_policy: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index([str(x).strip() for x in df.columns])
    bad = (df.columns == "") | (pd.Series(df.columns).str.lower().values == "nan")
    if bad.any():
        df = df.loc[:, ~bad]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.clip(lower=-float(input_clip), upper=float(input_clip))
    return dedup_columns(df, dedup_policy)


def align_to_gene_list(df: pd.DataFrame, gene_list: List[str]) -> Tuple[pd.DataFrame, int, int, int]:
    df = df.copy()
    df.columns = df.columns.map(str)
    gl = [str(g) for g in gene_list]
    cols = set(df.columns)
    overlap = len(cols & set(gl))
    missing = [g for g in gl if g not in cols]
    extra = len(cols - set(gl))
    if missing:
        pad = pd.DataFrame(np.zeros((df.shape[0], len(missing)), dtype=np.float32), index=df.index, columns=missing)
        df = pd.concat([df, pad], axis=1)
    df = df[gl]
    return df.astype(np.float32), overlap, len(missing), extra


# Label and exposed split

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
                print(f"[LABEL] subtype_map loaded from {path}: {out}", flush=True)
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
# Expression standardization

def make_expr_z(expr: pd.DataFrame, input_clip: float = 100.0, transform: str = "auto") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    x = expr.to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(x)
    valid = x[finite] if finite.any() else np.array([], dtype=np.float32)

    if transform == "auto":
        neg_frac = float((valid < 0).mean()) if valid.size else 0.0
        min_val = float(valid.min()) if valid.size else 0.0
        transform_used = "raw_z" if (min_val < 0.0 or neg_frac > 0.001) else "log1p_z"
    else:
        transform_used = str(transform)

    if transform_used == "raw_z":
        y = x.astype(np.float32, copy=True)
    elif transform_used == "log1p_z":
        y = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
    else:
        raise ValueError(f"unknown expr_transform: {transform_used}")

    mu = y.mean(axis=0).astype(np.float32)
    sd = y.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    z = ((y - mu[None, :]) / sd[None, :]).astype(np.float32)
    z = np.clip(z, -float(input_clip), float(input_clip)).astype(np.float32)
    return y, z, mu, sd, transform_used


def inverse_z_to_expr(z_new: np.ndarray, ybase: np.ndarray, mu: np.ndarray, sd: np.ndarray, transform_used: str, args: argparse.Namespace) -> np.ndarray:
    y_new = z_new * sd[None, :] + mu[None, :]
    if bool(args.quantile_clip):
        lo = np.quantile(ybase, float(args.quantile_low), axis=0)
        hi = np.quantile(ybase, float(args.quantile_high), axis=0)
        span = np.maximum(hi - lo, 1e-6)
        lo = lo - float(args.quantile_margin) * span
        hi = hi + float(args.quantile_margin) * span
        y_new = np.minimum(np.maximum(y_new, lo[None, :]), hi[None, :])

    if transform_used == "raw_z":
        return y_new.astype(np.float32)
    y_new = np.maximum(y_new, 0.0)
    x_new = np.expm1(y_new).astype(np.float32)
    return np.maximum(x_new, 0.0)


# Local knowledge compression

def collect_genes_from_obj(obj: Any, gene_set: Optional[set] = None) -> set:
    if gene_set is None:
        gene_set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"gene", "genes", "target", "targets", "regulator", "top_genes", "marker_genes", "members_uniprot", "members", "gene_symbols"}:
                if isinstance(v, (list, tuple)):
                    for x in v:
                        gx = norm_gene(x)
                        if gx:
                            gene_set.add(gx)
                else:
                    gx = norm_gene(v)
                    if gx:
                        gene_set.add(gx)
            else:
                collect_genes_from_obj(v, gene_set)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            collect_genes_from_obj(x, gene_set)
    return gene_set


def iter_jsonl(path: Path, max_records: int) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def source_bucket(ch: Dict[str, Any]) -> str:
    s = str(ch.get("source", "")).lower()
    if "trrust" in s:
        return "TRRUST"
    if "panglao" in s:
        return "PanglaoDB"
    if "gtex" in s:
        return "GTEx"
    if "reactome" in s:
        return "Reactome"
    if "msig" in s:
        return "MSigDB"
    if "chea" in s:
        return "ChEA"
    return str(ch.get("source", "Other")) or "Other"


def load_local_chunks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.knowledge_path:
        path = Path(args.knowledge_path)
    else:
        path = Path(args.prior_root) / "qwen_rag" / "knowledge_chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"knowledge_chunks.jsonl not found: {path}")

    allowed = [x.strip().lower() for x in str(args.allowed_sources).split(",") if x.strip()]
    chunks = []
    for ch in iter_jsonl(path, int(args.max_knowledge_records)):
        src = str(ch.get("source", ""))
        if allowed:
            low_src = src.lower()
            if not any(a in low_src for a in allowed):
                continue
        genes = sorted(collect_genes_from_obj(ch.get("metadata", {})))
        if not genes:
            genes = sorted(collect_genes_from_obj(ch))
        ch["_genes"] = genes
        ch["_source_bucket"] = source_bucket(ch)
        ch["_search_text"] = f"{src} {ch.get('type','')} {ch.get('title','')} {ch.get('text','')}".upper()
        chunks.append(ch)
    print(f"[RETRIEVE] loaded local chunks={len(chunks)} from {path}", flush=True)
    return chunks


def compress_gene_evidence(candidate_genes: List[str], chunks: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    cset = {norm_gene(g) for g in candidate_genes if norm_gene(g)}
    evidence = {
        g: {
            "gene": g,
            "source_counts": Counter(),
            "titles": [],
            "total_hits": 0,
            "reactome_hits": 0,
            "trrust_hits": 0,
            "panglao_hits": 0,
            "gtex_hits": 0,
        } for g in cset
    }
    scored_chunks = []
    disease_terms = [t for t in re.split(r"[^A-Za-z0-9]+", str(args.disease_context).upper()) if len(t) >= 3]

    for ch in chunks:
        genes = {norm_gene(g) for g in ch.get("_genes", []) or []}
        overlap = sorted(cset & genes)
        if not overlap:
            continue
        bucket = source_bucket(ch)
        text = str(ch.get("_search_text", ""))
        disease_hit = sum(1 for t in disease_terms if t in text)
        density = len(overlap) / max(1, len(genes))
        score = 2.5 * len(overlap) + 1.0 * min(5, disease_hit) + 2.0 * density
        scored_chunks.append((score, bucket, overlap, ch))
        title = str(ch.get("title", "") or ch.get("type", ""))[:120]
        for g in overlap:
            ev = evidence[g]
            ev["source_counts"][bucket] += 1
            ev["total_hits"] += 1
            if bucket == "Reactome":
                ev["reactome_hits"] += 1
            elif bucket == "TRRUST":
                ev["trrust_hits"] += 1
            elif bucket == "PanglaoDB":
                ev["panglao_hits"] += 1
            elif bucket == "GTEx":
                ev["gtex_hits"] += 1
            if title and len(ev["titles"]) < 3:
                ev["titles"].append(title)

    for ev in evidence.values():
        ev["source_counts"] = dict(ev["source_counts"])
        ev["evidence_score"] = float(
            1.30 * ev["trrust_hits"]
            + 1.15 * ev["panglao_hits"]
            + 1.10 * ev["reactome_hits"]
            + 1.05 * ev["gtex_hits"]
            + 0.20 * ev["total_hits"]
        )

    # Balanced selected chunks for audit/debug only.
    scored_chunks.sort(key=lambda x: -x[0])
    selected = []
    bucket_counter = Counter()
    max_per_bucket = max(1, int(args.max_chunks_per_source))
    used_chars = 0
    for score, bucket, overlap, ch in scored_chunks:
        if bucket_counter[bucket] >= max_per_bucket:
            continue
        txt = str(ch.get("text", ""))
        add_chars = len(txt[: int(args.max_chars_per_chunk)])
        if selected and used_chars + add_chars > int(args.max_context_chars):
            break
        ch2 = dict(ch)
        ch2["_score"] = float(score)
        ch2["_source_bucket"] = bucket
        ch2["_overlap_genes"] = overlap[:30]
        selected.append(ch2)
        bucket_counter[bucket] += 1
        used_chars += add_chars
        if len(selected) >= int(args.max_chunks):
            break
    print(f"[RETRIEVE][QCMCE] selected_chunks={len(selected)} sources={dict(bucket_counter)}", flush=True)
    return evidence, selected


# Candidate marker programs

BROAD_OR_TECHNICAL_GENES = {
    "MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "CCNB1", "CCNB2", "CDK1", "AURKA", "AURKB", "UBE2C", "BIRC5",
    "FOS", "JUN", "JUNB", "DUSP1", "ATF3", "EGR1", "IER2",
    "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1",
    "ACTB", "GAPDH", "B2M", "MALAT1", "TPT1", "EEF1A1",
}


def is_broad_or_technical(g: str) -> bool:
    g = norm_gene(g)
    return (
        g in BROAD_OR_TECHNICAL_GENES
        or re.match(r"^(RPL|RPS)\d", g) is not None
        or g.startswith("MT-")
        or re.match(r"^MT[A-Z0-9]+", g) is not None
    )


def select_top_genes_by_score(genes: List[str], scores: np.ndarray, n: int, positive: bool = True, family_cap: int = 5) -> List[str]:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores if positive else scores)
    out = []
    fam_counter = Counter()
    for j in order:
        val = float(scores[j])
        if positive and val <= 0:
            continue
        if (not positive) and val >= 0:
            continue
        g = norm_gene(genes[j])
        if not g:
            continue
        fam = gene_family_key(g)
        if fam_counter[fam] >= int(family_cap):
            continue
        out.append(g)
        fam_counter[fam] += 1
        if len(out) >= int(n):
            break
    return out


def build_marker_candidate_bank(expr: pd.DataFrame, labels: pd.Series, exposed_mask: np.ndarray, subtype_map: Dict[int, str], chunks: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    ybase, z, mu, sd, transform_used = make_expr_z(expr, input_clip=float(args.factor_z_clip), transform=str(args.expr_transform))
    genes = [norm_gene(g) for g in expr.columns]
    gene_to_idx = {g: j for j, g in enumerate(genes)}
    y = labels.to_numpy(dtype=int)
    ex = np.asarray(exposed_mask, dtype=bool)
    var = z.var(axis=0)
    keep = var > 1e-8

    subtype_candidates = {}
    all_candidate_genes = set()
    class_effect_lookup = {}

    for c in sorted(set(y[ex].tolist())):
        cls = ex & (y == c)
        rest = ex & (y != c)
        min_exposed_per_class = int(getattr(args, "min_exposed_per_class", 3))
        if cls.sum() < min_exposed_per_class or rest.sum() < min_exposed_per_class:
            continue

        mean_c = z[cls].mean(axis=0)
        mean_r = z[rest].mean(axis=0)
        var_c = z[cls].var(axis=0)
        var_r = z[rest].var(axis=0)
        effect = mean_c - mean_r
        pooled = np.sqrt(0.5 * (var_c + var_r) + 1e-6)
        z_effect = (effect / pooled).astype(np.float32)
        z_effect = np.where(keep, z_effect, 0.0)
        class_effect_lookup[str(int(c))] = {genes[j]: float(z_effect[j]) for j in np.where(np.abs(z_effect) > 0)[0]}

        up = select_top_genes_by_score(genes, z_effect, int(args.candidate_genes_per_subtype), positive=True, family_cap=int(args.max_genes_per_family))
        down = select_top_genes_by_score(genes, z_effect, int(args.down_candidate_genes_per_subtype), positive=False, family_cap=int(args.max_genes_per_family))

        up_rows, down_rows = [], []
        for g in up:
            j = gene_to_idx[g]
            up_rows.append({
                "gene": g,
                "z_effect": float(z_effect[j]),
                "effect": float(effect[j]),
                "class_mean_z": float(mean_c[j]),
                "rest_mean_z": float(mean_r[j]),
                "broad_or_technical": bool(is_broad_or_technical(g)),
            })
            all_candidate_genes.add(g)
        for g in down:
            j = gene_to_idx[g]
            down_rows.append({
                "gene": g,
                "z_effect": float(z_effect[j]),
                "effect": float(effect[j]),
                "class_mean_z": float(mean_c[j]),
                "rest_mean_z": float(mean_r[j]),
                "broad_or_technical": bool(is_broad_or_technical(g)),
            })
            all_candidate_genes.add(g)

        subtype_candidates[str(int(c))] = {
            "class_id": int(c),
            "class_name": str(subtype_map.get(int(c), f"class_{int(c)}")),
            "exposed_class_n": int(cls.sum()),
            "exposed_rest_n": int(rest.sum()),
            "up_candidates": up_rows,
            "down_candidates": down_rows,
        }

    evidence, selected_chunks = compress_gene_evidence(sorted(all_candidate_genes), chunks, args)
    # Add evidence and redundancy.
    gene_class_hits = defaultdict(list)
    for cid, p in subtype_candidates.items():
        for r in p["up_candidates"]:
            gene_class_hits[r["gene"]].append(cid)

    for cid, p in subtype_candidates.items():
        for key in ["up_candidates", "down_candidates"]:
            for r in p[key]:
                ev = evidence.get(r["gene"], {})
                r["evidence_score"] = safe_float(ev.get("evidence_score", 0.0), 0.0)
                r["evidence_sources"] = ev.get("source_counts", {})
                r["evidence_titles"] = ev.get("titles", [])
                r["multi_class_up_count"] = len(set(gene_class_hits.get(r["gene"], [])))

    return {
        "z": z,
        "ybase": ybase,
        "mu": mu,
        "sd": sd,
        "expr_transform": transform_used,
        "gene_list": genes,
        "subtype_candidates": subtype_candidates,
        "gene_evidence": evidence,
        "selected_chunks": selected_chunks,
        "class_effect_lookup": class_effect_lookup,
        "label_protocol": {"exposed_n": int(ex.sum()), "use_exposed_labels": True},
    }


def prompt_bank(candidate_bank: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = {"subtypes": []}
    for cid, p in candidate_bank["subtype_candidates"].items():
        def rank_key(r):
            return (
                safe_float(r.get("evidence_score"), 0.0),
                abs(safe_float(r.get("z_effect"), 0.0)),
                -safe_float(r.get("multi_class_up_count"), 0.0),
            )
        up = sorted(p["up_candidates"], key=rank_key, reverse=True)[: int(args.prompt_candidates_per_subtype)]
        down = sorted(p["down_candidates"], key=rank_key, reverse=True)[: int(args.prompt_down_candidates_per_subtype)]
        # Keep compact fields only.
        def compact(rows):
            rr = []
            for r in rows:
                rr.append({
                    "gene": r["gene"],
                    "z_effect": round(safe_float(r.get("z_effect"), 0.0), 4),
                    "evidence_score": round(safe_float(r.get("evidence_score"), 0.0), 4),
                    "sources": r.get("evidence_sources", {}),
                    "multi_class_up_count": r.get("multi_class_up_count", 0),
                    "broad_or_technical": r.get("broad_or_technical", False),
                })
            return rr
        out["subtypes"].append({
            "class_id": str(p["class_id"]),
            "class_name": p["class_name"],
            "exposed_class_n": p["exposed_class_n"],
            "up_candidates": compact(up),
            "down_candidates": compact(down),
        })
    return out


# Qwen marker confirmation

def build_qwen_marker_prompt(candidate_bank: Dict[str, Any], args: argparse.Namespace) -> str:
    """Build a JSON prompt requesting one marker program per class ID."""
    class_ids = []
    for cid, p in candidate_bank.get("subtype_candidates", {}).items():
        class_ids.append({"class_id": str(p["class_id"]), "class_name": str(p["class_name"])})

    task_context = str(args.disease_context or args.dataset_name or args.dataset_key or "RNA expression clustering")
    dataset_mode = str(getattr(args, "dataset_mode", "auto") or "auto")
    class_context = str(getattr(args, "class_context", "") or "")
    example_class = class_ids[0] if class_ids else {"class_id": "0", "class_name": "class_0"}
    payload = {
        "task": "Select class-specific marker genes from candidate lists for RNA expression clustering before scFoundation.",
        "dataset": {
            "dataset_name": str(args.dataset_name),
            "dataset_key": str(args.dataset_key),
            "dataset_mode": dataset_mode,
            "task_context": task_context,
            "class_context": class_context,
        },
        "required_class_ids": class_ids,
        "hard_rules": [
            "Return exactly one JSON object and nothing else.",
            "The top-level JSON key must be subtype_marker_programs for backward compatibility.",
            "subtype_marker_programs must be a list with exactly one object for each required class_id.",
            "Every object must contain class_id, class_name, selected_up_genes, selected_down_genes, rejected_genes, reason.",
            "class_id must be copied exactly from required_class_ids, for example \"0\" rather than a class name.",
            "Use only gene symbols from the candidate lists for the same class_id.",
            "selected_up_genes must contain at least the requested minimum when possible.",
            "selected_down_genes may be empty.",
            "Do not output numeric weights, scores, expression values, coefficients, markdown, or explanations outside JSON.",
        ],
        "selection_policy": [
            "Prefer high z_effect genes with evidence sources.",
            "Prefer genes that distinguish the current class from other classes in this dataset.",
            "Avoid multi_class_up_count > 1 unless the gene is a biologically meaningful identity marker for the current class.",
            "Avoid broad_or_technical genes when possible.",
            "Preserve biologically meaningful class, cell-type, subtype, tissue-state, or disease-state identity markers even if pathway evidence is sparse.",
            "Use the dataset task_context and class_context rather than assuming BRCA or any single disease.",
        ],
        "output_example": {
            "subtype_marker_programs": [
                {
                    "class_id": str(example_class.get("class_id", "0")),
                    "class_name": str(example_class.get("class_name", "class_0")),
                    "selected_up_genes": ["GENE1", "GENE2"],
                    "selected_down_genes": [],
                    "rejected_genes": [],
                    "reason": "class-specific exposed markers with supporting evidence for the current dataset"
                }
            ],
            "notes": []
        },
        "INPUT_JSON": prompt_bank(candidate_bank, args),
    }
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(prompt) > int(args.llm_context_chars):
        old_up = args.prompt_candidates_per_subtype
        old_down = args.prompt_down_candidates_per_subtype
        args.prompt_candidates_per_subtype = max(int(getattr(args, "prompt_min_candidates_per_subtype", 20)), old_up // 2)
        args.prompt_down_candidates_per_subtype = max(int(getattr(args, "prompt_min_down_candidates_per_subtype", 5)), old_down // 2)
        payload["INPUT_JSON"] = prompt_bank(candidate_bank, args)
        args.prompt_candidates_per_subtype = old_up
        args.prompt_down_candidates_per_subtype = old_down
        prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "/no_think\n"
        "Return valid JSON only. Use exactly the schema in output_example. "
        "Top-level key: subtype_marker_programs. No markdown. No prose.\n"
        "INPUT_JSON:\n" + prompt
    )


def call_ollama(prompt: str, args: argparse.Namespace, *, force_json: bool = False) -> Dict[str, Any]:
    host = str(args.ollama_host).strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": str(args.qwen_model_name),
        "prompt": prompt,
        "system": "You are a JSON-only engine. Output exactly one valid JSON object. No markdown. No explanation. No <think>.",
        "stream": False,
        "options": {
            "temperature": float(args.temperature),
            "num_predict": int(args.num_predict),
            "num_ctx": int(getattr(args, "num_ctx", 32768)),
        },
    }
    if force_json:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=int(args.qwen_timeout_sec)) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
        response = str(obj.get("response", "") or "")
        thinking = str(obj.get("thinking", "") or "")
        # Prefer response; fall back to thinking and record used_field.
        raw_text = response.strip() if response.strip() else thinking.strip()
        used = "response" if response.strip() else "thinking"
        return {
            "ok": True,
            "raw_text": raw_text,
            "api_response": obj,
            "used_field": used,
            "response_len": len(response),
            "thinking_len": len(thinking),
            "force_json": bool(force_json),
            "error": "",
        }
    except Exception as e:
        return {"ok": False, "raw_text": "", "api_response": {}, "error": repr(e)}


def strip_thinking_and_fences(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
    text = text.strip()
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _try_json_loads_relaxed(candidate: str) -> Optional[dict]:
    candidate = str(candidate or "").strip()
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            return _try_json_loads_relaxed(obj)
    except Exception:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_first_json_object(text: str) -> Optional[dict]:
    text = strip_thinking_and_fences(text)
    direct = _try_json_loads_relaxed(text)
    if isinstance(direct, dict):
        return direct
    starts = [m.start() for m in re.finditer(r"\{", text)]
    candidates = []
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    candidates.sort(key=lambda c: ("subtype_marker_programs" not in c and "programs" not in c, -len(c)))
    for cand in candidates:
        obj = _try_json_loads_relaxed(cand)
        if isinstance(obj, dict):
            return obj
    return None


def repair_marker_json(raw_text: str, args: argparse.Namespace) -> Dict[str, Any]:
    repair_prompt = f"""
/no_think
Convert the following model output into exactly one valid JSON object.
No markdown. No explanation. No <think>.
The required top-level keys are subtype_marker_programs and notes.
Each subtype_marker_program must contain: class_id, class_name, selected_up_genes, selected_down_genes, rejected_genes, reason.
If the original output uses genes/core_genes/up_genes/markers, map them to selected_up_genes.
If it uses target_class_id, map it to class_id.
Preserve gene symbols exactly.

OUTPUT_TO_REPAIR:
{raw_text[:16000]}
""".strip()
    # Request JSON format when repairing model output.
    return call_ollama(repair_prompt, args, force_json=True)


def _flatten_marker_rows(raw: Any) -> List[Dict[str, Any]]:
    """Accept common JSON shapes produced by Qwen."""
    if not isinstance(raw, dict):
        return []
    rows = raw.get("subtype_marker_programs", None)
    if rows is None:
        rows = raw.get("programs", None)
    if rows is None:
        rows = raw.get("marker_programs", None)
    if rows is None:
        rows = raw.get("subtypes", None)
    if isinstance(rows, dict):
        out = []
        for k, v in rows.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("class_id", str(k))
                out.append(vv)
            elif isinstance(v, list):
                out.append({"class_id": str(k), "selected_up_genes": v})
        return out
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    # Some models return class IDs at top level.
    out = []
    for k, v in raw.items():
        if str(k).isdigit() and isinstance(v, (dict, list)):
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("class_id", str(k))
                out.append(vv)
            else:
                out.append({"class_id": str(k), "selected_up_genes": v})
    return out



# Adaptive prompt configuration

def infer_exposed_k(labels: pd.Series, exposed_mask: np.ndarray) -> int:
    y = labels.to_numpy(dtype=int)
    ex = np.asarray(exposed_mask, dtype=bool)
    return int(len(sorted(set(y[ex].tolist()))))


def configure_qwen_universal_adaptive_profile(args: argparse.Namespace, k_classes: int) -> None:
    """Configure prompt limits from class count without using clustering metrics."""
    if int(getattr(args, "auto_quickcheck_clusters", 0)) <= 0:
        args.auto_quickcheck_clusters = int(k_classes)

    old = {
        "candidate_genes_per_subtype": int(args.candidate_genes_per_subtype),
        "down_candidate_genes_per_subtype": int(args.down_candidate_genes_per_subtype),
        "prompt_candidates_per_subtype": int(args.prompt_candidates_per_subtype),
        "prompt_down_candidates_per_subtype": int(args.prompt_down_candidates_per_subtype),
        "qwen_selected_up_genes": int(args.qwen_selected_up_genes),
        "qwen_selected_down_genes": int(args.qwen_selected_down_genes),
        "qwen_min_selected_up_genes": int(args.qwen_min_selected_up_genes),
        "prompt_competing_top_genes": int(getattr(args, "prompt_competing_top_genes", 8)),
    }

    if not bool(getattr(args, "qwen_adaptive_prompt_profile", True)):
        args._qwen_adaptive_profile_meta = {"enabled": False, "k_classes": int(k_classes), "original": old}
        return

    profile = str(getattr(args, "qwen_prompt_profile", "auto")).strip().lower()
    if profile == "auto":
        if int(k_classes) <= 6:
            profile = "small_k_strong"
        elif int(k_classes) <= 15:
            profile = "medium_k_compact"
        else:
            profile = "large_k_ultracompact"

    if profile == "small_k_strong":
        args.candidate_genes_per_subtype = 180
        args.down_candidate_genes_per_subtype = 50
        args.prompt_candidates_per_subtype = 80
        args.prompt_down_candidates_per_subtype = 25
        args.qwen_selected_up_genes = 40
        args.qwen_selected_down_genes = 10
        args.qwen_min_selected_up_genes = 10
        args.prompt_competing_top_genes = 8
        args.prompt_min_candidates_per_subtype = 35
        args.prompt_min_down_candidates_per_subtype = 10
    elif profile == "medium_k_compact":
        args.candidate_genes_per_subtype = 120
        args.down_candidate_genes_per_subtype = 25
        args.prompt_candidates_per_subtype = 30
        args.prompt_down_candidates_per_subtype = 8
        args.qwen_selected_up_genes = 20
        args.qwen_selected_down_genes = 3
        args.qwen_min_selected_up_genes = 6
        args.prompt_competing_top_genes = 6
        args.prompt_min_candidates_per_subtype = 15
        args.prompt_min_down_candidates_per_subtype = 4
        args.num_predict = min(int(args.num_predict), 3000)
    elif profile == "large_k_ultracompact":
        args.candidate_genes_per_subtype = 90
        args.down_candidate_genes_per_subtype = 20
        args.prompt_candidates_per_subtype = 20
        args.prompt_down_candidates_per_subtype = 5
        args.qwen_selected_up_genes = 15
        args.qwen_selected_down_genes = 2
        args.qwen_min_selected_up_genes = 5
        args.prompt_competing_top_genes = 4
        args.prompt_min_candidates_per_subtype = 10
        args.prompt_min_down_candidates_per_subtype = 3
        args.num_predict = min(int(args.num_predict), 2500)
    else:
        raise ValueError(f"unknown qwen_prompt_profile={profile}")

    if int(getattr(args, "prompt_max_competing_classes", 0)) <= 0:
        args.prompt_max_competing_classes = max(0, int(k_classes) - 1)

    args._qwen_adaptive_profile_meta = {
        "enabled": True,
        "resolved_profile": profile,
        "k_classes": int(k_classes),
        "rule": "small_k_strong if K<=6; medium_k_compact if 7<=K<=15; large_k_ultracompact if K>15",
        "original": old,
        "resolved": {
            "candidate_genes_per_subtype": int(args.candidate_genes_per_subtype),
            "down_candidate_genes_per_subtype": int(args.down_candidate_genes_per_subtype),
            "prompt_candidates_per_subtype": int(args.prompt_candidates_per_subtype),
            "prompt_down_candidates_per_subtype": int(args.prompt_down_candidates_per_subtype),
            "qwen_selected_up_genes": int(args.qwen_selected_up_genes),
            "qwen_selected_down_genes": int(args.qwen_selected_down_genes),
            "qwen_min_selected_up_genes": int(args.qwen_min_selected_up_genes),
            "prompt_competing_top_genes": int(getattr(args, "prompt_competing_top_genes", 0)),
        },
    }
    print(f"[QWEN-PROTOCOL] adaptive_profile={profile} K={k_classes} resolved={args._qwen_adaptive_profile_meta['resolved']}", flush=True)


def _compact_candidate_rows_for_prompt(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    def rank_key(r):
        return (
            safe_float(r.get("evidence_score"), 0.0),
            abs(safe_float(r.get("z_effect"), 0.0)),
            -safe_float(r.get("multi_class_up_count"), 0.0),
        )
    rows = sorted(rows, key=rank_key, reverse=True)[: int(n)]
    out = []
    for r in rows:
        out.append({
            "gene": r["gene"],
            "z_effect": round(safe_float(r.get("z_effect"), 0.0), 4),
            "evidence_score": round(safe_float(r.get("evidence_score"), 0.0), 4),
            "sources": r.get("evidence_sources", {}),
            "multi_class_up_count": r.get("multi_class_up_count", 0),
            "broad_or_technical": r.get("broad_or_technical", False),
        })
    return out


def prompt_bank_for_one_class(candidate_bank: Dict[str, Any], args: argparse.Namespace, target_cid: str) -> Dict[str, Any]:
    p = candidate_bank["subtype_candidates"][str(target_cid)]
    target = {
        "class_id": str(p["class_id"]),
        "class_name": str(p["class_name"]),
        "exposed_class_n": int(p["exposed_class_n"]),
        "up_candidates": _compact_candidate_rows_for_prompt(p["up_candidates"], int(args.prompt_candidates_per_subtype)),
        "down_candidates": _compact_candidate_rows_for_prompt(p["down_candidates"], int(args.prompt_down_candidates_per_subtype)),
    }

    competing = []
    max_classes = int(getattr(args, "prompt_max_competing_classes", 0))
    if max_classes <= 0:
        max_classes = max(0, len(candidate_bank.get("subtype_candidates", {})) - 1)
    top_genes = int(getattr(args, "prompt_competing_top_genes", 6))
    for cid, q in candidate_bank.get("subtype_candidates", {}).items():
        if str(cid) == str(target_cid):
            continue
        if len(competing) >= max_classes:
            break
        rows = _compact_candidate_rows_for_prompt(q["up_candidates"], top_genes)
        competing.append({
            "class_id": str(q["class_id"]),
            "class_name": str(q["class_name"]),
            "top_up_genes": [r["gene"] for r in rows],
        })
    return {"target_class": target, "competing_class_summaries": competing}


def build_qwen_marker_prompt_for_class(candidate_bank: Dict[str, Any], args: argparse.Namespace, target_cid: str) -> str:
    p = candidate_bank["subtype_candidates"][str(target_cid)]
    task_context = str(args.disease_context or args.dataset_name or args.dataset_key or "RNA expression clustering")
    dataset_mode = str(getattr(args, "dataset_mode", "auto") or "auto")
    class_context = str(getattr(args, "class_context", "") or "")
    payload = {
        "task": "Select marker genes for one target class from candidate lists for RNA expression clustering before scFoundation.",
        "dataset": {
            "dataset_name": str(args.dataset_name),
            "dataset_key": str(args.dataset_key),
            "dataset_mode": dataset_mode,
            "task_context": task_context,
            "class_context": class_context,
        },
        "target_class_id": str(p["class_id"]),
        "target_class_name": str(p["class_name"]),
        "hard_rules": [
            "Return exactly one JSON object and nothing else.",
            "The top-level JSON key must be subtype_marker_programs for backward compatibility.",
            "subtype_marker_programs must be a list with exactly one object for the target_class_id only.",
            "The object must contain class_id, class_name, selected_up_genes, selected_down_genes, rejected_genes, reason.",
            "class_id must equal target_class_id exactly.",
            "Use only gene symbols from target_class.up_candidates for selected_up_genes.",
            "Use only gene symbols from target_class.down_candidates for selected_down_genes.",
            "Do not output numeric weights, scores, expression values, coefficients, markdown, or explanations outside JSON.",
        ],
        "selection_policy": [
            "Prefer high z_effect genes with evidence sources.",
            "Prefer genes that distinguish the target class from competing_class_summaries.",
            "Avoid broad_or_technical genes when possible.",
            "Avoid genes that are top markers of many competing classes unless they are biologically necessary identity markers.",
            "Use the dataset task_context and class_context rather than assuming BRCA or any single disease.",
        ],
        "output_example": {
            "subtype_marker_programs": [
                {
                    "class_id": str(p["class_id"]),
                    "class_name": str(p["class_name"]),
                    "selected_up_genes": ["GENE1", "GENE2"],
                    "selected_down_genes": [],
                    "rejected_genes": [],
                    "reason": "class-specific exposed markers with supporting evidence for the current dataset"
                }
            ],
            "notes": []
        },
        "INPUT_JSON": prompt_bank_for_one_class(candidate_bank, args, str(target_cid)),
    }
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    prefix = "/no_think\n" if bool(getattr(args, "qwen_disable_thinking", True)) else ""
    return (
        prefix
        + "Return valid JSON only. Use exactly the schema in output_example. "
        + "Top-level key: subtype_marker_programs. No markdown. No prose.\n"
        + "INPUT_JSON:\n" + prompt
    )


def extract_marker_json_object(text: str) -> Optional[dict]:
    """Parse model output; accept either a dict or a top-level list of programs."""
    obj = extract_first_json_object(text)
    if isinstance(obj, dict):
        return obj
    stripped = strip_thinking_and_fences(text)
    try:
        arr = json.loads(stripped)
        if isinstance(arr, list):
            return {"subtype_marker_programs": arr, "notes": ["top_level_list_wrapped"]}
    except Exception:
        pass
    # Try extracting first JSON array.
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(stripped[start:end+1])
            if isinstance(arr, list):
                return {"subtype_marker_programs": arr, "notes": ["embedded_list_wrapped"]}
        except Exception:
            pass
    return None


def deconflict_marker_programs(marker_programs: Dict[str, Any], candidate_bank: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if not bool(getattr(args, "qwen_deconflict_markers", True)):
        return marker_programs
    programs = [dict(p) for p in marker_programs.get("subtype_marker_programs", [])]
    class_effect = candidate_bank.get("class_effect_lookup", {}) or {}
    gene_owner = {}
    for pi, prog in enumerate(programs):
        cid = str(prog.get("class_id", ""))
        for g in list(prog.get("selected_up_genes", []) or []):
            g = norm_gene(g)
            val = abs(safe_float(class_effect.get(cid, {}).get(g, 0.0), 0.0))
            if g not in gene_owner or val > gene_owner[g][1]:
                gene_owner[g] = (pi, val)
    for pi, prog in enumerate(programs):
        cid = str(prog.get("class_id", ""))
        kept, seen = [], set()
        for g in list(prog.get("selected_up_genes", []) or []):
            g = norm_gene(g)
            if not g or g in seen:
                continue
            if gene_owner.get(g, (None,))[0] == pi:
                kept.append(g); seen.add(g)
        if bool(getattr(args, "qwen_deconflict_refill", True)) and cid in candidate_bank.get("subtype_candidates", {}):
            target_n = int(getattr(args, "qwen_min_selected_up_genes", 6))
            max_n = int(getattr(args, "qwen_selected_up_genes", 20))
            for cand in candidate_bank["subtype_candidates"][cid].get("up_candidates", []):
                g = norm_gene(cand.get("gene", ""))
                if not g or g in seen or bool(cand.get("broad_or_technical", False)):
                    continue
                if g in gene_owner and gene_owner[g][0] != pi:
                    continue
                kept.append(g); seen.add(g)
                if len(kept) >= max(target_n, min(max_n, target_n)):
                    break
        prog["selected_up_genes"] = kept[: int(getattr(args, "qwen_selected_up_genes", 20))]
    out = dict(marker_programs)
    out["subtype_marker_programs"] = programs
    out.setdefault("notes", [])
    if isinstance(out["notes"], list):
        out["notes"].append("deterministic_deconflict_applied")
    return out


def _coerce_one_class_marker_program(parsed: Any, candidate_bank: Dict[str, Any], args: argparse.Namespace, cid: str) -> Dict[str, Any]:
    old_override = getattr(args, "_qwen_min_programs_override", None)
    setattr(args, "_qwen_min_programs_override", 1)
    try:
        one = coerce_marker_programs(parsed, candidate_bank, args)
        rows = [r for r in one.get("subtype_marker_programs", []) if str(r.get("class_id", "")) == str(cid)]
        if not rows:
            raise RuntimeError("no usable marker program for target class")
        return {"subtype_marker_programs": rows[:1], "notes": one.get("notes", [])}
    finally:
        setattr(args, "_qwen_min_programs_override", old_override)


def generate_marker_programs_with_qwen(candidate_bank: Dict[str, Any], args: argparse.Namespace, out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], str, str, str]:
    art = out_dir / "qwen_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    class_ids = sorted(candidate_bank.get("subtype_candidates", {}).keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x))

    mode = str(getattr(args, "qwen_generation_mode", "auto")).strip().lower()
    if mode not in {"auto", "single_call", "per_class"}:
        mode = "auto"
    if mode == "auto":
        mode = "single_call" if len(class_ids) <= int(getattr(args, "qwen_single_call_max_k", 6)) else "per_class"

    if mode == "single_call":
        prompt = build_qwen_marker_prompt(candidate_bank, args)
        if len(prompt) > int(getattr(args, "qwen_prompt_budget_chars", 60000)):
            print(f"[QWEN-PROTOCOL] single_call prompt_len={len(prompt)} exceeds budget; switch to per_class", flush=True)
            mode = "per_class"
        else:
            (art / "qwen_marker_prompt.txt").write_text(prompt, encoding="utf-8")
            llm = call_ollama(prompt, args, force_json=bool(args.force_json))
            raw_text = llm.get("raw_text", "")
            (art / "qwen_raw_response.txt").write_text(raw_text, encoding="utf-8")
            print(
                f"[QWEN] mode=single_call ok={llm.get('ok')} used_field={llm.get('used_field')} "
                f"response_len={llm.get('response_len')} thinking_len={llm.get('thinking_len')}",
                flush=True,
            )
            parsed = extract_marker_json_object(raw_text) if llm.get("ok") else None
            if parsed is None and bool(args.repair_on_parse_fail) and raw_text:
                rep = repair_marker_json(raw_text, args)
                print(
                    f"[QWEN][REPAIR] ok={rep.get('ok')} used_field={rep.get('used_field')} "
                    f"response_len={rep.get('response_len')} thinking_len={rep.get('thinking_len')}", flush=True,
                )
                if rep.get("ok"):
                    parsed = extract_marker_json_object(rep.get("raw_text", ""))
                    llm["repair"] = rep
            marker_programs = coerce_marker_programs(parsed, candidate_bank, args)
            args._qwen_generation_mode_resolved = "single_call"
            return marker_programs, llm, "qwen_valid_single_call", prompt, raw_text

    # Generate marker programs separately for each target class.
    per_dir = art / "per_class_qwen"
    per_dir.mkdir(parents=True, exist_ok=True)
    merged_rows = []
    llm_summary = {"ok": True, "mode": "per_class", "classes": []}
    prompts_for_report, raws_for_report = [], []
    base_up = int(args.prompt_candidates_per_subtype)
    base_down = int(args.prompt_down_candidates_per_subtype)
    base_comp = int(getattr(args, "prompt_competing_top_genes", 6))
    max_retries = max(0, int(getattr(args, "qwen_class_max_retries", 2)))
    retry_factor = float(getattr(args, "qwen_class_retry_compact_factor", 0.5))

    for cid in class_ids:
        class_ok = False
        last_error = None
        last_llm = None
        for attempt in range(max_retries + 1):
            scale = 1.0 if attempt == 0 else (retry_factor ** attempt)
            args.prompt_candidates_per_subtype = max(int(getattr(args, "prompt_min_candidates_per_subtype", 10)), int(round(base_up * scale)))
            args.prompt_down_candidates_per_subtype = max(int(getattr(args, "prompt_min_down_candidates_per_subtype", 3)), int(round(base_down * scale)))
            args.prompt_competing_top_genes = max(2, int(round(base_comp * scale)))
            prompt = build_qwen_marker_prompt_for_class(candidate_bank, args, str(cid))
            (per_dir / f"class_{cid}_attempt{attempt}_prompt.txt").write_text(prompt, encoding="utf-8")
            llm = call_ollama(prompt, args, force_json=bool(args.force_json))
            raw_text = llm.get("raw_text", "")
            last_llm = llm
            (per_dir / f"class_{cid}_attempt{attempt}_raw_response.txt").write_text(raw_text, encoding="utf-8")
            print(
                f"[QWEN][CLASS {cid}][TRY {attempt}] ok={llm.get('ok')} used_field={llm.get('used_field')} "
                f"response_len={llm.get('response_len')} thinking_len={llm.get('thinking_len')} prompt_len={len(prompt)}",
                flush=True,
            )
            parsed = extract_marker_json_object(raw_text) if llm.get("ok") else None
            if parsed is None and bool(args.repair_on_parse_fail) and raw_text:
                rep = repair_marker_json(raw_text, args)
                print(
                    f"[QWEN][CLASS {cid}][TRY {attempt}][REPAIR] ok={rep.get('ok')} used_field={rep.get('used_field')} "
                    f"response_len={rep.get('response_len')} thinking_len={rep.get('thinking_len')}", flush=True,
                )
                if rep.get("ok"):
                    parsed = extract_marker_json_object(rep.get("raw_text", ""))
                    llm["repair"] = rep
            try:
                one = _coerce_one_class_marker_program(parsed, candidate_bank, args, str(cid))
                merged_rows.extend(one.get("subtype_marker_programs", [])[:1])
                class_ok = True
                last_error = None
                llm_summary["classes"].append({
                    "class_id": str(cid), "status": "ok", "attempt": int(attempt),
                    "prompt_candidates_per_subtype": int(args.prompt_candidates_per_subtype),
                    "prompt_down_candidates_per_subtype": int(args.prompt_down_candidates_per_subtype),
                    "prompt_competing_top_genes": int(args.prompt_competing_top_genes),
                    "ok": bool(llm.get("ok")), "used_field": llm.get("used_field", ""),
                    "response_len": int(llm.get("response_len", 0) or 0), "thinking_len": int(llm.get("thinking_len", 0) or 0),
                })
                prompts_for_report.append(f"### CLASS {cid} TRY {attempt}\n" + prompt[:6000])
                raws_for_report.append(f"### CLASS {cid} TRY {attempt}\n" + raw_text[:6000])
                break
            except Exception as e:
                last_error = repr(e)
                save_json({"class_id": str(cid), "attempt": int(attempt), "error": last_error, "llm": llm}, per_dir / f"class_{cid}_attempt{attempt}_failure.json")
        # Restore prompt limits after processing each class.
        args.prompt_candidates_per_subtype = base_up
        args.prompt_down_candidates_per_subtype = base_down
        args.prompt_competing_top_genes = base_comp
        if not class_ok:
            llm_summary["ok"] = False
            llm_summary["classes"].append({"class_id": str(cid), "status": f"fail:{last_error}", "llm": last_llm})
            if bool(getattr(args, "qwen_per_class_strict", True)):
                raise RuntimeError(f"per-class Qwen marker parsing failed for class {cid}: {last_error}")

    marker_programs = {"subtype_marker_programs": merged_rows, "notes": ["qwen_generation_mode=per_class"]}
    marker_programs = deconflict_marker_programs(marker_programs, candidate_bank, args)
    marker_programs = coerce_marker_programs(marker_programs, candidate_bank, args)
    prompt_report = "\n\n".join(prompts_for_report)
    raw_report = "\n\n".join(raws_for_report)
    (art / "qwen_marker_prompt.txt").write_text(prompt_report, encoding="utf-8")
    (art / "qwen_raw_response.txt").write_text(raw_report, encoding="utf-8")
    args._qwen_generation_mode_resolved = "per_class"
    return marker_programs, llm_summary, "qwen_valid_per_class", prompt_report, raw_report

def coerce_marker_programs(raw: Any, candidate_bank: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("marker program output is not a dict")
    rows = _flatten_marker_rows(raw)
    if not isinstance(rows, list):
        raise RuntimeError("subtype_marker_programs is not a list")

    valid_by_class = {}
    up_by_class = {}
    down_by_class = {}
    candidate_by_class = {}
    name_to_cid = {}
    for cid, p in candidate_bank["subtype_candidates"].items():
        cid_str = str(p["class_id"])
        up = {norm_gene(r["gene"]) for r in p["up_candidates"]}
        down = {norm_gene(r["gene"]) for r in p["down_candidates"]}
        valid_by_class[cid_str] = up | down
        up_by_class[cid_str] = up
        down_by_class[cid_str] = down
        candidate_by_class[cid_str] = p
        name_to_cid[norm_gene(p["class_name"])] = cid_str
        name_to_cid[str(p["class_name"]).strip().lower()] = cid_str

    def infer_cid(r: Dict[str, Any]) -> str:
        cid = str(r.get("class_id", r.get("target_class_id", r.get("label", "")))).strip()
        if cid in valid_by_class:
            return cid
        cname = str(r.get("class_name", r.get("target_class_name", r.get("subtype", r.get("name", ""))))).strip()
        if cname.lower() in name_to_cid:
            return name_to_cid[cname.lower()]
        if norm_gene(cname) in name_to_cid:
            return name_to_cid[norm_gene(cname)]
        # Try class name embedded in program_id.
        pid = str(r.get("program_id", "")).lower()
        for k, v in name_to_cid.items():
            if isinstance(k, str) and k and k.lower() in pid:
                return v
        return ""

    def list_from_fields(r: Dict[str, Any], fields: List[str]) -> List[Any]:
        vals = []
        for f in fields:
            v = r.get(f, None)
            if isinstance(v, list):
                vals.extend(v)
            elif isinstance(v, str):
                # Support comma-separated or whitespace gene lists.
                vals.extend(re.split(r"[,;\s]+", v.strip()))
        return vals

    def clean_genes(xs, valid, max_n):
        out, seen = [], set()
        if not isinstance(xs, list):
            return out
        for x in xs:
            # If item is {gene: ...}, accept it.
            if isinstance(x, dict):
                x = x.get("gene", x.get("symbol", ""))
            g = norm_gene(x)
            if g and g in valid and g not in seen:
                out.append(g)
                seen.add(g)
            if len(out) >= max_n:
                break
        return out

    programs = []
    seen_class = set()
    parse_debug = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = infer_cid(r)
        if cid not in valid_by_class or cid in seen_class:
            parse_debug.append({"raw_keys": list(r.keys()), "cid": cid, "status": "bad_or_duplicate_class"})
            continue
        p = candidate_by_class[cid]
        up_fields = [
            "selected_up_genes", "up_genes", "selected_genes", "genes", "marker_genes",
            "markers", "core_genes", "support_genes", "positive_genes", "high_genes"
        ]
        down_fields = ["selected_down_genes", "down_genes", "negative_genes", "low_genes"]
        reject_fields = ["rejected_genes", "forbidden_genes", "excluded_genes"]
        up_raw = list_from_fields(r, up_fields)
        down_raw = list_from_fields(r, down_fields)
        rej_raw = list_from_fields(r, reject_fields)

        up = clean_genes(up_raw, up_by_class[cid], int(args.qwen_selected_up_genes))
        # If Qwen returned valid subtype candidates in a generic genes field that include down genes,
        # keep only up candidates for selected_up_genes and put down candidates in down.
        down = clean_genes(down_raw, down_by_class[cid], int(args.qwen_selected_down_genes))
        if not down:
            down = clean_genes(up_raw, down_by_class[cid], int(args.qwen_selected_down_genes))
        rej = clean_genes(rej_raw, valid_by_class[cid], 200)

        # Fill missing up genes from ranked candidates and record qwen_partial_fill.
        qwen_original_up_n = len(up)
        if 0 < len(up) < int(args.qwen_min_selected_up_genes) and bool(getattr(args, "qwen_allow_partial_fill", True)):
            for cand in p["up_candidates"]:
                g = norm_gene(cand["gene"])
                if g not in up and g not in rej and not bool(cand.get("broad_or_technical", False)):
                    up.append(g)
                if len(up) >= int(args.qwen_min_selected_up_genes):
                    break

        if len(up) < int(args.qwen_min_selected_up_genes):
            parse_debug.append({
                "class_id": cid,
                "raw_keys": list(r.keys()),
                "valid_up_from_qwen": qwen_original_up_n,
                "valid_down": len(down),
                "status": "too_few_up_genes"
            })
            continue
        programs.append({
            "class_id": cid,
            "class_name": str(r.get("class_name", p["class_name"]))[:80],
            "selected_up_genes": up,
            "selected_down_genes": [g for g in down if g not in set(up)],
            "rejected_genes": rej,
            "reason": str(r.get("reason", ""))[:300],
            "qwen_original_valid_up_genes": int(qwen_original_up_n),
            "qwen_partial_fill": bool(qwen_original_up_n < len(up)),
        })
        seen_class.add(cid)

    override_min = getattr(args, "_qwen_min_programs_override", None)
    if override_min is not None:
        min_programs = int(override_min)
    else:
        min_programs = max(2, len(candidate_by_class) // 2)
    if len(programs) < min_programs:
        raise RuntimeError(
            f"too few usable Qwen marker programs: {len(programs)}/{len(candidate_by_class)}; "
            f"debug={parse_debug[:5]}"
        )
    return {
        "subtype_marker_programs": programs,
        "notes": raw.get("notes", []) if isinstance(raw.get("notes", []), list) else [],
        "parse_debug": parse_debug,
        "qwen_partial_fill_program_count": int(sum(1 for p in programs if p.get("qwen_partial_fill"))),
    }


def rule_marker_programs(candidate_bank: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    programs = []
    for cid, p in candidate_bank["subtype_candidates"].items():
        rows = [r for r in p["up_candidates"] if not bool(r.get("broad_or_technical", False))]
        rows = sorted(rows, key=lambda r: (safe_float(r.get("evidence_score"), 0.0), safe_float(r.get("z_effect"), 0.0)), reverse=True)
        up = [r["gene"] for r in rows[: int(args.qwen_selected_up_genes)]]
        down_rows = sorted(p["down_candidates"], key=lambda r: abs(safe_float(r.get("z_effect"), 0.0)), reverse=True)
        down = [r["gene"] for r in down_rows[: int(args.qwen_selected_down_genes)]]
        if len(up) < int(args.qwen_min_selected_up_genes):
            continue
        programs.append({
            "class_id": str(p["class_id"]),
            "class_name": p["class_name"],
            "selected_up_genes": up,
            "selected_down_genes": down,
            "rejected_genes": [],
            "reason": "rule fallback from exposed markers; not Qwen-confirmed",
        })
    return {"subtype_marker_programs": programs, "notes": ["rule_fallback_marker_programs"]}


# Marker contrast enhancement

def build_marker_arrays(marker_programs: Dict[str, Any], candidate_bank: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    genes = [norm_gene(g) for g in candidate_bank["gene_list"]]
    gene_to_idx = {g: j for j, g in enumerate(genes)}
    class_effect = candidate_bank["class_effect_lookup"]
    out = {}
    for p in marker_programs.get("subtype_marker_programs", []):
        cid = str(p["class_id"])
        up = [norm_gene(g) for g in p.get("selected_up_genes", []) if norm_gene(g) in gene_to_idx]
        down = [norm_gene(g) for g in p.get("selected_down_genes", []) if norm_gene(g) in gene_to_idx]
        signed = []
        for g in up:
            eff = abs(safe_float(class_effect.get(cid, {}).get(g, 1.0), 1.0))
            signed.append((gene_to_idx[g], g, +1.0, eff))
        for g in down:
            eff = abs(safe_float(class_effect.get(cid, {}).get(g, -1.0), 1.0))
            signed.append((gene_to_idx[g], g, -1.0, eff))
        if signed:
            out[cid] = {
                "class_id": cid,
                "class_name": p.get("class_name", f"class_{cid}"),
                "signed_genes": signed,
                "up_genes": up,
                "down_genes": down,
                "reason": p.get("reason", ""),
            }
    return out, gene_to_idx


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return ((x - med) / (1.4826 * mad + 1e-6)).astype(np.float32)


def compute_marker_scores(z: np.ndarray, marker_arrays: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    cids = sorted(marker_arrays.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x))
    scores = np.zeros((z.shape[0], len(cids)), dtype=np.float32)
    for k, cid in enumerate(cids):
        genes = marker_arrays[cid]["signed_genes"]
        if not genes:
            continue
        vals = np.zeros(z.shape[0], dtype=np.float32)
        denom = 0.0
        for j, _, sign, eff in genes:
            w = min(5.0, max(0.25, float(eff)))
            vals += float(sign) * float(w) * z[:, j].astype(np.float32)
            denom += abs(w)
        vals = vals / max(1e-6, denom)
        scores[:, k] = robust_z(vals)
    return scores, cids


def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) / max(1e-6, float(temp))
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _sigmoid_scalar(x: float, temp: float = 1.0) -> float:
    """Stable scalar sigmoid used for soft prior reliability."""
    temp = max(1e-6, float(temp))
    x = float(x) / temp
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _row_normalize_nonnegative(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    a = np.maximum(a, 0.0)
    s = a.sum(axis=1, keepdims=True)
    return a / np.maximum(float(eps), s)


def build_context_adaptive_module_prior(
    expr: pd.DataFrame,
    candidate_bank: Dict[str, Any],
    marker_programs: Dict[str, Any],
    args: argparse.Namespace,
    labels: Optional[pd.Series] = None,
    exposed_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build module activities and reliability-scaled priors from marker programs.

    Expression data and exposed labels determine module reliability.
    """
    marker_arrays, _ = build_marker_arrays(marker_programs, candidate_bank)
    if not marker_arrays:
        raise RuntimeError("no usable marker arrays for context-adaptive prior")

    z = candidate_bank["z"].astype(np.float32)
    scores, cids = compute_marker_scores(z, marker_arrays)
    raw_probs = softmax(scores, temp=float(args.prior_assignment_temperature))

    n, k = scores.shape
    yy = labels.to_numpy(dtype=int) if labels is not None else None
    ex = np.asarray(exposed_mask, dtype=bool) if exposed_mask is not None else np.zeros(n, dtype=bool)
    gene_evidence = candidate_bank.get("gene_evidence", {}) or {}

    reliability = np.zeros(k, dtype=np.float32)
    module_rows: List[Dict[str, Any]] = []

    # Helper: evidence support among Qwen-confirmed genes.
    def evidence_mean_for_program(cid: str) -> float:
        vals = []
        for _, g, _, _ in marker_arrays[cid].get("signed_genes", []):
            vals.append(safe_float(gene_evidence.get(g, {}).get("evidence_score", 0.0), 0.0))
        return float(np.mean(vals)) if vals else 0.0

    for kk, cid in enumerate(cids):
        class_name = str(marker_arrays[cid].get("class_name", f"class_{cid}"))
        if yy is None or ex.sum() == 0 or not str(cid).lstrip("-").isdigit():
            module_effect = 0.0
            exposed_top1 = 0.0
            exposed_margin = 0.0
            cls_n = 0
            rest_n = 0
        else:
            cid_int = int(cid)
            cls = ex & (yy == cid_int)
            rest = ex & (yy != cid_int)
            cls_n = int(cls.sum())
            rest_n = int(rest.sum())
            if cls_n >= int(args.min_exposed_per_class) and rest_n >= int(args.min_exposed_per_class):
                s_cls = scores[cls, kk]
                s_rest = scores[rest, kk]
                pooled = math.sqrt(0.5 * (float(np.var(s_cls)) + float(np.var(s_rest))) + 1e-6)
                module_effect = (float(np.mean(s_cls)) - float(np.mean(s_rest))) / max(1e-6, pooled)
                top = np.argmax(raw_probs[cls], axis=1) if cls_n else np.array([], dtype=int)
                exposed_top1 = float(np.mean(top == kk)) if top.size else 0.0
                sorted_probs = np.sort(raw_probs[cls], axis=1) if cls_n else np.zeros((0, k), dtype=np.float32)
                exposed_margin = float(np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])) if sorted_probs.shape[1] >= 2 else 0.0
            else:
                module_effect = 0.0
                exposed_top1 = 0.0
                exposed_margin = 0.0

        evidence_mean = evidence_mean_for_program(cid)
        evidence_gate = float(np.clip(math.log1p(max(0.0, evidence_mean)) / math.log1p(float(args.prior_evidence_norm)), 0.0, 1.0))
        effect_gate = _sigmoid_scalar(module_effect - float(args.module_min_effect_z), float(args.module_reliability_temperature))
        top1_gate = float(np.clip(exposed_top1, 0.0, 1.0))
        margin_gate = float(np.clip(exposed_margin / max(1e-6, float(args.prior_margin_norm)), 0.0, 1.0))

        # The prior is accepted only if both data effect and exposed-label agreement are present.
        rel = effect_gate * (0.45 + 0.35 * top1_gate + 0.20 * margin_gate) * (0.30 + 0.70 * evidence_gate)
        if module_effect < float(args.module_min_effect_z):
            rel *= float(args.low_effect_prior_shrink)
        if rel < float(args.module_min_reliability):
            rel = 0.0
        reliability[kk] = np.float32(np.clip(rel, 0.0, 1.0))

        module_rows.append({
            "class_id": str(cid),
            "class_name": class_name,
            "signed_gene_count": int(len(marker_arrays[cid].get("signed_genes", []))),
            "up_gene_count": int(len(marker_arrays[cid].get("up_genes", []))),
            "down_gene_count": int(len(marker_arrays[cid].get("down_genes", []))),
            "exposed_class_n": int(cls_n),
            "exposed_rest_n": int(rest_n),
            "module_effect_z": float(module_effect),
            "exposed_top1_fraction": float(exposed_top1),
            "exposed_probability_margin": float(exposed_margin),
            "evidence_mean": float(evidence_mean),
            "evidence_gate": float(evidence_gate),
            "module_reliability": float(reliability[kk]),
            "reason": str(marker_arrays[cid].get("reason", "")),
        })

    weighted = raw_probs * reliability.reshape(1, -1)
    prior_probs = _row_normalize_nonnegative(weighted)
    prior_conf = weighted.max(axis=1).astype(np.float32) if weighted.size else np.zeros(n, dtype=np.float32)
    chosen_idx = np.argmax(weighted, axis=1) if weighted.size else np.zeros(n, dtype=int)
    chosen_cid = [str(cids[int(i)]) for i in chosen_idx]

    active_modules = int((reliability > 0).sum())
    if active_modules == 0:
        prior_probs = np.zeros_like(raw_probs, dtype=np.float32)
        prior_conf = np.zeros(n, dtype=np.float32)

    gene_count = len(candidate_bank.get("gene_list", []))
    module_gene_weight = np.zeros((len(cids), gene_count), dtype=np.float32)
    module_gene_mask = np.zeros((len(cids), gene_count), dtype=np.float32)
    module_signed_gene_names = []
    for kk, cid in enumerate(cids):
        signed_names = []
        denom = 0.0
        for j, g, sign, eff in marker_arrays[cid].get("signed_genes", []):
            if 0 <= int(j) < gene_count:
                w = min(5.0, max(0.25, abs(float(eff))))
                module_gene_weight[kk, int(j)] = float(sign) * float(w)
                module_gene_mask[kk, int(j)] = 1.0
                signed_names.append({
                    "gene": str(g),
                    "sign": float(sign),
                    "effect_abs": float(abs(float(eff))),
                    "weight_raw": float(w),
                })
                denom += abs(float(w))
        if denom > 1e-6:
            module_gene_weight[kk, :] = module_gene_weight[kk, :] / float(denom)
        module_signed_gene_names.append(signed_names)

    return {
        "module_ids": [str(c) for c in cids],
        "module_names": [str(marker_arrays[c].get("class_name", f"class_{c}")) for c in cids],
        "gene_names": [str(g) for g in candidate_bank.get("gene_list", [])],
        "module_activity": scores.astype(np.float32),
        "module_raw_probability": raw_probs.astype(np.float32),
        "module_prior_probability": prior_probs.astype(np.float32),
        "module_reliability": reliability.astype(np.float32),
        "module_gene_weight": module_gene_weight.astype(np.float32),
        "module_gene_mask": module_gene_mask.astype(np.float32),
        "module_signed_gene_names": module_signed_gene_names,
        "sample_prior_confidence": prior_conf.astype(np.float32),
        "sample_prior_class_id": chosen_cid,
        "module_reliability_table": module_rows,
        "active_module_count": active_modules,
        "fairness": "Module reliability uses exposed labels only; no ALL/UNEXPOSED labels are used.",
    }


def save_context_prior_artifacts(
    out_dir: Path,
    sample_index: pd.Index,
    prior_pack: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Export artifacts that can be consumed by the downstream scFoundation script."""
    art = out_dir / "qwen_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    module_ids = [str(x) for x in prior_pack.get("module_ids", [])]
    module_names = [str(x) for x in prior_pack.get("module_names", [])]
    cols = [f"{cid}:{name}" for cid, name in zip(module_ids, module_names)]

    activity = np.asarray(prior_pack.get("module_activity", np.zeros((len(sample_index), 0))), dtype=np.float32)
    raw_prob = np.asarray(prior_pack.get("module_raw_probability", np.zeros_like(activity)), dtype=np.float32)
    prior_prob = np.asarray(prior_pack.get("module_prior_probability", np.zeros_like(activity)), dtype=np.float32)
    reliability = np.asarray(prior_pack.get("module_reliability", np.zeros((len(cols),))), dtype=np.float32)
    conf = np.asarray(prior_pack.get("sample_prior_confidence", np.zeros((len(sample_index),))), dtype=np.float32)
    module_gene_weight = np.asarray(prior_pack.get("module_gene_weight", np.zeros((len(cols), 0))), dtype=np.float32)
    module_gene_mask = np.asarray(prior_pack.get("module_gene_mask", np.zeros_like(module_gene_weight)), dtype=np.float32)

    pd.DataFrame(activity, index=sample_index, columns=cols).to_csv(art / "qwen_module_activity.csv")
    pd.DataFrame(raw_prob, index=sample_index, columns=cols).to_csv(art / "qwen_module_raw_probability.csv")
    pd.DataFrame(prior_prob, index=sample_index, columns=cols).to_csv(art / "qwen_module_prior_probability.csv")
    if module_gene_weight.size:
        gene_cols = [str(g) for g in prior_pack.get("gene_names", [])]
        if len(gene_cols) != module_gene_weight.shape[1]:
            gene_cols = [f"gene_{i}" for i in range(module_gene_weight.shape[1])]
        pd.DataFrame(module_gene_weight, index=cols, columns=gene_cols).to_csv(art / "qwen_module_gene_weight.csv")
        pd.DataFrame(module_gene_mask, index=cols, columns=gene_cols).to_csv(art / "qwen_module_gene_mask.csv")
    pd.DataFrame(prior_pack.get("module_reliability_table", [])).to_csv(art / "qwen_module_reliability.tsv", sep="\t", index=False)
    pd.DataFrame({
        "sample_id": [str(x) for x in sample_index],
        "prior_class_id": prior_pack.get("sample_prior_class_id", [""] * len(sample_index)),
        "prior_confidence": conf,
    }).to_csv(art / "qwen_sample_prior_assignment.tsv", sep="\t", index=False)

    summary = {
        "module_ids": module_ids,
        "module_names": module_names,
        "active_module_count": int(prior_pack.get("active_module_count", 0)),
        "module_reliability": {cid: float(r) for cid, r in zip(module_ids, reliability.tolist())},
        "mean_sample_prior_confidence": float(np.mean(conf)) if conf.size else 0.0,
        "max_sample_prior_confidence": float(np.max(conf)) if conf.size else 0.0,
        "recommended_downstream_usage": {
            "use_in_scfoundation": True,
            "preferred_target": "module_activity_auxiliary_loss_or_gene_module_regularization_not_sample_level_KL",
            "files": {
                "module_activity": "qwen_artifacts/qwen_module_activity.csv",
                "module_prior_probability": "qwen_artifacts/qwen_module_prior_probability.csv",
                "module_reliability": "qwen_artifacts/qwen_module_reliability.tsv",
                "npz": "qwen_artifacts/qwen_prior_for_scfoundation.npz",
            },
        },
        "fairness": prior_pack.get("fairness", "exposed-only prior validation"),
    }
    save_json(summary, art / "qwen_context_adaptive_prior_summary.json")

    np.savez_compressed(
        art / "qwen_prior_for_scfoundation.npz",
        sample_ids=np.asarray([str(x) for x in sample_index], dtype=object),
        module_ids=np.asarray(module_ids, dtype=object),
        module_names=np.asarray(module_names, dtype=object),
        module_activity=activity,
        module_raw_probability=raw_prob,
        module_prior_probability=prior_prob,
        module_reliability=reliability,
        module_gene_weight=module_gene_weight.astype(np.float32),
        module_gene_mask=module_gene_mask.astype(np.float32),
        gene_names=np.asarray([str(g) for g in prior_pack.get("gene_names", [])], dtype=object),
        module_signed_gene_names=np.asarray(prior_pack.get("module_signed_gene_names", []), dtype=object),
        sample_prior_confidence=conf,
        sample_prior_class_id=np.asarray(prior_pack.get("sample_prior_class_id", [""] * len(sample_index)), dtype=object),
    )

    if bool(args.export_sample_prior_graph) and prior_prob.shape[0] <= int(args.sample_prior_graph_max_n) and prior_prob.shape[1] > 0:
        # Build a cosine kNN graph from prior probabilities.
        x = prior_prob.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(1e-12, norms)
        sim = x @ x.T
        np.fill_diagonal(sim, -np.inf)
        k = max(1, int(args.sample_prior_graph_k))
        rows = []
        for i in range(sim.shape[0]):
            idx = np.argsort(-sim[i])[:k]
            for j in idx:
                if not np.isfinite(sim[i, j]):
                    continue
                rows.append({
                    "src": str(sample_index[i]),
                    "dst": str(sample_index[j]),
                    "prior_similarity": float(sim[i, j]),
                })
        pd.DataFrame(rows).to_csv(art / "qwen_sample_prior_knn_edges.tsv", sep="\t", index=False)


def apply_qcmce(
    expr: pd.DataFrame,
    candidate_bank: Dict[str, Any],
    marker_programs: Dict[str, Any],
    args: argparse.Namespace,
    strength: float,
    temperature: float,
    gate_quantile: float,
    variant_name: str,
    labels: Optional[pd.Series] = None,
    exposed_mask: Optional[np.ndarray] = None,
    prior_pack: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    z = candidate_bank["z"].astype(np.float32)
    ybase = candidate_bank["ybase"]
    mu = candidate_bank["mu"]
    sd = candidate_bank["sd"]
    transform_used = candidate_bank["expr_transform"]

    marker_arrays, gene_to_idx = build_marker_arrays(marker_programs, candidate_bank)
    if not marker_arrays:
        raise RuntimeError("no usable marker arrays for contrast enhancement")
    scores, cids = compute_marker_scores(z, marker_arrays)
    probs = softmax(scores, temp=float(temperature))

    # Compute expression adjustments from the context-adaptive prior.
    if prior_pack is not None and bool(args.use_context_prior_for_expression):
        prior_probs = np.asarray(prior_pack.get("module_prior_probability", probs), dtype=np.float32)
        prior_conf = np.asarray(prior_pack.get("sample_prior_confidence", np.zeros(probs.shape[0])), dtype=np.float32)
        if prior_probs.shape == probs.shape:
            order = np.argsort(-prior_probs, axis=1)
            top = order[:, 0]
            confidence = np.clip(prior_conf, 0.0, 1.0).astype(np.float32)
            chosen_cid = [cids[int(t)] for t in top]
        else:
            order = np.argsort(-probs, axis=1)
            top = order[:, 0]
            second = order[:, 1] if probs.shape[1] > 1 else order[:, 0]
            margin = probs[np.arange(probs.shape[0]), top] - probs[np.arange(probs.shape[0]), second]
            confidence = np.clip(margin, 0.0, 1.0).astype(np.float32)
            chosen_cid = [cids[int(t)] for t in top]
    else:
        order = np.argsort(-probs, axis=1)
        top = order[:, 0]
        second = order[:, 1] if probs.shape[1] > 1 else order[:, 0]
        margin = probs[np.arange(probs.shape[0]), top] - probs[np.arange(probs.shape[0]), second]
        confidence = np.clip(margin, 0.0, 1.0).astype(np.float32)
        chosen_cid = [cids[int(t)] for t in top]

    module_rel = {}
    if prior_pack is not None:
        rel = np.asarray(prior_pack.get("module_reliability", np.ones(len(cids))), dtype=np.float32)
        module_rel = {cid: float(rel[k]) for k, cid in enumerate(cids) if k < len(rel)}
        confidence = np.asarray([float(confidence[i]) * max(0.0, module_rel.get(chosen_cid[i], 1.0)) for i in range(len(chosen_cid))], dtype=np.float32)

    # Use exposed labels to select the enhancement program for exposed samples.
    chosen_cid = [str(x) for x in chosen_cid]
    if bool(args.anchor_exposed_labels) and labels is not None and exposed_mask is not None:
        yy = labels.to_numpy(dtype=int)
        ex = np.asarray(exposed_mask, dtype=bool)
        for i in np.where(ex)[0]:
            cid_true = str(int(yy[i]))
            if cid_true in marker_arrays:
                chosen_cid[i] = cid_true
                anchor = float(args.exposed_anchor_confidence) * max(0.0, module_rel.get(cid_true, 1.0))
                confidence[i] = max(float(confidence[i]), anchor)

    z_new = z.copy()
    delta = np.zeros_like(z, dtype=np.float32)

    # Gate expression updates by gene quantiles and sample-level marker affinity.
    gene_q_hi = np.quantile(z, float(gate_quantile), axis=0)
    gene_q_lo = np.quantile(z, 1.0 - float(gate_quantile), axis=0)

    for i in range(z.shape[0]):
        cid = chosen_cid[i]
        prog = marker_arrays.get(cid)
        if prog is None:
            continue
        conf = float(confidence[i])
        if conf <= 1e-6:
            continue
        for j, g, sign, eff in prog["signed_genes"]:
            eff_scale = min(2.0, max(0.25, abs(float(eff)) / 2.0))
            if sign > 0:
                # The high-expression gate approaches 1 as expression increases.
                gate = 1.0 / (1.0 + math.exp(-float(z[i, j] - gene_q_hi[j]) / max(1e-6, float(args.gate_temperature))))
                d = float(strength) * conf * eff_scale * gate
            else:
                gate = 1.0 / (1.0 + math.exp(float(z[i, j] - gene_q_lo[j]) / max(1e-6, float(args.gate_temperature))))
                d = -float(strength) * conf * eff_scale * gate
            delta[i, j] += d

    delta = np.clip(delta, -float(args.marker_delta_clip), float(args.marker_delta_clip)).astype(np.float32)
    z_new = np.clip(z + delta, -float(args.factor_z_clip), float(args.factor_z_clip)).astype(np.float32)
    x_new = inverse_z_to_expr(z_new, ybase, mu, sd, transform_used, args)

    x_orig = expr.to_numpy(dtype=np.float32, copy=False)
    if bool(args.keep_allzero_genes_zero):
        allzero = (np.abs(x_orig).sum(axis=0) <= 1e-12)
        x_new[:, allzero] = 0.0
    out = pd.DataFrame(x_new, index=expr.index, columns=expr.columns)

    touched = sorted({g for p in marker_arrays.values() for _, g, _, _ in p["signed_genes"]})
    meta = {
        "variant": variant_name,
        "strength": float(strength),
        "temperature": float(temperature),
        "gate_quantile": float(gate_quantile),
        "marker_program_count": int(len(marker_arrays)),
        "touched_gene_count": int(len(touched)),
        "touched_genes": touched,
        "confidence_summary": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "max": float(np.max(confidence)),
        },
        "chosen_class_counts": dict(Counter(chosen_cid)),
        "context_prior": {
            "used_for_expression": bool(prior_pack is not None and bool(args.use_context_prior_for_expression)),
            "active_module_count": int(prior_pack.get("active_module_count", 0)) if isinstance(prior_pack, dict) else None,
            "module_reliability": {str(k): float(v) for k, v in module_rel.items()} if module_rel else {},
        },
        "delta_stats_z": {
            "mean_abs_delta_z": float(np.abs(delta).mean()),
            "max_abs_delta_z": float(np.abs(delta).max()),
            "nonzero_delta_fraction": float((np.abs(delta) > 1e-12).mean()),
        },
        "expr_transform": transform_used,
    }
    return out, meta


# Quick check and selection

def _cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    cost = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cost[i, j] = np.sum((y_true == c) & (y_pred == k))
    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    return float(cost[row_ind, col_ind].sum() / max(1, len(y_true)))


def _cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for k in np.unique(y_pred):
        mask = y_pred == k
        if mask.sum() == 0:
            continue
        total += Counter(y_true[mask]).most_common(1)[0][1]
    return float(total / max(1, len(y_true)))


def quick_embedding(expr: pd.DataFrame, args: argparse.Namespace, seed: int) -> np.ndarray:
    x = expr.to_numpy(dtype=np.float32, copy=False)
    transform = str(getattr(args, "expr_transform", "auto"))
    finite = np.isfinite(x)
    valid = x[finite] if finite.any() else np.array([], dtype=np.float32)
    if transform == "auto":
        neg_frac = float((valid < 0).mean()) if valid.size else 0.0
        min_val = float(valid.min()) if valid.size else 0.0
        transform_used = "raw_z" if (min_val < 0.0 or neg_frac > 0.001) else "log1p_z"
    else:
        transform_used = transform
    if transform_used == "raw_z":
        x = x.astype(np.float32, copy=True)
    else:
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    pca_dim = int(getattr(args, "auto_quickcheck_pca_dim", 50))
    pca_dim = max(2, min(pca_dim, x.shape[0] - 1, x.shape[1] - 1))
    return PCA(n_components=pca_dim, random_state=seed).fit_transform(x)


def fast_cluster_metrics(expr: pd.DataFrame, labels: pd.Series, mask: np.ndarray, args: argparse.Namespace, seed: int) -> Dict[str, float]:
    emb = quick_embedding(expr, args, seed)
    y = labels.to_numpy(dtype=int)
    m = np.asarray(mask, dtype=bool)
    k = int(getattr(args, "auto_quickcheck_clusters", 0))
    if k <= 1:
        k = len(np.unique(y[m]))
    k = max(2, min(k, emb.shape[0] - 1))
    pred = KMeans(n_clusters=k, random_state=seed, n_init=30, max_iter=300).fit_predict(emb)
    return {
        "ACC": _cluster_acc(y[m], pred[m]),
        "NMI": float(normalized_mutual_info_score(y[m], pred[m])),
        "ARI": float(adjusted_rand_score(y[m], pred[m])),
        "PUR": _cluster_purity(y[m], pred[m]),
    }


def label_free_metrics(expr: pd.DataFrame, args: argparse.Namespace, seed: int) -> Dict[str, float]:
    emb = quick_embedding(expr, args, seed)
    k = int(getattr(args, "auto_quickcheck_clusters", 0))
    if k <= 1:
        k = 5
    k = max(2, min(k, emb.shape[0] - 1))
    pred = KMeans(n_clusters=k, random_state=seed, n_init=30, max_iter=300).fit_predict(emb)
    counts = np.bincount(pred, minlength=k)
    max_frac = float(counts.max() / max(1, counts.sum()))
    min_frac = float(counts.min() / max(1, counts.sum()))
    sil = 0.0
    try:
        if len(np.unique(pred)) > 1 and emb.shape[0] > len(np.unique(pred)):
            sil = float(silhouette_score(emb, pred, metric="euclidean"))
    except Exception:
        sil = 0.0
    return {"silhouette": sil, "max_cluster_frac": max_frac, "min_cluster_frac": min_frac}


def select_variant(
    before: pd.DataFrame,
    variants: Dict[str, pd.DataFrame],
    variant_effects: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    labels: Optional[pd.Series] = None,
    exposed_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    seeds = [int(x) for x in str(getattr(args, "auto_quickcheck_seeds", "0,1,2,3,4")).split(",") if str(x).strip()]
    metric = str(getattr(args, "auto_quickcheck_metric", "ARI")).upper()

    safe_names = []
    for name, eff in variant_effects.items():
        if name == "noop":
            safe_names.append(name)
            continue
        cf = float(eff.get("changed_fraction", 0.0))
        mad = float(eff.get("mean_abs_diff", 0.0))
        mx = float(eff.get("max_abs_diff", 0.0))
        if cf < float(args.min_changed_fraction):
            continue
        if cf > float(args.max_changed_fraction):
            continue
        if mad > float(args.max_mean_abs_diff):
            continue
        if mx > float(args.max_abs_diff):
            continue
        safe_names.append(name)

    if bool(args.auto_quickcheck) and labels is not None and exposed_mask is not None:
        raw_scores = [fast_cluster_metrics(before, labels, exposed_mask, args, s) for s in seeds]
        raw_mean = {k: float(np.mean([r[k] for r in raw_scores])) for k in raw_scores[0]}
        results = {}
        best_name = "noop"
        best_score = 0.0
        best_delta = 0.0
        min_nmi = float(args.selector_min_nmi_delta)
        min_acc = float(args.selector_min_acc_delta)
        min_pur = float(args.selector_min_pur_delta)
        min_pos = float(args.selector_min_positive_seed_fraction)

        for name in safe_names:
            mat = variants[name]
            scores = [fast_cluster_metrics(mat, labels, exposed_mask, args, s) for s in seeds]
            mean = {k: float(np.mean([r[k] for r in scores])) for k in scores[0]}
            delta = {k: float(mean[k] - raw_mean[k]) for k in mean}
            per_seed_delta = []
            for r, rr in zip(scores, raw_scores):
                per_seed_delta.append({k: float(r[k] - rr[k]) for k in r})
            positive_seed_fraction = float(np.mean([
                (d.get("ARI", 0.0) > 0.0) or (d.get("NMI", 0.0) > 0.0)
                for d in per_seed_delta
            ]))
            aux_ok = (
                delta.get("NMI", 0.0) >= min_nmi
                and delta.get("ACC", 0.0) >= min_acc
                and delta.get("PUR", 0.0) >= min_pur
                and positive_seed_fraction >= min_pos
            )
            score = (
                float(delta.get(metric, 0.0))
                + float(args.selector_nmi_weight) * float(delta.get("NMI", 0.0))
                + float(args.selector_acc_weight) * float(delta.get("ACC", 0.0))
                + float(args.selector_pur_weight) * float(delta.get("PUR", 0.0))
            )
            if not aux_ok and name != "noop":
                score -= 1.0
            results[name] = {
                "mean": mean,
                "delta_vs_raw": delta,
                "per_seed": scores,
                "per_seed_delta_vs_raw": per_seed_delta,
                "positive_seed_fraction": positive_seed_fraction,
                "selector_aux_ok": bool(aux_ok),
                "selector_score": float(score),
            }
            d = float(delta.get(metric, 0.0))
            if score > best_score and d >= float(args.auto_quickcheck_min_delta):
                best_score = float(score)
                best_delta = float(d)
                best_name = name

        return {
            "mode": "exposed_label_quickcheck",
            "selected_variant": best_name,
            "selected_metric": metric,
            "selected_delta": float(best_delta),
            "selected_score": float(best_score),
            "min_delta": float(args.auto_quickcheck_min_delta),
            "raw_mean": raw_mean,
            "results": results,
            "safe_names": safe_names,
            "seeds": seeds,
            "fairness": "selected_by_exposed_only",
        }

    # Label-free fallback.
    return {
        "mode": "no_quickcheck",
        "selected_variant": "noop",
        "selected_score": 0.0,
        "results": {},
        "safe_names": safe_names,
        "seeds": seeds,
        "fairness": "no_labels_used",
    }


# Reporting

def save_report(out_dir: Path, candidate_bank: Dict[str, Any], marker_programs: Dict[str, Any], prompt: str, raw_text: str, llm: Dict[str, Any], meta: Dict[str, Any]) -> None:
    art = out_dir / "qwen_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "qwen_marker_prompt.txt").write_text(prompt or "", encoding="utf-8")
    (art / "qwen_raw_response.txt").write_text(raw_text or "", encoding="utf-8")
    save_json(llm.get("api_response", {}) if isinstance(llm, dict) else {}, art / "qwen_api_response.json")
    save_json(marker_programs, art / "qwen_marker_programs.json")
    save_json(meta, art / "qwen_meta.json")

    compact_candidate = {k: v for k, v in candidate_bank.items() if k not in {"z", "ybase", "mu", "sd"}}
    save_json(compact_candidate, art / "candidate_bank_compact.json")
    write_jsonl(candidate_bank.get("selected_chunks", []), art / "selected_chunks.jsonl")

    rows = []
    for cid, p in candidate_bank.get("subtype_candidates", {}).items():
        for r in p.get("up_candidates", [])[:150]:
            row = {"class_id": cid, "class_name": p.get("class_name"), "candidate_type": "up"}
            row.update({k: v for k, v in r.items() if k not in {"evidence_titles"}})
            rows.append(row)
        for r in p.get("down_candidates", [])[:60]:
            row = {"class_id": cid, "class_name": p.get("class_name"), "candidate_type": "down"}
            row.update({k: v for k, v in r.items() if k not in {"evidence_titles"}})
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(art / "candidate_markers.tsv", sep="\t", index=False)

    md = []
    md.append("# v20-universal-adaptive/QCAMP Qwen Context-Adaptive Module Prior Report\n")
    md.append(f"- qwen_program_source: `{meta.get('qwen_program_source')}`")
    md.append(f"- selected_matrix_variant: `{meta.get('selected_matrix_variant')}`")
    md.append(f"- selected_delta: `{meta.get('auto_quickcheck', {}).get('selected_delta')}`")
    md.append(f"- final_changed_fraction: `{meta.get('matrix_effect', {}).get('final_effect', {}).get('changed_fraction')}`")
    md.append("\n## Marker programs\n")
    for p in marker_programs.get("subtype_marker_programs", []):
        md.append(f"### class {p.get('class_id')} {p.get('class_name')}")
        md.append(f"- up: {', '.join(p.get('selected_up_genes', [])[:50])}")
        if p.get("selected_down_genes"):
            md.append(f"- down: {', '.join(p.get('selected_down_genes', [])[:30])}")
        md.append(f"- reason: {p.get('reason','')}")
    (out_dir / "qwen_full_report.md").write_text("\n".join(md), encoding="utf-8")


# Args and main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v20-universal-adaptive/QCAMP Qwen-confirmed class marker/module prior before scFoundation")

    # Data
    p.add_argument("--data_path", type=str, default="/data/LJL/Main_Dataset/Main_Dataset/Classification_datasets/GS-BRCA/Top/BRCA_mRNA_top.csv")
    p.add_argument("--gene_index_path", type=str, default="/data/LJL/scFoundationmain/scFoundationmain/model/OS_scRNA_gene_index.19264.tsv")
    p.add_argument("--output_dir", type=str, default="/data1/LJL/ATTENTION_MAP/deepsearch_qwen/qwen_v20adaptive_all/brca_qwen_v20adaptive")
    p.add_argument("--dataset_name", type=str, default="GS-BRCA")
    p.add_argument("--dataset_key", type=str, default="BRCA")
    p.add_argument("--disease_context", type=str, default="breast cancer molecular subtype clustering")
    p.add_argument("--dataset_mode", type=str, default="bulk", choices=["auto", "bulk", "scrna", "spatial", "other"],
                   help="Dataset modality/context hint for generic Qwen prompting; does not use evaluation labels.")
    p.add_argument("--class_context", type=str, default="breast cancer molecular subtypes including LumA, HER2-enriched, LumB, Normal-like, Basal-like",
                   help="Optional human-readable class/cell-type/subtype context for the current dataset.")
    p.add_argument("--input_clip", type=float, default=50.0)
    p.add_argument("--dedup_policy", type=str, default="sum", choices=["first", "mean", "sum"])
    p.add_argument("--orientation_overlap_min", type=int, default=50)
    p.add_argument("--orientation_ratio", type=float, default=2.0)

    # Labels
    p.add_argument("--use_exposed_labels", type=str2bool, default=True)
    p.add_argument("--label_path", type=str, default="/data/LJL/Main_Dataset/Main_Dataset/Classification_datasets/GS-BRCA/Top/BRCA_label_num.csv")
    p.add_argument("--split_dir", type=str, default="/data1/LJL/ATTENTION_MAP/scF_new_model/new_best/split")
    p.add_argument("--exposed_index_path", type=str, default="")
    p.add_argument("--subtype_map_path", type=str, default="/data/LJL/scFoundationmain/scFoundationmain/BRCA_label_mapping.xlsx")
    
    p.add_argument("--labeled_ratio", type=float, default=0.20,
               help="Ratio of exposed labeled samples to create when split_dir has no exposed split.")
    p.add_argument("--stratified_sample", type=str2bool, default=True,
               help="Create exposed split by class-balanced sampling when split files are missing.")
    p.add_argument("--create_exposed_split_if_missing", type=str2bool, default=True,
               help="If true, create and save exposed split in split_dir when no exposed split exists.")

    p.add_argument("--min_exposed_per_class", type=int, default=3,
               help="Minimum exposed samples required for class-vs-rest marker candidate construction.")

    # Expression transform
    p.add_argument("--expr_transform", type=str, default="auto", choices=["auto", "raw_z", "log1p_z"])
    p.add_argument("--factor_z_clip", type=float, default=100.0)
    p.add_argument("--quantile_clip", type=str2bool, default=False)
    p.add_argument("--quantile_low", type=float, default=0.001)
    p.add_argument("--quantile_high", type=float, default=0.999)
    p.add_argument("--quantile_margin", type=float, default=0.10)
    p.add_argument("--keep_allzero_genes_zero", type=str2bool, default=True)

    # Knowledge
    p.add_argument("--prior_root", type=str, default="/data/LJL/scFoundationmain/data/bio_prior_processed")
    p.add_argument("--knowledge_path", type=str, default="")
    p.add_argument("--allowed_sources", type=str, default="Reactome,TRRUST,GTEx,PanglaoDB")
    p.add_argument("--max_knowledge_records", type=int, default=300000)
    p.add_argument("--max_chunks", type=int, default=32)
    p.add_argument("--max_chunks_per_source", type=int, default=10)
    p.add_argument("--max_context_chars", type=int, default=24000)
    p.add_argument("--max_chars_per_chunk", type=int, default=900)

    # Candidates and Qwen output
    p.add_argument("--candidate_genes_per_subtype", type=int, default=180)
    p.add_argument("--down_candidate_genes_per_subtype", type=int, default=50)
    p.add_argument("--prompt_candidates_per_subtype", type=int, default=80)
    p.add_argument("--prompt_down_candidates_per_subtype", type=int, default=25)
    p.add_argument("--qwen_selected_up_genes", type=int, default=40)
    p.add_argument("--qwen_selected_down_genes", type=int, default=10)
    p.add_argument("--qwen_min_selected_up_genes", type=int, default=10)
    p.add_argument("--max_genes_per_family", type=int, default=5)
    p.add_argument("--allow_rule_fallback", type=str2bool, default=False)
    p.add_argument("--qwen_allow_partial_fill", type=str2bool, default=True)

    # Configure Qwen calls using class count and prompt length.
    p.add_argument("--qwen_generation_mode", type=str, default="auto", choices=["auto", "single_call", "per_class"],
                   help="auto: single_call for small-K/short-prompt data, otherwise per_class; both keep the same Qwen marker-confirmation role.")
    p.add_argument("--qwen_single_call_max_k", type=int, default=6)
    p.add_argument("--qwen_prompt_budget_chars", type=int, default=60000)
    p.add_argument("--qwen_adaptive_prompt_profile", type=str2bool, default=True)
    p.add_argument("--qwen_prompt_profile", type=str, default="auto",
                   choices=["auto", "small_k_strong", "medium_k_compact", "large_k_ultracompact"])
    p.add_argument("--prompt_min_candidates_per_subtype", type=int, default=10)
    p.add_argument("--prompt_min_down_candidates_per_subtype", type=int, default=3)
    p.add_argument("--qwen_disable_thinking", type=str2bool, default=True)
    p.add_argument("--qwen_per_class_strict", type=str2bool, default=True)
    p.add_argument("--qwen_class_max_retries", type=int, default=2)
    p.add_argument("--qwen_class_retry_compact_factor", type=float, default=0.5)
    p.add_argument("--qwen_use_competing_context", type=str2bool, default=True)
    p.add_argument("--prompt_competing_top_genes", type=int, default=8)
    p.add_argument("--prompt_max_competing_classes", type=int, default=0,
                   help="0 means all competing classes up to K-1.")
    p.add_argument("--qwen_deconflict_markers", type=str2bool, default=True)
    p.add_argument("--qwen_deconflict_refill", type=str2bool, default=True)

    # Ollama/Qwen
    p.add_argument("--qwen_model_name", type=str, default="qwen3.5:9b")
    p.add_argument("--ollama_host", type=str, default="127.0.0.1:21434")
    p.add_argument("--qwen_timeout_sec", type=int, default=900)
    p.add_argument("--num_predict", type=int, default=5000)
    p.add_argument("--num_ctx", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repair_on_parse_fail", type=str2bool, default=True)
    p.add_argument("--force_json", type=str2bool, default=True)
    p.add_argument("--llm_context_chars", type=int, default=24000)

    # Contrast enhancement variants
    p.add_argument("--marker_strength_values", type=str, default="0.10,0.20,0.35,0.50")
    p.add_argument("--marker_temperature_values", type=str, default="1.0,2.0,4.0")
    p.add_argument("--marker_gate_quantile_values", type=str, default="0.60,0.70")
    p.add_argument("--marker_delta_clip", type=float, default=0.75)
    p.add_argument("--gate_temperature", type=float, default=0.50)
    p.add_argument("--anchor_exposed_labels", type=str2bool, default=True)
    p.add_argument("--exposed_anchor_confidence", type=float, default=0.75)

    # Context-adaptive prior
    p.add_argument("--qwen_prior_mode", type=str, default="prior_only", choices=["prior_only", "adaptive_expression", "both"],
                   help="prior_only exports context-adaptive prior artifacts and leaves qwen_guided_expression.csv unchanged; adaptive_expression also generates local expression variants; both is alias of adaptive_expression plus artifacts.")
    p.add_argument("--prior_assignment_temperature", type=float, default=1.50)
    p.add_argument("--module_min_effect_z", type=float, default=0.35,
                   help="Minimum exposed class-vs-rest module activity effect before a Qwen module is trusted.")
    p.add_argument("--module_reliability_temperature", type=float, default=0.35)
    p.add_argument("--module_min_reliability", type=float, default=0.05)
    p.add_argument("--low_effect_prior_shrink", type=float, default=0.15)
    p.add_argument("--prior_evidence_norm", type=float, default=8.0)
    p.add_argument("--prior_margin_norm", type=float, default=0.30)
    p.add_argument("--use_context_prior_for_expression", type=str2bool, default=True)
    p.add_argument("--export_sample_prior_graph", type=str2bool, default=True)
    p.add_argument("--sample_prior_graph_k", type=int, default=10)
    p.add_argument("--sample_prior_graph_max_n", type=int, default=5000)

    # Safety and selector
    p.add_argument("--min_changed_fraction", type=float, default=0.001)
    p.add_argument("--max_changed_fraction", type=float, default=0.08)
    p.add_argument("--max_mean_abs_diff", type=float, default=0.05)
    p.add_argument("--max_abs_diff", type=float, default=3.0)
    p.add_argument("--auto_quickcheck", type=str2bool, default=False)
    p.add_argument("--auto_quickcheck_metric", type=str, default="ARI")
    p.add_argument("--auto_quickcheck_min_delta", type=float, default=0.002)
    p.add_argument("--auto_quickcheck_seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--auto_quickcheck_pca_dim", type=int, default=50)
    p.add_argument("--auto_quickcheck_clusters", type=int, default=0)
    p.add_argument("--selector_min_nmi_delta", type=float, default=-0.0005)
    p.add_argument("--selector_min_acc_delta", type=float, default=-0.0010)
    p.add_argument("--selector_min_pur_delta", type=float, default=-0.0010)
    p.add_argument("--selector_min_positive_seed_fraction", type=float, default=0.60)
    p.add_argument("--selector_nmi_weight", type=float, default=0.25)
    p.add_argument("--selector_acc_weight", type=float, default=0.10)
    p.add_argument("--selector_pur_weight", type=float, default=0.10)

    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_dir = out_dir / "qwen_variants"
    var_dir.mkdir(parents=True, exist_ok=True)

    print("[START] Qwen context-adaptive marker/module prior v20-universal-adaptive/QCAMP", flush=True)
    print(f"[ARGS] data_path={args.data_path}", flush=True)
    print(f"[ARGS] gene_index_path={args.gene_index_path}", flush=True)
    print(f"[ARGS] output_dir={args.output_dir}", flush=True)

    gene_list = load_gene_list(args.gene_index_path)
    raw = read_expression(args.data_path)
    oriented = infer_orientation(raw, gene_list, int(args.orientation_overlap_min), float(args.orientation_ratio))
    cleaned = clean_expression(oriented, float(args.input_clip), str(args.dedup_policy))
    aligned, overlap, missing, extra = align_to_gene_list(cleaned, gene_list)
    print(f"[ALIGN] aligned_shape={aligned.shape} overlap={overlap} padded_missing={missing} extra={extra}", flush=True)

    labels = None
    exposed_mask = None
    subtype_map = load_subtype_map(args.subtype_map_path, args.dataset_key)

    if bool(args.use_exposed_labels):
        labels = load_label_series(args.label_path, aligned.index)

        try:
            exposed_mask, exposed_src = load_exposed_mask(
                args.split_dir,
                args.exposed_index_path,
                aligned.shape[0],
                aligned.index,
            )
        except FileNotFoundError as e:
            if not bool(getattr(args, "create_exposed_split_if_missing", True)):
                raise

            print(
                f"[LABEL][WARN] no existing exposed split found: {e}; "
                f"create exposed split in split_dir={args.split_dir}",
                flush=True,
            )

            exposed_mask, exposed_src = create_and_save_exposed_mask_qwen(
                labels=labels,
                sample_index=aligned.index,
                save_dir=args.split_dir,
                ratio=float(getattr(args, "labeled_ratio", 0.20)),
                seed=int(getattr(args, "seed", 0)),
                stratified=bool(getattr(args, "stratified_sample", True)),
                label_path=str(args.label_path),
            )

        if int(np.asarray(exposed_mask).sum()) <= 0:
            raise RuntimeError("[LABEL] exposed_mask has 0 samples after load/create")

        print(f"[LABEL] exposed_source={exposed_src}", flush=True)
        print(f"[LABEL] exposed_label_counts={dict(Counter(labels.to_numpy(dtype=int)[exposed_mask].tolist()))}", flush=True)

        exposed_k = infer_exposed_k(labels, exposed_mask)
        print(f"[LABEL] inferred_K_from_exposed={exposed_k}", flush=True)

        configure_qwen_universal_adaptive_profile(args, exposed_k)
    else:
        raise RuntimeError(
            "v20-universal-adaptive/QCAMP requires --use_exposed_labels true "
            "because data-adaptive prior validation uses exposed labels only."
        )

    # Build candidates and compact evidence.
    chunks = load_local_chunks(args)
    candidate_bank = build_marker_candidate_bank(aligned, labels, exposed_mask, subtype_map, chunks, args)

    # Confirm marker candidates with Qwen using the configured call protocol.
    prompt = ""
    raw_text = ""
    llm = {}
    qwen_source = "qwen_valid"
    try:
        marker_programs, llm, qwen_source, prompt, raw_text = generate_marker_programs_with_qwen(candidate_bank, args, out_dir)
    except Exception as e:
        if bool(args.allow_rule_fallback):
            print(f"[QWEN][PROGRAM_FAIL] {e}; using rule fallback marker programs", flush=True)
            marker_programs = rule_marker_programs(candidate_bank, args)
            qwen_source = "rule_fallback_after_invalid_qwen"
        else:
            # Save the Qwen response and prompt on failure.
            diag = {
                "method": "Qwen Context-Adaptive Module Prior (v20-universal-adaptive/QCAMP)",
                "failure": "invalid_qwen_marker_programs",
                "error": repr(e),
                "qwen_ok": llm.get("ok"),
                "used_field": llm.get("used_field"),
                "response_len": llm.get("response_len"),
                "thinking_len": llm.get("thinking_len"),
            }
            save_report(out_dir, candidate_bank, {"subtype_marker_programs": [], "notes": []}, prompt, raw_text, llm, diag)
            save_json(diag, out_dir / "run_meta.json")
            raise RuntimeError(
                f"Qwen marker program parsing failed and --allow_rule_fallback is false: {e}. "
                f"Diagnostics saved to {out_dir / 'qwen_artifacts'}"
            )

    # Generate context-adaptive prior artifacts and optional expression variants.
    before = aligned.copy()
    before.to_csv(out_dir / "aligned_expression_before_qwen.csv")

    context_prior = build_context_adaptive_module_prior(
        before, candidate_bank, marker_programs, args, labels=labels, exposed_mask=exposed_mask
    )
    save_context_prior_artifacts(out_dir, before.index, context_prior, args)
    print(
        f"[PRIOR] active_modules={context_prior.get('active_module_count')} "
        f"mean_sample_conf={float(np.mean(context_prior.get('sample_prior_confidence', np.zeros(before.shape[0])))):.6f}",
        flush=True,
    )

    variants: Dict[str, pd.DataFrame] = {"noop": before.copy()}
    variant_meta: Dict[str, Dict[str, Any]] = {"noop": {"variant": "noop", "qwen_prior_mode": str(args.qwen_prior_mode)}}

    strengths = parse_float_list(args.marker_strength_values, [0.10, 0.20, 0.35, 0.50])
    temps = parse_float_list(args.marker_temperature_values, [1.0, 2.0, 4.0])
    gates = parse_float_list(args.marker_gate_quantile_values, [0.60, 0.70])

    if str(args.qwen_prior_mode) in {"adaptive_expression", "both"}:
        for s in strengths:
            for t in temps:
                for gq in gates:
                    name = f"qcamp_s{str(s).replace('.', 'p')}_t{str(t).replace('.', 'p')}_q{str(gq).replace('.', 'p')}"
                    mat, vm = apply_qcmce(
                        before, candidate_bank, marker_programs, args, s, t, gq, name,
                        labels=labels, exposed_mask=exposed_mask, prior_pack=context_prior
                    )
                    variants[name] = mat
                    variant_meta[name] = vm
    else:
        print("[PRIOR_ONLY] qwen_guided_expression.csv will remain identical to aligned_expression_before_qwen.csv; use exported prior artifacts in scFoundation.", flush=True)

    variant_effects = {}
    for name, mat in variants.items():
        mat.to_csv(var_dir / f"{name}.csv")
        eff = compare_matrices(before, mat)
        variant_effects[name] = eff
        variant_meta[name]["matrix_effect"] = eff

    auto_qc = select_variant(before, variants, variant_effects, args, labels=labels, exposed_mask=exposed_mask)
    selected = str(auto_qc.get("selected_variant", "noop"))
    final = variants.get(selected, before.copy())
    final.to_csv(out_dir / "qwen_guided_expression.csv")

    final_eff = compare_matrices(before, final)
    meta = {
        "method": "Qwen Context-Adaptive Module Prior (v20-universal-adaptive/QCAMP)",
        "qwen_program_source": qwen_source,
        "selected_matrix_variant": selected,
        "marker_program_count": len(marker_programs.get("subtype_marker_programs", [])),
        "marker_gene_counts": {
            str(p.get("class_id")): {
                "up": len(p.get("selected_up_genes", [])),
                "down": len(p.get("selected_down_genes", [])),
            } for p in marker_programs.get("subtype_marker_programs", [])
        },
        "context_adaptive_prior": {
            "mode": str(args.qwen_prior_mode),
            "active_module_count": int(context_prior.get("active_module_count", 0)),
            "module_reliability_table": context_prior.get("module_reliability_table", []),
            "artifact_summary": "qwen_artifacts/qwen_context_adaptive_prior_summary.json",
            "npz_for_scfoundation": "qwen_artifacts/qwen_prior_for_scfoundation.npz",
            "module_prior_probability_csv": "qwen_artifacts/qwen_module_prior_probability.csv",
        },
        "data_stats": {"before": matrix_stats(before), "after": matrix_stats(final)},
        "alignment": {"overlap": int(overlap), "padded_missing": int(missing), "extra": int(extra), "shape": list(before.shape)},
        "label_protocol": {
            "use_exposed_labels": True,
            "exposed_n": int(np.asarray(exposed_mask).sum()),
            "fairness": "Qwen candidates and selector use exposed labels only; ALL/UNEXPOSED labels not used inside this script.",
        },
        "matrix_effect": {"final_effect": final_eff, "variant_effects": variant_effects},
        "variant_meta": variant_meta,
        "auto_quickcheck": auto_qc,
        "qwen_call": {
            "ok": llm.get("ok"),
            "used_field": llm.get("used_field"),
            "response_len": llm.get("response_len"),
            "thinking_len": llm.get("thinking_len"),
            "force_json": llm.get("force_json"),
            "error": llm.get("error"),
        },
        "args": vars(args),
    }

    save_json(meta, out_dir / "run_meta.json")
    save_report(out_dir, candidate_bank, marker_programs, prompt, raw_text, llm, meta)

    print(f"[DONE] aligned expression: {out_dir / 'aligned_expression_before_qwen.csv'}", flush=True)
    print(f"[DONE] qwen-guided expression: {out_dir / 'qwen_guided_expression.csv'}", flush=True)
    print(f"[DONE] context-adaptive prior: {out_dir / 'qwen_artifacts' / 'qwen_prior_for_scfoundation.npz'}", flush=True)
    print(f"[DONE] module prior probabilities: {out_dir / 'qwen_artifacts' / 'qwen_module_prior_probability.csv'}", flush=True)
    print(f"[DONE] variants dir: {var_dir}", flush=True)
    print(f"[DONE] run meta: {out_dir / 'run_meta.json'}", flush=True)
    print(f"[SELECT] selected_matrix_variant={selected}", flush=True)
    print(
        f"[EFFECT] changed_fraction={final_eff.get('changed_fraction', 0):.8f} "
        f"mean_abs_diff={final_eff.get('mean_abs_diff', 0):.8f} "
        f"max_abs_diff={final_eff.get('max_abs_diff', 0):.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
