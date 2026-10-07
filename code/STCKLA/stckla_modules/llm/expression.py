"""llm.py definitions moved here without algorithm changes."""

import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse
try:
    import scanpy as sc
except Exception:
    sc = None
from ..logging_utils import diagnostic_print


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
    diagnostic_print(f"[ALIGN] overlap(columns,gene_list)={overlap_cols} | overlap(index,gene_list)={overlap_idx}", flush=True)
    if overlap_cols >= overlap_min or overlap_idx >= overlap_min:
        if overlap_cols > overlap_idx * ratio:
            diagnostic_print("[ALIGN] use input as samples x genes", flush=True)
            return df
        if overlap_idx > overlap_cols * ratio:
            diagnostic_print("[ALIGN] transpose genes x samples -> samples x genes", flush=True)
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
