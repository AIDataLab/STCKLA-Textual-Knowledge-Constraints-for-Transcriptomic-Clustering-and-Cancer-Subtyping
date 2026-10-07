"""base.py definitions moved here without algorithm changes."""

from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from ..logging_utils import diagnostic_print


def _clean_gene_names(cols):
    return [str(c).strip() for c in cols]


def gene_align_diagnostics(df_cols, gene_list):
    cols = set(_clean_gene_names(df_cols))
    gl = set(_clean_gene_names(gene_list))
    overlap = len(cols & gl)
    missing = len(gl - cols)
    extra = len(cols - gl)
    diagnostic_print(f"[GENE][CHECK] raw_cols={len(cols)} | gene_list={len(gl)} | overlap={overlap} | missing_to_pad={missing} | extra_not_in_list={extra}")
    if overlap < 500:
        print("[GENE][WARN] overlap < 500: very likely NOT gene-symbol aligned -> huge zero padding -> embedding degrades.")
    elif overlap < 5000:
        print("[GENE][WARN] overlap is relatively low; alignment may be partial.")
    else:
        diagnostic_print("[GENE][OK] overlap looks reasonable.")
    return overlap, missing, extra


def load_gene_list(gene_index_path: str) -> List[str]:
    gene_list_df = pd.read_csv(gene_index_path, header=0, delimiter="\t")
    gl = [str(x) for x in list(gene_list_df["gene_name"])]
    seen = set()
    out = []
    for g in gl:
        g = str(g).strip()
        if g and (g not in seen):
            out.append(g)
            seen.add(g)
    return out


def ensure_samples_by_rows_no_label(
    df: pd.DataFrame,
    gene_list: List[str],
    *,
    orient_overlap_min: int,
    orient_ratio: float,
) -> pd.DataFrame:
    cols = set(_clean_gene_names(df.columns))
    idxs = set(_clean_gene_names(df.index))
    gl = set(_clean_gene_names(gene_list))

    overlap_cols = len(cols & gl)
    overlap_idx = len(idxs & gl)

    diagnostic_print(f"[ALIGN-NOLABEL] overlap(columns,gene_list)={overlap_cols} | overlap(index,gene_list)={overlap_idx}")

    if overlap_cols >= orient_overlap_min or overlap_idx >= orient_overlap_min:
        if overlap_cols > overlap_idx * orient_ratio:
            diagnostic_print("[ALIGN-NOLABEL] assume df is (samples, genes) based on columns overlap.")
            return df
        if overlap_idx > overlap_cols * orient_ratio:
            diagnostic_print("[ALIGN-NOLABEL] assume df is (genes, samples) -> transpose to (samples, genes).")
            return df.T

    print("[ALIGN-NOLABEL][WARN] cannot confidently infer orientation; keep as-is.")
    return df


def dedup_columns(df: pd.DataFrame, policy: str = "sum") -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(str)
    if not df.columns.duplicated().any():
        return df
    if policy == "first":
        return df.loc[:, ~df.columns.duplicated(keep="first")]
    if policy == "mean":
        return df.T.groupby(df.columns).mean().T
    return df.T.groupby(df.columns).sum().T


def read_any_to_df_raw(path: str) -> pd.DataFrame:
    if path.endswith("npz"):
        mat = scipy.sparse.load_npz(path)
        return pd.DataFrame(mat.toarray())
    if path.endswith("h5ad"):
        ad = sc.read_h5ad(path)
        idx = ad.obs_names.tolist()
        try:
            col = ad.var.gene_name.tolist()
        except Exception:
            col = ad.var_names.tolist()
        mat = ad.X.toarray() if issparse(ad.X) else ad.X
        return pd.DataFrame(mat, index=idx, columns=col)
    if path.endswith("npy"):
        return pd.DataFrame(np.load(path))
    return pd.read_csv(path, index_col=0)


def clean_numeric_df(df: pd.DataFrame, *, input_clip: float, dedup_policy: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index([str(x).strip() for x in df.columns.map(str)])
    bad = (df.columns == "") | (pd.Series(df.columns).str.lower().values == "nan")
    if bad.any():
        df = df.loc[:, ~bad]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.clip(lower=-float(input_clip), upper=float(input_clip))
    df = dedup_columns(df, policy=str(dedup_policy))
    return df


def align_to_gene_list(df: pd.DataFrame, gene_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = df.columns.map(str)
    gene_list = [str(x) for x in gene_list]

    missing = list(set(gene_list) - set(df.columns))
    if len(missing) > 0:
        pad = pd.DataFrame(
            np.zeros((df.shape[0], len(missing)), dtype=np.float32),
            index=df.index,
            columns=missing,
        )
        df = pd.concat([df, pad], axis=1)
    df = df[gene_list]
    return df, missing
