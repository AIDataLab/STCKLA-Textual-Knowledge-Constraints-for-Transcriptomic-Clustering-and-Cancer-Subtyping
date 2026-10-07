"""llm.py definitions moved here without algorithm changes."""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


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
