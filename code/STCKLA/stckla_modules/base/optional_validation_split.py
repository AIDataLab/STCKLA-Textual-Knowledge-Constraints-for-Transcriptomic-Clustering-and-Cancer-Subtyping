"""base.py definitions moved here without algorithm changes. Optional path; retained for compatibility."""

import os
import json
import time
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def split_exposed_train_val_masks(
    y_id: np.ndarray,
    exposed_mask: np.ndarray,
    *,
    val_ratio: float = 0.20,
    seed: int = 0,
    stratified: bool = True,
    min_val_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split exposed samples into disjoint training and validation masks.

    Both outputs are subsets of exposed_mask. Validation samples are
    reserved for checkpoint selection. Evaluation of unexposed samples
    excludes the entire exposed_mask.
    """
    y_id = np.asarray(y_id, dtype=np.int64).reshape(-1)
    exposed_mask = np.asarray(exposed_mask, dtype=bool).reshape(-1)

    if len(y_id) != len(exposed_mask):
        raise RuntimeError(
            f"[VALSPLIT] y_id len {len(y_id)} != exposed_mask len {len(exposed_mask)}"
        )

    val_ratio = float(val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"[VALSPLIT] val_ratio must be in (0,1), got {val_ratio}")

    rng = np.random.RandomState(int(seed))

    train_mask = np.zeros_like(exposed_mask, dtype=bool)
    val_mask = np.zeros_like(exposed_mask, dtype=bool)

    exposed_idx = np.where(exposed_mask)[0]
    if exposed_idx.size == 0:
        raise RuntimeError("[VALSPLIT] exposed_mask has 0 samples.")

    if not stratified:
        perm = rng.permutation(exposed_idx)
        n_val = int(round(len(perm) * val_ratio))
        n_val = max(1, n_val)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_mask[val_idx] = True
        train_mask[train_idx] = True
    else:
        classes = np.unique(y_id[exposed_idx])
        for c in classes:
            idx_c = exposed_idx[y_id[exposed_idx] == c]
            idx_c = rng.permutation(idx_c)

            if len(idx_c) <= 1:
                # Assign singleton classes to the training split.
                train_mask[idx_c] = True
                continue

            n_val_c = int(round(len(idx_c) * val_ratio))
            n_val_c = max(int(min_val_per_class), n_val_c)
            n_val_c = min(n_val_c, len(idx_c) - 1)

            val_idx_c = idx_c[:n_val_c]
            train_idx_c = idx_c[n_val_c:]

            val_mask[val_idx_c] = True
            train_mask[train_idx_c] = True

    if bool(np.any(train_mask & val_mask)):
        raise RuntimeError("[VALSPLIT] train_mask and val_mask overlap.")

    if not bool(np.all((train_mask | val_mask) <= exposed_mask)):
        raise RuntimeError("[VALSPLIT] train/val mask must be subset of exposed_mask.")

    print(
        f"[VALSPLIT] exposed={int(exposed_mask.sum())} "
        f"train_labeled={int(train_mask.sum())} "
        f"val_labeled={int(val_mask.sum())} "
        f"val_ratio={val_ratio} stratified={stratified}"
    )

    return train_mask, val_mask


def save_labeled_train_val_split(
    save_dir: str,
    sample_index: pd.Index,
    exposed_mask: np.ndarray,
    train_labeled_mask: np.ndarray,
    val_labeled_mask: np.ndarray,
    *,
    seed: int,
    val_ratio: float,
):
    """Save the training and validation masks within the exposed subset."""
    os.makedirs(save_dir, exist_ok=True)

    exposed_mask = np.asarray(exposed_mask, dtype=bool)
    train_labeled_mask = np.asarray(train_labeled_mask, dtype=bool)
    val_labeled_mask = np.asarray(val_labeled_mask, dtype=bool)

    n_total = len(sample_index)
    for name, m in [
        ("exposed_mask", exposed_mask),
        ("train_labeled_mask", train_labeled_mask),
        ("val_labeled_mask", val_labeled_mask),
    ]:
        if m.ndim != 1 or len(m) != n_total:
            raise RuntimeError(f"[VALSPLIT] {name} shape={m.shape}, expect ({n_total},)")

    if bool(np.any(train_labeled_mask & val_labeled_mask)):
        raise RuntimeError("[VALSPLIT] train_labeled_mask overlaps val_labeled_mask.")

    if not bool(np.all((train_labeled_mask | val_labeled_mask) <= exposed_mask)):
        raise RuntimeError("[VALSPLIT] train/val masks must be subsets of exposed_mask.")

    train_rows = np.where(train_labeled_mask)[0].astype(np.int64)
    val_rows = np.where(val_labeled_mask)[0].astype(np.int64)
    exposed_rows = np.where(exposed_mask)[0].astype(np.int64)

    np.save(os.path.join(save_dir, "train_labeled_mask.npy"), train_labeled_mask.astype(np.bool_))
    np.save(os.path.join(save_dir, "val_labeled_mask.npy"), val_labeled_mask.astype(np.bool_))
    np.save(os.path.join(save_dir, "train_labeled_rows.npy"), train_rows)
    np.save(os.path.join(save_dir, "val_labeled_rows.npy"), val_rows)

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id_type": "row_index",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "n_total": int(n_total),
        "n_exposed": int(exposed_mask.sum()),
        "n_train_labeled": int(train_labeled_mask.sum()),
        "n_val_labeled": int(val_labeled_mask.sum()),
        "exposed_rows": exposed_rows.tolist(),
        "train_labeled_rows": train_rows.tolist(),
        "val_labeled_rows": val_rows.tolist(),
    }

    with open(os.path.join(save_dir, "labeled_train_val_split.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[VALSPLIT] saved train/val labeled split to {save_dir} | "
        f"train={len(train_rows)} val={len(val_rows)} exposed={len(exposed_rows)}"
    )


def load_fixed_labeled_train_val_split_if_exists(
    save_dir: str,
    n_samples: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load saved training and validation masks when both files exist."""
    train_path = os.path.join(save_dir, "train_labeled_mask.npy")
    val_path = os.path.join(save_dir, "val_labeled_mask.npy")

    if not (os.path.isfile(train_path) and os.path.isfile(val_path)):
        return None, None

    try:
        train_mask = np.asarray(np.load(train_path), dtype=bool)
        val_mask = np.asarray(np.load(val_path), dtype=bool)

        if train_mask.ndim != 1 or len(train_mask) != int(n_samples):
            print(f"[VALSPLIT][WARN] train_labeled_mask shape mismatch: {train_mask.shape}")
            return None, None

        if val_mask.ndim != 1 or len(val_mask) != int(n_samples):
            print(f"[VALSPLIT][WARN] val_labeled_mask shape mismatch: {val_mask.shape}")
            return None, None

        if bool(np.any(train_mask & val_mask)):
            print("[VALSPLIT][WARN] train/val masks overlap; ignore existing split.")
            return None, None

        print(
            f"[VALSPLIT] loaded fixed train/val labeled split from {save_dir}: "
            f"train={int(train_mask.sum())} val={int(val_mask.sum())}"
        )
        return train_mask, val_mask

    except Exception as e:
        print("[VALSPLIT][WARN] failed to load train/val split:", repr(e))
        return None, None
