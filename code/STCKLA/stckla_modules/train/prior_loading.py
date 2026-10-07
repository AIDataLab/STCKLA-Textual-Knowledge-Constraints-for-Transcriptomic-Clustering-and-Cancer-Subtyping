"""train.py definitions moved here without algorithm changes."""

from typing import Optional, Dict, Any
import numpy as np
import torch
from pathlib import Path


class ExprSemiDatasetWithIndex(torch.utils.data.Dataset):
    """Return (x, y, is_labeled, idx) for sample-aligned Qwen prior lookup."""
    def __init__(self, x_np, y_id=None, labeled_mask=None, unlabeled_label_value=-1):
        self.x = torch.as_tensor(np.asarray(x_np, dtype=np.float32), dtype=torch.float32)
        n = int(self.x.shape[0])
        if y_id is None:
            y = np.full(n, int(unlabeled_label_value), dtype=np.int64)
        else:
            y = np.asarray(y_id, dtype=np.int64).copy()
            if y.shape[0] != n:
                raise ValueError(f"[QWEN-DATASET] y_id length={y.shape[0]} != n={n}")
        if labeled_mask is None:
            is_lab = np.ones(n, dtype=bool) if y_id is not None else np.zeros(n, dtype=bool)
        else:
            is_lab = np.asarray(labeled_mask, dtype=bool).copy()
            if is_lab.shape[0] != n:
                raise ValueError(f"[QWEN-DATASET] labeled_mask length={is_lab.shape[0]} != n={n}")
        y[~is_lab] = int(unlabeled_label_value)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.is_lab = torch.as_tensor(is_lab, dtype=torch.bool)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.is_lab[idx], torch.tensor(idx, dtype=torch.long)


def _decode_npz_string_array(x):
    arr = np.asarray(x)
    out = []
    for v in arr.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def load_qwen_context_prior_npz(npz_path: str, sample_index, num_classes: int) -> Optional[Dict[str, Any]]:
    """Load module priors, confidence scores, and sample IDs from an NPZ file.

    Accepted keys:
      - module_prior_probability or prior_prob: [N, M]
      - sample_prior_confidence or prior_confidence: [N]
      - sample_ids or sample_index: [N]
      - module_ids: [M]

    Map numeric module class IDs to [0, num_classes); ignore unknown IDs.
    """
    if not npz_path:
        return None
    p = Path(str(npz_path))
    if not p.exists():
        raise FileNotFoundError(f"[QWEN-PRIOR] npz not found: {p}")

    z = np.load(str(p), allow_pickle=True)
    keys = set(z.files)

    # probability key compatibility
    if "prior_prob" in keys:
        prob_key = "prior_prob"
    elif "module_prior_probability" in keys:
        prob_key = "module_prior_probability"
    elif "module_raw_probability" in keys:
        prob_key = "module_raw_probability"
        print(
            "[QWEN-PRIOR][WARN] module_prior_probability not found; "
            "fallback to module_raw_probability.",
            flush=True,
        )
    else:
        raise KeyError(
            f"[QWEN-PRIOR] no probability matrix found in {p}; "
            f"expected one of prior_prob/module_prior_probability/module_raw_probability; keys={list(z.keys())}"
        )

    prior_prob = np.asarray(z[prob_key], dtype=np.float32)
    if prior_prob.ndim != 2:
        raise ValueError(f"[QWEN-PRIOR] {prob_key} must be 2D, got {prior_prob.shape}")

    module_activity_raw = None
    if "module_activity" in keys:
        module_activity_raw = np.asarray(z["module_activity"], dtype=np.float32)
        if module_activity_raw.ndim != 2:
            raise ValueError(f"[QWEN-PRIOR] module_activity must be 2D, got {module_activity_raw.shape}")

    n_current = len(sample_index)
    current_samples = [str(x) for x in list(sample_index)]

    # sample id key compatibility
    sample_key = None
    if "sample_index" in keys:
        sample_key = "sample_index"
    elif "sample_ids" in keys:
        sample_key = "sample_ids"

    if sample_key is not None:
        prior_samples = _decode_npz_string_array(z[sample_key])
        if prior_samples != current_samples:
            pos = {s: i for i, s in enumerate(prior_samples)}
            miss = [s for s in current_samples if s not in pos]
            if miss:
                # Align prior sample IDs with gexpr_aligned.index; reject mismatches.
                raise ValueError(
                    f"[QWEN-PRIOR] {sample_key} mismatch: missing {len(miss)} current samples in prior, "
                    f"first_missing={miss[:5]}, prior_first={prior_samples[:5]}, current_first={current_samples[:5]}"
                )
            order = np.asarray([pos[s] for s in current_samples], dtype=np.int64)
            prior_prob = prior_prob[order]
            if module_activity_raw is not None:
                module_activity_raw = module_activity_raw[order]
        else:
            order = None
    else:
        if prior_prob.shape[0] != n_current:
            raise ValueError(
                f"[QWEN-PRIOR] no sample_index/sample_ids and N mismatch: "
                f"prior={prior_prob.shape[0]} current={n_current}"
            )
        order = None

    # confidence key compatibility
    if "sample_prior_confidence" in keys:
        conf_raw = np.asarray(z["sample_prior_confidence"], dtype=np.float32)
    elif "prior_confidence" in keys:
        conf_raw = np.asarray(z["prior_confidence"], dtype=np.float32)
    else:
        conf_raw = prior_prob.max(axis=1).astype(np.float32)

    if sample_key is not None and order is not None:
        conf_raw = conf_raw[order]

    if prior_prob.shape[0] != n_current:
        raise ValueError(f"[QWEN-PRIOR] prior N={prior_prob.shape[0]} != current N={n_current}")

    # module/class id alignment
    if "module_ids" in keys:
        module_ids = _decode_npz_string_array(z["module_ids"])
    else:
        module_ids = [str(i) for i in range(prior_prob.shape[1])]

    aligned_prob = np.zeros((n_current, int(num_classes)), dtype=np.float32)
    aligned_activity = np.zeros((n_current, int(num_classes)), dtype=np.float32)
    module_reliability = np.zeros((int(num_classes),), dtype=np.float32)

    # Align module gene weights to class IDs for visible-gene selection: [C, G].
    module_gene_weight_raw = None
    module_gene_mask_raw = None
    aligned_module_gene_weight = None
    aligned_module_gene_mask = None
    if "module_gene_weight" in keys:
        module_gene_weight_raw = np.asarray(z["module_gene_weight"], dtype=np.float32)
        if module_gene_weight_raw.ndim == 2 and module_gene_weight_raw.shape[0] == prior_prob.shape[1]:
            gdim = int(module_gene_weight_raw.shape[1])
            aligned_module_gene_weight = np.zeros((int(num_classes), gdim), dtype=np.float32)
            if "module_gene_mask" in keys:
                module_gene_mask_raw = np.asarray(z["module_gene_mask"], dtype=np.float32)
                if module_gene_mask_raw.shape != module_gene_weight_raw.shape:
                    module_gene_mask_raw = None
            if module_gene_mask_raw is None:
                module_gene_mask_raw = (np.abs(module_gene_weight_raw) > 0).astype(np.float32)
            aligned_module_gene_mask = np.zeros((int(num_classes), gdim), dtype=np.float32)
        else:
            print(
                f"[QWEN-VISIBLE][WARN] ignore module_gene_weight with shape "
                f"{getattr(module_gene_weight_raw, 'shape', None)}; expected [M,G] with M={prior_prob.shape[1]}",
                flush=True,
            )
            module_gene_weight_raw = None
            module_gene_mask_raw = None
    if "module_reliability" in keys:
        rel_raw = np.asarray(z["module_reliability"], dtype=np.float32).reshape(-1)
    else:
        rel_raw = np.ones((prior_prob.shape[1],), dtype=np.float32)

    used_modules = []
    for j, mid in enumerate(module_ids):
        try:
            cid = int(float(str(mid)))
        except Exception:
            continue
        if 0 <= cid < int(num_classes):
            aligned_prob[:, cid] += prior_prob[:, j]
            if module_activity_raw is not None and j < module_activity_raw.shape[1]:
                aligned_activity[:, cid] += module_activity_raw[:, j]
            if j < rel_raw.shape[0]:
                module_reliability[cid] = max(float(module_reliability[cid]), float(rel_raw[j]))
            if aligned_module_gene_weight is not None and module_gene_weight_raw is not None and j < module_gene_weight_raw.shape[0]:
                aligned_module_gene_weight[cid] += module_gene_weight_raw[j].astype(np.float32)
                aligned_module_gene_mask[cid] = np.maximum(
                    aligned_module_gene_mask[cid],
                    module_gene_mask_raw[j].astype(np.float32),
                )
            used_modules.append(str(mid))

    row_sum = aligned_prob.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 1e-12
    if valid.any():
        aligned_prob[valid] = aligned_prob[valid] / row_sum[valid]
    if (~valid).any():
        aligned_prob[~valid] = 1.0 / float(num_classes)

    conf = np.asarray(conf_raw, dtype=np.float32).reshape(-1)
    if conf.shape[0] != n_current:
        raise ValueError(f"[QWEN-PRIOR] confidence length={conf.shape[0]} != N={n_current}")
    conf = np.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)

    print(
        f"[QWEN-PRIOR] loaded {p} | prob_key={prob_key} sample_key={sample_key} "
        f"prior_prob={aligned_prob.shape} used_modules={used_modules} "
        f"mean_conf={float(conf.mean()):.6f} max_conf={float(conf.max()):.6f} "
        f"visible_gene_weight={None if aligned_module_gene_weight is None else aligned_module_gene_weight.shape}",
        flush=True,
    )
    return {
        "active": True,
        "path": str(p),
        "prior_prob": aligned_prob.astype(np.float32),
        "module_activity": aligned_activity.astype(np.float32),
        "module_reliability": module_reliability.astype(np.float32),
        "aligned_module_gene_weight": None if aligned_module_gene_weight is None else aligned_module_gene_weight.astype(np.float32),
        "aligned_module_gene_mask": None if aligned_module_gene_mask is None else aligned_module_gene_mask.astype(np.float32),
        "prior_confidence": conf.astype(np.float32),
        "used_modules": used_modules,
        "raw_module_ids": module_ids,
        "prob_key": prob_key,
        "sample_key": sample_key,
    }
