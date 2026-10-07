"""llm.py definitions moved here without algorithm changes. Optional path; retained for compatibility."""

import argparse
import math
from collections import Counter
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from .context_prior import (
    build_marker_arrays,
    compute_marker_scores,
    softmax,
)
from .expression import (
    inverse_z_to_expr,
)


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
