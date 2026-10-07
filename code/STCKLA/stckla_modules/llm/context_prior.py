"""llm.py definitions moved here without algorithm changes."""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .utils import (
    norm_gene,
    safe_float,
    save_json,
)


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
