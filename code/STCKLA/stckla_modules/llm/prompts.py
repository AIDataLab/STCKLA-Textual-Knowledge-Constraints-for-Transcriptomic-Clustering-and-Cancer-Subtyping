"""llm.py definitions moved here without algorithm changes."""

import argparse
import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from ..logging_utils import diagnostic_print
from .utils import (
    safe_float,
)


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
    diagnostic_print(f"[QWEN-PROTOCOL] adaptive_profile={profile} K={k_classes} resolved={args._qwen_adaptive_profile_meta['resolved']}", flush=True)


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
