"""llm.py definitions moved here without algorithm changes."""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from .ollama_client import (
    call_ollama,
    extract_marker_json_object,
    repair_marker_json,
)
from .prompts import (
    build_qwen_marker_prompt,
    build_qwen_marker_prompt_for_class,
)
from .utils import (
    norm_gene,
    safe_float,
    save_json,
)


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
