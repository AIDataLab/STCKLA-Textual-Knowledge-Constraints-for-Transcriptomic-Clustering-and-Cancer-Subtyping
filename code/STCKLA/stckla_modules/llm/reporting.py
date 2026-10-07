"""llm.py definitions moved here without algorithm changes."""

from pathlib import Path
from typing import Any, Dict
import pandas as pd
from .utils import (
    save_json,
    write_jsonl,
)


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
