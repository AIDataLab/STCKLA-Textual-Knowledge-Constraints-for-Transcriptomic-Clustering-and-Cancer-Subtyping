"""llm.py definitions moved here without algorithm changes."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..logging_utils import diagnostic_print
from .cli import (
    build_parser,
)
from .context_prior import (
    build_context_adaptive_module_prior,
    save_context_prior_artifacts,
)
from .expression import (
    align_to_gene_list,
    clean_expression,
    infer_orientation,
    load_gene_list,
    read_expression,
)
from .knowledge_candidates import (
    build_marker_candidate_bank,
    load_local_chunks,
)
from .labels_and_splits import (
    create_and_save_exposed_mask_qwen,
    load_exposed_mask,
    load_label_series,
    load_subtype_map,
)
from .marker_programs import (
    generate_marker_programs_with_qwen,
    rule_marker_programs,
)
from .optional_expression import (
    apply_qcmce,
    select_variant,
)
from .prompts import (
    configure_qwen_universal_adaptive_profile,
    infer_exposed_k,
)
from .reporting import (
    save_report,
)
from .utils import (
    compare_matrices,
    matrix_stats,
    parse_float_list,
    save_json,
    set_seed,
)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_dir = out_dir / "qwen_variants"
    var_dir.mkdir(parents=True, exist_ok=True)

    print("[START] Qwen context-adaptive marker/module prior v20-universal-adaptive/QCAMP", flush=True)
    diagnostic_print(f"[ARGS] data_path={args.data_path}", flush=True)
    diagnostic_print(f"[ARGS] gene_index_path={args.gene_index_path}", flush=True)
    diagnostic_print(f"[ARGS] output_dir={args.output_dir}", flush=True)

    gene_list = load_gene_list(args.gene_index_path)
    raw = read_expression(args.data_path)
    oriented = infer_orientation(raw, gene_list, int(args.orientation_overlap_min), float(args.orientation_ratio))
    cleaned = clean_expression(oriented, float(args.input_clip), str(args.dedup_policy))
    aligned, overlap, missing, extra = align_to_gene_list(cleaned, gene_list)
    print(f"[ALIGN] aligned_shape={aligned.shape} overlap={overlap} padded_missing={missing} extra={extra}", flush=True)

    labels = None
    exposed_mask = None
    subtype_map = load_subtype_map(args.subtype_map_path, args.dataset_key)

    if bool(args.use_exposed_labels):
        labels = load_label_series(args.label_path, aligned.index)

        try:
            exposed_mask, exposed_src = load_exposed_mask(
                args.split_dir,
                args.exposed_index_path,
                aligned.shape[0],
                aligned.index,
            )
        except FileNotFoundError as e:
            if not bool(getattr(args, "create_exposed_split_if_missing", True)):
                raise

            print(
                f"[LABEL][WARN] no existing exposed split found: {e}; "
                f"create exposed split in split_dir={args.split_dir}",
                flush=True,
            )

            exposed_mask, exposed_src = create_and_save_exposed_mask_qwen(
                labels=labels,
                sample_index=aligned.index,
                save_dir=args.split_dir,
                ratio=float(getattr(args, "labeled_ratio", 0.20)),
                seed=int(getattr(args, "seed", 0)),
                stratified=bool(getattr(args, "stratified_sample", True)),
                label_path=str(args.label_path),
            )

        if int(np.asarray(exposed_mask).sum()) <= 0:
            raise RuntimeError("[LABEL] exposed_mask has 0 samples after load/create")

        print(f"[LABEL] exposed_source={exposed_src}", flush=True)
        diagnostic_print(f"[LABEL] exposed_label_counts={dict(Counter(labels.to_numpy(dtype=int)[exposed_mask].tolist()))}", flush=True)

        exposed_k = infer_exposed_k(labels, exposed_mask)
        print(f"[LABEL] inferred_K_from_exposed={exposed_k}", flush=True)

        configure_qwen_universal_adaptive_profile(args, exposed_k)
    else:
        raise RuntimeError(
            "v20-universal-adaptive/QCAMP requires --use_exposed_labels true "
            "because data-adaptive prior validation uses exposed labels only."
        )

    # Build candidates and compact evidence.
    chunks = load_local_chunks(args)
    candidate_bank = build_marker_candidate_bank(aligned, labels, exposed_mask, subtype_map, chunks, args)

    # Confirm marker candidates with Qwen using the configured call protocol.
    prompt = ""
    raw_text = ""
    llm = {}
    qwen_source = "qwen_valid"
    try:
        marker_programs, llm, qwen_source, prompt, raw_text = generate_marker_programs_with_qwen(candidate_bank, args, out_dir)
    except Exception as e:
        if bool(args.allow_rule_fallback):
            print(f"[QWEN][PROGRAM_FAIL] {e}; using rule fallback marker programs", flush=True)
            marker_programs = rule_marker_programs(candidate_bank, args)
            qwen_source = "rule_fallback_after_invalid_qwen"
        else:
            # Save the Qwen response and prompt on failure.
            diag = {
                "method": "Qwen Context-Adaptive Module Prior (v20-universal-adaptive/QCAMP)",
                "failure": "invalid_qwen_marker_programs",
                "error": repr(e),
                "qwen_ok": llm.get("ok"),
                "used_field": llm.get("used_field"),
                "response_len": llm.get("response_len"),
                "thinking_len": llm.get("thinking_len"),
            }
            save_report(out_dir, candidate_bank, {"subtype_marker_programs": [], "notes": []}, prompt, raw_text, llm, diag)
            save_json(diag, out_dir / "run_meta.json")
            raise RuntimeError(
                f"Qwen marker program parsing failed and --allow_rule_fallback is false: {e}. "
                f"Diagnostics saved to {out_dir / 'qwen_artifacts'}"
            )

    # Generate context-adaptive prior artifacts and optional expression variants.
    before = aligned.copy()
    before.to_csv(out_dir / "aligned_expression_before_qwen.csv")

    context_prior = build_context_adaptive_module_prior(
        before, candidate_bank, marker_programs, args, labels=labels, exposed_mask=exposed_mask
    )
    save_context_prior_artifacts(out_dir, before.index, context_prior, args)
    print(
        f"[PRIOR] active_modules={context_prior.get('active_module_count')} "
        f"mean_sample_conf={float(np.mean(context_prior.get('sample_prior_confidence', np.zeros(before.shape[0])))):.6f}",
        flush=True,
    )

    variants: Dict[str, pd.DataFrame] = {"noop": before.copy()}
    variant_meta: Dict[str, Dict[str, Any]] = {"noop": {"variant": "noop", "qwen_prior_mode": str(args.qwen_prior_mode)}}

    strengths = parse_float_list(args.marker_strength_values, [0.10, 0.20, 0.35, 0.50])
    temps = parse_float_list(args.marker_temperature_values, [1.0, 2.0, 4.0])
    gates = parse_float_list(args.marker_gate_quantile_values, [0.60, 0.70])

    if str(args.qwen_prior_mode) in {"adaptive_expression", "both"}:
        for s in strengths:
            for t in temps:
                for gq in gates:
                    name = f"qcamp_s{str(s).replace('.', 'p')}_t{str(t).replace('.', 'p')}_q{str(gq).replace('.', 'p')}"
                    mat, vm = apply_qcmce(
                        before, candidate_bank, marker_programs, args, s, t, gq, name,
                        labels=labels, exposed_mask=exposed_mask, prior_pack=context_prior
                    )
                    variants[name] = mat
                    variant_meta[name] = vm
    else:
        print("[PRIOR_ONLY] qwen_guided_expression.csv will remain identical to aligned_expression_before_qwen.csv; use exported prior artifacts in scFoundation.", flush=True)

    variant_effects = {}
    for name, mat in variants.items():
        mat.to_csv(var_dir / f"{name}.csv")
        eff = compare_matrices(before, mat)
        variant_effects[name] = eff
        variant_meta[name]["matrix_effect"] = eff

    auto_qc = select_variant(before, variants, variant_effects, args, labels=labels, exposed_mask=exposed_mask)
    selected = str(auto_qc.get("selected_variant", "noop"))
    final = variants.get(selected, before.copy())
    final.to_csv(out_dir / "qwen_guided_expression.csv")

    final_eff = compare_matrices(before, final)
    meta = {
        "method": "Qwen Context-Adaptive Module Prior (v20-universal-adaptive/QCAMP)",
        "qwen_program_source": qwen_source,
        "selected_matrix_variant": selected,
        "marker_program_count": len(marker_programs.get("subtype_marker_programs", [])),
        "marker_gene_counts": {
            str(p.get("class_id")): {
                "up": len(p.get("selected_up_genes", [])),
                "down": len(p.get("selected_down_genes", [])),
            } for p in marker_programs.get("subtype_marker_programs", [])
        },
        "context_adaptive_prior": {
            "mode": str(args.qwen_prior_mode),
            "active_module_count": int(context_prior.get("active_module_count", 0)),
            "module_reliability_table": context_prior.get("module_reliability_table", []),
            "artifact_summary": "qwen_artifacts/qwen_context_adaptive_prior_summary.json",
            "npz_for_scfoundation": "qwen_artifacts/qwen_prior_for_scfoundation.npz",
            "module_prior_probability_csv": "qwen_artifacts/qwen_module_prior_probability.csv",
        },
        "data_stats": {"before": matrix_stats(before), "after": matrix_stats(final)},
        "alignment": {"overlap": int(overlap), "padded_missing": int(missing), "extra": int(extra), "shape": list(before.shape)},
        "label_protocol": {
            "use_exposed_labels": True,
            "exposed_n": int(np.asarray(exposed_mask).sum()),
            "fairness": "Qwen candidates and selector use exposed labels only; ALL/UNEXPOSED labels not used inside this script.",
        },
        "matrix_effect": {"final_effect": final_eff, "variant_effects": variant_effects},
        "variant_meta": variant_meta,
        "auto_quickcheck": auto_qc,
        "qwen_call": {
            "ok": llm.get("ok"),
            "used_field": llm.get("used_field"),
            "response_len": llm.get("response_len"),
            "thinking_len": llm.get("thinking_len"),
            "force_json": llm.get("force_json"),
            "error": llm.get("error"),
        },
        "args": vars(args),
    }

    save_json(meta, out_dir / "run_meta.json")
    save_report(out_dir, candidate_bank, marker_programs, prompt, raw_text, llm, meta)

    print(f"[DONE] aligned expression: {out_dir / 'aligned_expression_before_qwen.csv'}", flush=True)
    print(f"[DONE] qwen-guided expression: {out_dir / 'qwen_guided_expression.csv'}", flush=True)
    print(f"[DONE] context-adaptive prior: {out_dir / 'qwen_artifacts' / 'qwen_prior_for_scfoundation.npz'}", flush=True)
    print(f"[DONE] module prior probabilities: {out_dir / 'qwen_artifacts' / 'qwen_module_prior_probability.csv'}", flush=True)
    diagnostic_print(f"[DONE] variants dir: {var_dir}", flush=True)
    print(f"[DONE] run meta: {out_dir / 'run_meta.json'}", flush=True)
    diagnostic_print(f"[SELECT] selected_matrix_variant={selected}", flush=True)
    diagnostic_print(
        f"[EFFECT] changed_fraction={final_eff.get('changed_fraction', 0):.8f} "
        f"mean_abs_diff={final_eff.get('mean_abs_diff', 0):.8f} "
        f"max_abs_diff={final_eff.get('max_abs_diff', 0):.8f}",
        flush=True,
    )
