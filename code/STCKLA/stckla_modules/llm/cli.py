"""llm.py definitions moved here without algorithm changes."""

import argparse
from .utils import (
    str2bool,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v20-universal-adaptive/QCAMP Qwen-confirmed class marker/module prior before scFoundation")

    # Data
    p.add_argument("--data_path", type=str, default="/data/LJL/Main_Dataset/Main_Dataset/Classification_datasets/GS-BRCA/Top/BRCA_mRNA_top.csv")
    p.add_argument("--gene_index_path", type=str, default="/data/LJL/scFoundationmain/scFoundationmain/model/OS_scRNA_gene_index.19264.tsv")
    p.add_argument("--output_dir", type=str, default="/data1/LJL/ATTENTION_MAP/deepsearch_qwen/qwen_v20adaptive_all/brca_qwen_v20adaptive")
    p.add_argument("--dataset_name", type=str, default="GS-BRCA")
    p.add_argument("--dataset_key", type=str, default="BRCA")
    p.add_argument("--disease_context", type=str, default="breast cancer molecular subtype clustering")
    p.add_argument("--dataset_mode", type=str, default="bulk", choices=["auto", "bulk", "scrna", "spatial", "other"],
                   help="Dataset modality/context hint for generic Qwen prompting; does not use evaluation labels.")
    p.add_argument("--class_context", type=str, default="breast cancer molecular subtypes including LumA, HER2-enriched, LumB, Normal-like, Basal-like",
                   help="Optional human-readable class/cell-type/subtype context for the current dataset.")
    p.add_argument("--input_clip", type=float, default=50.0)
    p.add_argument("--dedup_policy", type=str, default="sum", choices=["first", "mean", "sum"])
    p.add_argument("--orientation_overlap_min", type=int, default=50)
    p.add_argument("--orientation_ratio", type=float, default=2.0)

    # Labels
    p.add_argument("--use_exposed_labels", type=str2bool, default=True)
    p.add_argument("--label_path", type=str, default="/data/LJL/Main_Dataset/Main_Dataset/Classification_datasets/GS-BRCA/Top/BRCA_label_num.csv")
    p.add_argument("--split_dir", type=str, default="/data1/LJL/ATTENTION_MAP/scF_new_model/new_best/split")
    p.add_argument("--exposed_index_path", type=str, default="")
    p.add_argument("--subtype_map_path", type=str, default="/data/LJL/scFoundationmain/scFoundationmain/BRCA_label_mapping.xlsx")
    
    p.add_argument("--labeled_ratio", type=float, default=0.20,
               help="Ratio of exposed labeled samples to create when split_dir has no exposed split.")
    p.add_argument("--stratified_sample", type=str2bool, default=True,
               help="Create exposed split by class-balanced sampling when split files are missing.")
    p.add_argument("--create_exposed_split_if_missing", type=str2bool, default=True,
               help="If true, create and save exposed split in split_dir when no exposed split exists.")

    p.add_argument("--min_exposed_per_class", type=int, default=3,
               help="Minimum exposed samples required for class-vs-rest marker candidate construction.")

    # Expression transform
    p.add_argument("--expr_transform", type=str, default="auto", choices=["auto", "raw_z", "log1p_z"])
    p.add_argument("--factor_z_clip", type=float, default=100.0)
    p.add_argument("--quantile_clip", type=str2bool, default=False)
    p.add_argument("--quantile_low", type=float, default=0.001)
    p.add_argument("--quantile_high", type=float, default=0.999)
    p.add_argument("--quantile_margin", type=float, default=0.10)
    p.add_argument("--keep_allzero_genes_zero", type=str2bool, default=True)

    # Knowledge
    p.add_argument("--prior_root", type=str, default="/data/LJL/scFoundationmain/data/bio_prior_processed")
    p.add_argument("--knowledge_path", type=str, default="")
    p.add_argument("--allowed_sources", type=str, default="Reactome,TRRUST,GTEx,PanglaoDB")
    p.add_argument("--max_knowledge_records", type=int, default=300000)
    p.add_argument("--max_chunks", type=int, default=32)
    p.add_argument("--max_chunks_per_source", type=int, default=10)
    p.add_argument("--max_context_chars", type=int, default=24000)
    p.add_argument("--max_chars_per_chunk", type=int, default=900)

    # Candidates and Qwen output
    p.add_argument("--candidate_genes_per_subtype", type=int, default=180)
    p.add_argument("--down_candidate_genes_per_subtype", type=int, default=50)
    p.add_argument("--prompt_candidates_per_subtype", type=int, default=80)
    p.add_argument("--prompt_down_candidates_per_subtype", type=int, default=25)
    p.add_argument("--qwen_selected_up_genes", type=int, default=40)
    p.add_argument("--qwen_selected_down_genes", type=int, default=10)
    p.add_argument("--qwen_min_selected_up_genes", type=int, default=10)
    p.add_argument("--max_genes_per_family", type=int, default=5)
    p.add_argument("--allow_rule_fallback", type=str2bool, default=False)
    p.add_argument("--qwen_allow_partial_fill", type=str2bool, default=True)

    # Configure Qwen calls using class count and prompt length.
    p.add_argument("--qwen_generation_mode", type=str, default="auto", choices=["auto", "single_call", "per_class"],
                   help="auto: single_call for small-K/short-prompt data, otherwise per_class; both keep the same Qwen marker-confirmation role.")
    p.add_argument("--qwen_single_call_max_k", type=int, default=6)
    p.add_argument("--qwen_prompt_budget_chars", type=int, default=60000)
    p.add_argument("--qwen_adaptive_prompt_profile", type=str2bool, default=True)
    p.add_argument("--qwen_prompt_profile", type=str, default="auto",
                   choices=["auto", "small_k_strong", "medium_k_compact", "large_k_ultracompact"])
    p.add_argument("--prompt_min_candidates_per_subtype", type=int, default=10)
    p.add_argument("--prompt_min_down_candidates_per_subtype", type=int, default=3)
    p.add_argument("--qwen_disable_thinking", type=str2bool, default=True)
    p.add_argument("--qwen_per_class_strict", type=str2bool, default=True)
    p.add_argument("--qwen_class_max_retries", type=int, default=2)
    p.add_argument("--qwen_class_retry_compact_factor", type=float, default=0.5)
    p.add_argument("--qwen_use_competing_context", type=str2bool, default=True)
    p.add_argument("--prompt_competing_top_genes", type=int, default=8)
    p.add_argument("--prompt_max_competing_classes", type=int, default=0,
                   help="0 means all competing classes up to K-1.")
    p.add_argument("--qwen_deconflict_markers", type=str2bool, default=True)
    p.add_argument("--qwen_deconflict_refill", type=str2bool, default=True)

    # Ollama/Qwen
    p.add_argument("--qwen_model_name", type=str, default="qwen3.5:9b")
    p.add_argument("--ollama_host", type=str, default="127.0.0.1:21434")
    p.add_argument("--qwen_timeout_sec", type=int, default=900)
    p.add_argument("--num_predict", type=int, default=5000)
    p.add_argument("--num_ctx", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repair_on_parse_fail", type=str2bool, default=True)
    p.add_argument("--force_json", type=str2bool, default=True)
    p.add_argument("--llm_context_chars", type=int, default=24000)

    # Contrast enhancement variants
    p.add_argument("--marker_strength_values", type=str, default="0.10,0.20,0.35,0.50")
    p.add_argument("--marker_temperature_values", type=str, default="1.0,2.0,4.0")
    p.add_argument("--marker_gate_quantile_values", type=str, default="0.60,0.70")
    p.add_argument("--marker_delta_clip", type=float, default=0.75)
    p.add_argument("--gate_temperature", type=float, default=0.50)
    p.add_argument("--anchor_exposed_labels", type=str2bool, default=True)
    p.add_argument("--exposed_anchor_confidence", type=float, default=0.75)

    # Context-adaptive prior
    p.add_argument("--qwen_prior_mode", type=str, default="prior_only", choices=["prior_only", "adaptive_expression", "both"],
                   help="prior_only exports context-adaptive prior artifacts and leaves qwen_guided_expression.csv unchanged; adaptive_expression also generates local expression variants; both is alias of adaptive_expression plus artifacts.")
    p.add_argument("--prior_assignment_temperature", type=float, default=1.50)
    p.add_argument("--module_min_effect_z", type=float, default=0.35,
                   help="Minimum exposed class-vs-rest module activity effect before a Qwen module is trusted.")
    p.add_argument("--module_reliability_temperature", type=float, default=0.35)
    p.add_argument("--module_min_reliability", type=float, default=0.05)
    p.add_argument("--low_effect_prior_shrink", type=float, default=0.15)
    p.add_argument("--prior_evidence_norm", type=float, default=8.0)
    p.add_argument("--prior_margin_norm", type=float, default=0.30)
    p.add_argument("--use_context_prior_for_expression", type=str2bool, default=True)
    p.add_argument("--export_sample_prior_graph", type=str2bool, default=True)
    p.add_argument("--sample_prior_graph_k", type=int, default=10)
    p.add_argument("--sample_prior_graph_max_n", type=int, default=5000)

    # Safety and selector
    p.add_argument("--min_changed_fraction", type=float, default=0.001)
    p.add_argument("--max_changed_fraction", type=float, default=0.08)
    p.add_argument("--max_mean_abs_diff", type=float, default=0.05)
    p.add_argument("--max_abs_diff", type=float, default=3.0)
    p.add_argument("--auto_quickcheck", type=str2bool, default=False)
    p.add_argument("--auto_quickcheck_metric", type=str, default="ARI")
    p.add_argument("--auto_quickcheck_min_delta", type=float, default=0.002)
    p.add_argument("--auto_quickcheck_seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--auto_quickcheck_pca_dim", type=int, default=50)
    p.add_argument("--auto_quickcheck_clusters", type=int, default=0)
    p.add_argument("--selector_min_nmi_delta", type=float, default=-0.0005)
    p.add_argument("--selector_min_acc_delta", type=float, default=-0.0010)
    p.add_argument("--selector_min_pur_delta", type=float, default=-0.0010)
    p.add_argument("--selector_min_positive_seed_fraction", type=float, default=0.60)
    p.add_argument("--selector_nmi_weight", type=float, default=0.25)
    p.add_argument("--selector_acc_weight", type=float, default=0.10)
    p.add_argument("--selector_pur_weight", type=float, default=0.10)

    p.add_argument("--seed", type=int, default=0)
    return p
