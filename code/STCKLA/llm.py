"""Compatibility entry point. Implementation: stckla_modules/llm/."""


import argparse
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import scanpy as sc
except Exception:
    sc = None



# Basic helpers


from stckla_modules.llm.utils import (
    str2bool,
    set_seed,
    norm_gene,
    gene_family_key,
    safe_float,
    parse_float_list,
    save_json,
    write_jsonl,
    matrix_stats,
    compare_matrices,
)

from stckla_modules.llm.expression import (
    load_gene_list,
    read_expression,
    infer_orientation,
    dedup_columns,
    clean_expression,
    align_to_gene_list,
    make_expr_z,
    inverse_z_to_expr,
)

from stckla_modules.llm.labels_and_splits import (
    _brca_default_subtype_map,
    load_subtype_map,
    load_label_series,
    _extract_indices_from_obj,
    load_exposed_mask,
    save_exposed_ids_qwen,
    pick_labeled_indices_qwen,
    create_and_save_exposed_mask_qwen,
)

from stckla_modules.llm.knowledge_candidates import (
    collect_genes_from_obj,
    iter_jsonl,
    source_bucket,
    load_local_chunks,
    compress_gene_evidence,
    BROAD_OR_TECHNICAL_GENES,
    is_broad_or_technical,
    select_top_genes_by_score,
    build_marker_candidate_bank,
)

from stckla_modules.llm.prompts import (
    prompt_bank,
    build_qwen_marker_prompt,
    infer_exposed_k,
    configure_qwen_universal_adaptive_profile,
    _compact_candidate_rows_for_prompt,
    prompt_bank_for_one_class,
    build_qwen_marker_prompt_for_class,
)

from stckla_modules.llm.ollama_client import (
    call_ollama,
    strip_thinking_and_fences,
    _try_json_loads_relaxed,
    extract_first_json_object,
    repair_marker_json,
    extract_marker_json_object,
)

from stckla_modules.llm.marker_programs import (
    _flatten_marker_rows,
    deconflict_marker_programs,
    _coerce_one_class_marker_program,
    generate_marker_programs_with_qwen,
    coerce_marker_programs,
    rule_marker_programs,
)

from stckla_modules.llm.context_prior import (
    build_marker_arrays,
    robust_z,
    compute_marker_scores,
    softmax,
    _sigmoid_scalar,
    _row_normalize_nonnegative,
    build_context_adaptive_module_prior,
    save_context_prior_artifacts,
)

from stckla_modules.llm.optional_expression import (
    apply_qcmce,
    _cluster_acc,
    _cluster_purity,
    quick_embedding,
    fast_cluster_metrics,
    label_free_metrics,
    select_variant,
)

from stckla_modules.llm.reporting import (
    save_report,
)

from stckla_modules.llm.cli import (
    build_parser,
)

from stckla_modules.llm.pipeline import (
    main,
)



if __name__ == "__main__":
    main()
