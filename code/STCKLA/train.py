"""Compatibility entry point. Implementation: stckla_modules/train/."""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import argparse
import time
from typing import Optional, Dict, Any, Tuple, List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from load import load_model_frommmf  # noqa
import load as load_mod

import json
import pandas as pd

from base import (
    set_seed,
    sdp_kernel_ctx,
    amp_autocast_kwargs,
    install_scfoundation_patches,
    try_resume,
    save_local,
    save_exposed_ids,
    load_exposed_sample_ids,
    build_keep_mask_from_exposed,
    save_eval_split,
    gene_align_diagnostics,
    load_gene_list,
    ensure_samples_by_rows_no_label,
    read_any_to_df_raw,
    clean_numeric_df,
    align_to_gene_list,
    ResidualFeatureAdapter,
    ProjectionHead,
    ClassifierHead,
    PrototypeLayer,
    build_x_all,
    pool_tokens_like_train,
    load_labels,
    load_or_create_fixed_labeled_mask,
    load_labels_1col_numeric,
    remap_labels_to_0k,
    ExprSemiDataset,
    augment_expression,
    supervised_contrastive_loss,
    prototype_loss,
    consistency_loss,
    cluster_and_eval,
)

# Args
from stckla_modules.train.config import (
    Args,
    _str2bool,
    parse_args_from_class,
    args,
    normalize_infer_cluster_eval_mode,
    get_rng_state_all,
    set_rng_state_all,
    _module_state_cpu,
    _load_state_to_modules,
)

from stckla_modules.train.evaluation import (
    run_epoch_end_eval,
    infer_and_cluster_after_train,
)

from stckla_modules.train.encoder import (
    set_trainable,
    _prepare_encoder_inputs,
    encode_and_project_full,
    encode_and_project_light,
)

from stckla_modules.train.monitoring import (
    compute_embeddings_for_monitor,
    kmeans_monitor,
    _prepare_embedding_for_cluster_metrics,
    cluster_predict_for_checkpoint_stability,
    compute_label_free_cluster_quality_for_checkpoint,
)

from stckla_modules.train.prior_loading import (
    ExprSemiDatasetWithIndex,
    _decode_npz_string_array,
    load_qwen_context_prior_npz,
)

from stckla_modules.train.prior_guidance import (
    qwen_prior_weight_for_epoch,
    make_qwen_prior_batch_mask,
    qwen_prior_kl_loss_from_logits,
    qwen_prior_kl_loss_from_prototypes,
    QwenModuleActivityHead,
    qwen_module_activity_loss,
    _ensure_qwen_visible_tensors,
    _resolve_qwen_visible_quota_for_sample,
    _select_qwen_genes_for_one_sample,
    apply_qwen_visible_gene_injection,
    restore_encoder_values_after_qwen_visible_injection,
)

from stckla_modules.train.training_loop import (
    train_one,
)

from stckla_modules.train.pipeline import (
    main,
)

# Preserve historical class names for pickle and full-object checkpoints.
for _exported in (
    Args,
    ExprSemiDatasetWithIndex,
    QwenModuleActivityHead,
):
    _exported.__module__ = __name__
del _exported


if __name__ == "__main__":
    main()
