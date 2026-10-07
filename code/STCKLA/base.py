"""Compatibility entry point. Implementation: stckla_modules/base/."""


import os
import json
import time
import random
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment


# Utils
from stckla_modules.base.runtime import (
    set_seed,
    sdp_kernel_ctx,
    amp_autocast_kwargs,
)

from stckla_modules.base.scfoundation_patches import (
    gatherData_fixed_len_impl,
    _topk_keep_mask,
    build_get_encoder_decoder_patch_fn,
    install_scfoundation_patches,
)

from stckla_modules.base.checkpoints import (
    _local_paths,
    has_finetuned,
    try_resume,
    save_feature_list,
    save_label_meta,
    save_local,
)

from stckla_modules.base.labels_and_splits import (
    save_exposed_ids,
    load_fixed_exposed_mask_if_exists,
    load_exposed_sample_ids,
    build_keep_mask_from_exposed,
    save_eval_split,
    load_labels,
    pick_labeled_indices,
    load_or_create_fixed_labeled_mask,
    load_labels_1col_numeric,
    remap_labels_to_0k,
)

from stckla_modules.base.expression import (
    _clean_gene_names,
    gene_align_diagnostics,
    load_gene_list,
    ensure_samples_by_rows_no_label,
    dedup_columns,
    read_any_to_df_raw,
    clean_numeric_df,
    align_to_gene_list,
)

from stckla_modules.base.model_components import (
    ResidualFeatureAdapter,
    ProjectionHead,
    ClassifierHead,
    PrototypeLayer,
    build_x_all,
    pool_tokens_like_train,
    ExprSemiDataset,
    augment_expression,
)

from stckla_modules.base.optional_validation_split import (
    split_exposed_train_val_masks,
    save_labeled_train_val_split,
    load_fixed_labeled_train_val_split_if_exists,
)

from stckla_modules.base.losses import (
    supervised_contrastive_loss,
    prototype_loss,
    consistency_loss,
)

from stckla_modules.base.clustering import (
    cluster_acc,
    purity_score,
    preprocess_for_clustering,
    cluster_and_eval,
)

# Preserve historical class names for pickle and full-object checkpoints.
for _exported in (
    ResidualFeatureAdapter,
    ProjectionHead,
    ClassifierHead,
    PrototypeLayer,
    ExprSemiDataset,
):
    _exported.__module__ = __name__
del _exported
