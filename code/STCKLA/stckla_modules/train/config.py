"""train.py definitions moved here without algorithm changes."""

import argparse
from typing import Optional, Dict, Any
import random
import numpy as np
import torch
import torch.nn as nn


class Args:
    data_path = "/data/LJL/Main_Dataset/Main_Dataset/Classification_datasets/GS-BRCA/Top/BRCA_mRNA_top.csv"
    gene_index_path = "/data/LJL/scFoundationmain/scFoundationmain/model/OS_scRNA_gene_index.19264.tsv"

    label_path = "/data/LJL/Main_Dataset/Main_Dataset/Classification_datasets/GS-BRCA/Top/BRCA_label_num.csv"
    semi_supervised = True
    labeled_ratio = 0.20
    stratified_sample = True
    unlabeled_label_value = -1

    use_exposed_cluster_checkpoint = True
    exposed_cluster_metric = "ARI"  # ACC | NMI | ARI | PUR

    # Exposed + label-free temporal stability + self-loss checkpoint selection
    exposed_cluster_candidate_eps = 0.005
    exposed_cluster_stability_weight = 0.35
    exposed_cluster_loss_weight = 0.02
    exposed_cluster_loss_eps = 1e-8

    # 0 means use all samples for temporal stability.
    # For large datasets, set 3000 or 5000 to reduce cost.
    exposed_cluster_stability_max_samples = 3000

    # Rerank checkpoints after training.
    # This uses exposed labels + label-free cluster trajectory only.
    exposed_cluster_posthoc_rerank = True
    exposed_cluster_stability_window = 2

    exposed_cluster_tie_eps = 0.003
    # Combine exposed-score filtering, structure quality, and temporal stability.
    exposed_cluster_tie_break = "quality"  # quality | plateau_center | earlier | lower_loss

    # Plateau-center selection:
    # candidate epochs must have exposed score close to the best exposed score;
    # stable plateau is decided by smooth temporal stability.
    exposed_cluster_plateau_stability_eps = 0.03
    exposed_cluster_plateau_min_len = 3
    exposed_cluster_plateau_center_radius = 1
    # Quantile used to select an epoch within the stable plateau.
    # A value of 0.50 selects the median position.
    exposed_cluster_plateau_center_quantile = 0.65

    # Label-free structure-quality checkpoint selection.
    # Compute quality metrics from embeddings and cluster assignments only.
    exposed_cluster_quality_weight = 0.35
    exposed_cluster_quality_silhouette_weight = 0.40
    exposed_cluster_quality_ch_weight = 0.20
    exposed_cluster_quality_db_weight = 0.20
    exposed_cluster_quality_balance_weight = 0.20
    exposed_cluster_collapse_weight = 0.20
    exposed_cluster_endpoint_weight = 0.03
    exposed_cluster_endpoint_start = 0.90
    exposed_cluster_min_cluster_frac = 0.005
    exposed_cluster_max_cluster_frac = 0.80
    exposed_cluster_quality_max_samples = 3000

    local_model_dir = "/data1/LJL/ATTENTION_MAP/scF_qualityckpt_ep40/model_brca_qwen"
    exposed_save_dir = "/data1/LJL/ATTENTION_MAP/scF_new_model/new_best/split"

    ckpt_path = "/data/LJL/scFoundationmain/scFoundationmain/model/models/models.ckpt"
    key = "cell"

    master_fixed_dim = 19264
    device = "cuda:0"

    epochs = 40
    batch_size = 1
    accum_steps = 8
    lr = 3e-4
    backbone_lr_scale = 0.05
    weight_decay = 0.02
    grad_clip = 0.5
    num_workers = 0

    input_clip = 50.0
    dedup_policy = "sum"

    valid_eps = 1e-12
    mask_prob = 0.10
    mae_encoder_max_seq_len = 15000

    mask_mode = "pos"
    mask_eps = 1e-6

    pre_normalized = "F"
    totalcount_mode = "sumabs"

    freeze_scfoundation = False
    unfreeze_mode = "second_last" # second_last | last_n
    unfreeze_last_n = 2

    amp = False
    amp_dtype = "fp16"
    force_sdp = True

    use_adapter = True
    adapter_hidden_dim = 512
    adapter_alpha = 0.20
    adapter_dropout = 0.10

    encoder_visible_max_len = 2500
    encoder_topk_by = "abs"


    qwen_visible_inject = True
    qwen_visible_quota = 300
    qwen_visible_quota_mode = "auto"      # auto | fixed | adaptive
    qwen_visible_quota_fraction = 0.12
    qwen_visible_quota_min = 80
    qwen_visible_quota_max = 300
    qwen_visible_mode = "sample_module"  # sample_module | global
    qwen_visible_module_topk = 1
    qwen_visible_gene_topk_per_module = 300
    qwen_visible_boost_value = 1000.0
    qwen_visible_require_nonzero = True
    qwen_visible_eps = 1e-8
    qwen_visible_debug = True

    seed = 0
    strict_reproducible = True
    restore_rng_after_eval = True
    print_debug_first_batch = True

    monitor_every_steps = 200
    monitor_max_samples = 512
    monitor_kmeans_k = 5
    monitor_pca_dim = 50
    monitor_cluster_prep = "raw"

    proj_dim = 256
    proj_hidden_dim = 1024
    proj_dropout = 0.10
    cls_hidden_dim = 512
    cls_dropout = 0.10
    proto_temperature = 0.10
    supcon_temperature = 0.10

    aug_noise_std = 0.01
    aug_drop_prob = 0.05

    w_mae = 0.50
    w_ce = 0.50
    w_supcon = 1.00
    w_proto = 0.50
    w_cons = 0.1
    pseudo_label_start_epoch = 100
    pseudo_label_thresh = 0.99
    w_pseudo = 0.00

    # Qwen module prior
    use_qwen_context_prior = True
    qwen_prior_npz = "/data1/LJL/ATTENTION_MAP/deepsearch_qwen/qwen_v20adaptive_all/brca_qwen_v20adaptive/qwen_artifacts/qwen_prior_for_scfoundation.npz"
    qwen_prior_loss_type = "activity_mse"      # activity_mse | kl | proto | both | none
    qwen_prior_apply_to = "all"                # all | unlabeled | labeled
    qwen_prior_loss_weight = 0.005
    qwen_prior_conf_threshold = 0.0
    qwen_prior_warmup_epochs = 8
    qwen_prior_temperature = 1.0
    qwen_prior_proto_temperature = 0.10
    qwen_prior_min_conf = 1e-6

    # Auxiliary head for module-activity prediction.
    qwen_activity_hidden_dim = 128
    qwen_activity_dropout = 0.10
    qwen_activity_target_clip = 5.0
    qwen_activity_use_conf_weight = False
    qwen_activity_min_sample_weight = 0.25

    resume_if_present = False

    orient_overlap_min = 500
    orient_ratio = 1.2

    run_infer_after_train = True
    infer_save_path = "/data1/LJL/ATTENTION_MAP/scF_qualityckpt_ep40/infer_brca_qwen"
    infer_task_name = "GS_BRCA"
    infer_ckpt_name = "scfoundation_qwen_qualityckpt"
    infer_batch_size = 2
    infer_cluster_eval_mode = "final"  # final | every_epoch | both
    infer_every_epoch_subdir = True

    use_exposed_filter = True

    k_fixed = 5
    cluster_prep = "raw"
    pca_dim = 50


def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args_from_class(cls):
    parser = argparse.ArgumentParser(
        description="scFoundation semi-supervised training script (CLI enabled)"
    )

    for name, default in cls.__dict__.items():
        if name.startswith("_") or callable(default):
            continue

        arg_name = f"--{name}"

        if isinstance(default, bool):
            parser.add_argument(arg_name, type=_str2bool, default=default)
        elif isinstance(default, int):
            parser.add_argument(arg_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg_name, type=float, default=default)
        else:
            parser.add_argument(arg_name, type=str, default=default)

    return parser.parse_args()


args = parse_args_from_class(Args)


def normalize_infer_cluster_eval_mode(v):
    mode = str(v).strip().lower()
    if mode not in {"final", "every_epoch", "both"}:
        raise ValueError(
            f"Unsupported infer_cluster_eval_mode={v}, expected 'final', 'every_epoch' or 'both'"
        )
    return mode


def get_rng_state_all():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state_all(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _module_state_cpu(module: Optional[nn.Module]):
    if module is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_state_to_modules(
    ckpt_state: Dict[str, Any],
    pretrainmodel: nn.Module,
    adapter: nn.Module,
    pred_head: nn.Module,
    cls_head: Optional[nn.Module],
    proj_head: nn.Module,
    proto_layer: Optional[nn.Module],
    device: torch.device,
):
    pretrainmodel.load_state_dict(ckpt_state["pretrainmodel"], strict=False)
    adapter.load_state_dict(ckpt_state["adapter"], strict=True)
    pred_head.load_state_dict(ckpt_state["pred_head"], strict=False)
    proj_head.load_state_dict(ckpt_state["proj_head"], strict=True)

    if cls_head is not None and ckpt_state.get("cls_head") is not None:
        cls_head.load_state_dict(ckpt_state["cls_head"], strict=True)

    if proto_layer is not None and ckpt_state.get("proto_layer") is not None:
        proto_layer.load_state_dict(ckpt_state["proto_layer"], strict=True)

    pretrainmodel.to(device)
    adapter.to(device)
    pred_head.to(device)
    proj_head.to(device)
    if cls_head is not None:
        cls_head.to(device)
    if proto_layer is not None:
        proto_layer.to(device)
