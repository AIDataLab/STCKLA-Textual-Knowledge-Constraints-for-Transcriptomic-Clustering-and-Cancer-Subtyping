
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

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

def run_epoch_end_eval(
    epoch_idx: int,
    pretrainmodel: nn.Module,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    proj_head: nn.Module,
    cls_head: Optional[nn.Module],
    proto_layer: Optional[nn.Module],
    gexpr_aligned,
    device: torch.device,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
):
    model_states = {
        "pretrainmodel": pretrainmodel.training,
        "adapter": adapter.training,
        "proj_head": proj_head.training,
        "cls_head": (cls_head.training if cls_head is not None else None),
        "proto_layer": (proto_layer.training if proto_layer is not None else None),
    }

    print("\n" + "=" * 80)
    print(f"[EVAL] epoch {epoch_idx+1} finished, start inference + clustering ...")
    print("=" * 80)

    old_save_path = str(args.infer_save_path)
    old_ckpt_name = str(args.infer_ckpt_name)

    if bool(getattr(args, "infer_every_epoch_subdir", True)):
        args.infer_save_path = os.path.join(old_save_path, f"epoch_{epoch_idx+1:03d}")
    args.infer_ckpt_name = f"{old_ckpt_name}_epoch{epoch_idx+1:03d}"

    rng_state = None
    if bool(getattr(args, "restore_rng_after_eval", True)):
        rng_state = get_rng_state_all()

    infer_and_cluster_after_train(
        pretrainmodel=pretrainmodel,
        pretrainconfig=pretrainconfig,
        adapter=adapter,
        proj_head=proj_head,
        cls_head=cls_head,
        proto_layer=proto_layer,
        gexpr_aligned=gexpr_aligned,
        device=device,
        args=args,
        qwen_prior_pack=qwen_prior_pack,
    )

    if rng_state is not None:
        set_rng_state_all(rng_state)

    args.infer_save_path = old_save_path
    args.infer_ckpt_name = old_ckpt_name

    pretrainmodel.train(model_states["pretrainmodel"])
    adapter.train(model_states["adapter"])
    proj_head.train(model_states["proj_head"])
    if cls_head is not None and model_states["cls_head"] is not None:
        cls_head.train(model_states["cls_head"])
    if proto_layer is not None and model_states["proto_layer"] is not None:
        proto_layer.train(model_states["proto_layer"])


# Trainable policy
def set_trainable(pretrainmodel: nn.Module, args):
    for p in pretrainmodel.parameters():
        p.requires_grad_(False)

    if args.freeze_scfoundation:
        print("[FREEZE] scFoundation frozen; train adapter+heads only.")
        return

    enc = getattr(pretrainmodel, "encoder", None)
    layers = getattr(enc, "transformer_encoder", None) if enc is not None else None
    if layers is None:
        print("[FREEZE][WARN] encoder.transformer_encoder not found; keep frozen.")
        return

    n = len(layers)
    if n <= 0:
        print("[FREEZE][WARN] encoder has 0 layers; keep frozen.")
        return

    mode = str(getattr(args, "unfreeze_mode", "second_last")).lower().strip()

    if mode == "second_last":
        idx = max(0, n - 2)
        for p in layers[idx].parameters():
            p.requires_grad_(True)
        print(f"[FREEZE] unfreeze encoder second-last layer only: idx={idx} / total={n}")

    elif mode == "last_n":
        k = max(1, int(getattr(args, "unfreeze_last_n", 1)))
        start = max(0, n - k)
        for idx in range(start, n):
            for p in layers[idx].parameters():
                p.requires_grad_(True)
        print(f"[FREEZE] unfreeze encoder last {n - start} layers: idx={list(range(start, n))} / total={n}")

    else:
        raise ValueError(f"Unknown unfreeze_mode={mode}, expected 'second_last' or 'last_n'")


# Forward helpers
def _prepare_encoder_inputs(
    batch_x: torch.Tensor,
    adapter: nn.Module,
    pretrainconfig: Dict[str, Any],
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[torch.Tensor] = None,
):
    batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-args.input_clip, args.input_clip)

    with torch.cuda.amp.autocast(enabled=False):
        gene_x = adapter(batch_x.float()) if args.use_adapter else batch_x.float()
    gene_x = torch.nan_to_num(gene_x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-args.input_clip, args.input_clip)

    x_all = build_x_all(
        gene_x,
        pre_normalized=str(args.pre_normalized),
        totalcount_mode=str(args.totalcount_mode),
        eps=float(args.valid_eps),
    )

    x_all_for_visible = x_all
    qwen_visible_info = None
    if bool(getattr(args, "qwen_visible_inject", False)):
        x_all_for_visible, qwen_visible_info = apply_qwen_visible_gene_injection(
            x_all=x_all,
            qwen_prior_pack=qwen_prior_pack,
            sample_indices=sample_indices,
            args=args,
        )

    (
        encoder_data,
        encoder_pos_ids,
        encoder_pad,
        encoder_labels,
        decoder_data,
        decoder_pad,
        data_raw,
        mask_labels,
        decoder_pos_ids,
    ) = load_mod.getEncoerDecoderData(x_all_for_visible.float(), x_all.float(), pretrainconfig)

    if qwen_visible_info is not None and bool(qwen_visible_info.get("active", False)):
        encoder_data, encoder_labels = restore_encoder_values_after_qwen_visible_injection(
            encoder_data=encoder_data,
            encoder_labels=encoder_labels,
            encoder_pos_ids=encoder_pos_ids,
            encoder_pad=encoder_pad,
            x_all_original=x_all,
        )

    return {
        "x_all": x_all,
        "encoder_data": encoder_data,
        "encoder_pos_ids": encoder_pos_ids,
        "encoder_pad": encoder_pad,
        "encoder_labels": encoder_labels,
        "decoder_data": decoder_data,
        "decoder_pad": decoder_pad,
        "data_raw": data_raw,
        "mask_labels": mask_labels,
        "decoder_pos_ids": decoder_pos_ids,
    }


def encode_and_project_full(
    pretrainmodel: nn.Module,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    proj_head: nn.Module,
    cls_head: Optional[nn.Module],
    batch_x: torch.Tensor,
    device: torch.device,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[torch.Tensor] = None,
):
    pack = _prepare_encoder_inputs(
        batch_x, adapter, pretrainconfig, args,
        qwen_prior_pack=qwen_prior_pack,
        sample_indices=sample_indices,
    )

    encoder_data = pack["encoder_data"]
    encoder_pos_ids = pack["encoder_pos_ids"]
    encoder_pad = pack["encoder_pad"]
    encoder_labels = pack["encoder_labels"]
    decoder_data = pack["decoder_data"]
    decoder_pad = pack["decoder_pad"]
    data_raw = pack["data_raw"]
    mask_labels = pack["mask_labels"]
    decoder_pos_ids = pack["decoder_pos_ids"]

    with sdp_kernel_ctx(device, args.force_sdp):
        with torch.cuda.amp.autocast(**amp_autocast_kwargs(args.amp, args.amp_dtype)):
            pred_full = pretrainmodel(
                encoder_data,
                encoder_pad,
                encoder_pos_ids,
                encoder_labels,
                decoder_data,
                False,
                mask_labels,
                decoder_pos_ids,
                decoder_pad,
                output_attentions=False,
            )

            x_tok = pretrainmodel.token_emb(torch.unsqueeze(encoder_data, 2).float(), output_weight=0)
            pos_emb = pretrainmodel.pos_emb(encoder_pos_ids)
            h = pretrainmodel.encoder(x_tok + pos_emb, encoder_pad)
            pooled = pool_tokens_like_train(h)
            z = proj_head(pooled.float())
            logits = cls_head(pooled.float()) if cls_head is not None else None

    m = mask_labels.bool()
    diff = pred_full[m] - data_raw[m]
    mae_loss = torch.zeros((), device=device, dtype=pred_full.dtype) if diff.numel() == 0 else (diff * diff).mean()

    return {
        "x_all": pack["x_all"],
        "encoder_pad": encoder_pad,
        "mask_labels": mask_labels,
        "mae_loss": mae_loss,
        "pooled": pooled,
        "z": z,
        "logits": logits,
        "enc_nonpad": int((~encoder_pad).sum().item()),
    }


def encode_and_project_light(
    pretrainmodel: nn.Module,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    proj_head: nn.Module,
    cls_head: Optional[nn.Module],
    batch_x: torch.Tensor,
    device: torch.device,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[torch.Tensor] = None,
):
    pack = _prepare_encoder_inputs(
        batch_x, adapter, pretrainconfig, args,
        qwen_prior_pack=qwen_prior_pack,
        sample_indices=sample_indices,
    )

    encoder_data = pack["encoder_data"]
    encoder_pos_ids = pack["encoder_pos_ids"]
    encoder_pad = pack["encoder_pad"]
    mask_labels = pack["mask_labels"]

    with sdp_kernel_ctx(device, args.force_sdp):
        with torch.cuda.amp.autocast(**amp_autocast_kwargs(args.amp, args.amp_dtype)):
            x_tok = pretrainmodel.token_emb(torch.unsqueeze(encoder_data, 2).float(), output_weight=0)
            pos_emb = pretrainmodel.pos_emb(encoder_pos_ids)
            h = pretrainmodel.encoder(x_tok + pos_emb, encoder_pad)
            pooled = pool_tokens_like_train(h)
            z = proj_head(pooled.float())
            logits = cls_head(pooled.float()) if cls_head is not None else None

    zero = torch.zeros((), device=device, dtype=pooled.dtype)

    return {
        "x_all": pack["x_all"],
        "encoder_pad": encoder_pad,
        "mask_labels": mask_labels,
        "mae_loss": zero,
        "pooled": pooled,
        "z": z,
        "logits": logits,
        "enc_nonpad": int((~encoder_pad).sum().item()),
    }


# Embedding monitor
def compute_embeddings_for_monitor(
    pretrainmodel: nn.Module,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    proj_head: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    max_samples: int,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    pretrainmodel_was_training = pretrainmodel.training
    adapter_was_training = adapter.training
    proj_was_training = proj_head.training

    pretrainmodel.eval()
    adapter.eval()
    proj_head.eval()

    embs = []
    n = min(int(max_samples), int(x_np.shape[0]))
    if sample_indices is None:
        sample_indices_arr = np.arange(int(x_np.shape[0]), dtype=np.int64)
    else:
        sample_indices_arr = np.asarray(sample_indices, dtype=np.int64)
        if sample_indices_arr.shape[0] != int(x_np.shape[0]):
            raise RuntimeError(
                f"[MON] sample_indices length={sample_indices_arr.shape[0]} != x_np rows={x_np.shape[0]}"
            )

    with torch.no_grad():
        for i in range(n):
            gene_x = torch.tensor(x_np[i], device=device).unsqueeze(0)
            sample_idx_t = torch.tensor([int(sample_indices_arr[i])], device=device, dtype=torch.long)
            out = encode_and_project_light(
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                cls_head=None,
                batch_x=gene_x,
                device=device,
                args=args,
                qwen_prior_pack=qwen_prior_pack,
                sample_indices=sample_idx_t,
            )
            embs.append(out["z"].detach().float().cpu().numpy())

    emb = np.squeeze(np.array(embs))
    if emb.ndim != 2:
        raise RuntimeError(f"[MON] embedding must be 2D, got {emb.shape}")

    if pretrainmodel_was_training:
        pretrainmodel.train(True)
    if adapter_was_training:
        adapter.train(True)
    if proj_was_training:
        proj_head.train(True)

    return emb.astype(np.float32)


def kmeans_monitor(emb: np.ndarray, k: int = 32, seed: int = 0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20, max_iter=300)
    pred = km.fit_predict(emb)
    inertia = float(km.inertia_)
    return {
        "kmeans_inertia": inertia,
        "n": float(emb.shape[0]),
        "d": float(emb.shape[1]),
        "k": float(k),
        "uniq_clusters": float(len(np.unique(pred)))
    }

def _prepare_embedding_for_cluster_metrics(
    emb: np.ndarray,
    *,
    seed: int,
    cluster_prep: str = "raw",
    pca_dim: int = 50,
) -> np.ndarray:
    """Apply checkpoint KMeans preprocessing to stability and quality metrics."""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise RuntimeError(f"[STABILITY] emb must be 2D, got shape={emb.shape}")

    x = emb
    prep = str(cluster_prep).lower().strip()

    if prep == "raw":
        pass
    elif prep == "l2":
        from sklearn.preprocessing import normalize
        x = normalize(x)
    elif prep == "pca":
        from sklearn.decomposition import PCA
        d = min(int(pca_dim), x.shape[1], x.shape[0] - 1)
        if d >= 2:
            x = PCA(n_components=d, random_state=int(seed)).fit_transform(x)
    elif prep == "pca_l2":
        from sklearn.preprocessing import normalize
        from sklearn.decomposition import PCA
        d = min(int(pca_dim), x.shape[1], x.shape[0] - 1)
        if d >= 2:
            x = PCA(n_components=d, random_state=int(seed)).fit_transform(x)
        x = normalize(x)
    else:
        raise ValueError(f"[STABILITY] unknown cluster_prep={cluster_prep}")

    return np.asarray(x, dtype=np.float32)


def cluster_predict_for_checkpoint_stability(
    emb: np.ndarray,
    *,
    k: int,
    seed: int,
    cluster_prep: str = "raw",
    pca_dim: int = 50,
) -> np.ndarray:

    x = _prepare_embedding_for_cluster_metrics(
        emb,
        seed=int(seed),
        cluster_prep=str(cluster_prep),
        pca_dim=int(pca_dim),
    )

    from sklearn.cluster import KMeans
    km = KMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=20,
        max_iter=300,
    )
    pred = km.fit_predict(x)
    return np.asarray(pred, dtype=np.int64)


def compute_label_free_cluster_quality_for_checkpoint(
    emb: np.ndarray,
    pred: np.ndarray,
    *,
    k: int,
    seed: int,
    cluster_prep: str,
    pca_dim: int,
    args,
) -> Dict[str, float]:
    """Compute cluster-quality metrics from embeddings and KMeans assignments."""
    emb = np.asarray(emb, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)

    if emb.ndim != 2 or pred.shape[0] != emb.shape[0] or emb.shape[0] <= 2:
        return {
            "quality_valid": 0.0,
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "balance_entropy": 0.0,
            "min_cluster_frac": 0.0,
            "max_cluster_frac": 1.0,
            "cluster_size_cv": float("inf"),
            "collapse_penalty": 1.0,
            "uniq_clusters": float(len(np.unique(pred)) if pred.size else 0),
        }

    n = int(emb.shape[0])
    k = int(k)
    uniq, counts = np.unique(pred, return_counts=True)
    probs = counts.astype(np.float64) / max(1, counts.sum())
    min_frac = float(probs.min()) if probs.size else 0.0
    max_frac = float(probs.max()) if probs.size else 1.0
    mean_size = float(np.mean(counts)) if counts.size else 0.0
    size_cv = float(np.std(counts) / max(mean_size, 1e-8)) if counts.size else float("inf")
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / max(np.log(max(2, len(probs))), 1e-12)) if probs.size > 1 else 0.0

    min_allowed = float(getattr(args, "exposed_cluster_min_cluster_frac", 0.005))
    max_allowed = float(getattr(args, "exposed_cluster_max_cluster_frac", 0.80))
    collapse_penalty = 0.0
    if min_frac < min_allowed:
        collapse_penalty += float((min_allowed - min_frac) / max(min_allowed, 1e-8))
    if max_frac > max_allowed:
        collapse_penalty += float((max_frac - max_allowed) / max(1.0 - max_allowed, 1e-8))
    if len(uniq) < k:
        collapse_penalty += float((k - len(uniq)) / max(1, k))

    sil = float("nan")
    ch = float("nan")
    db = float("nan")

    if len(uniq) >= 2 and len(uniq) < n:
        x = _prepare_embedding_for_cluster_metrics(
            emb,
            seed=int(seed),
            cluster_prep=str(cluster_prep),
            pca_dim=int(pca_dim),
        )
        # Quality metrics can be expensive for large scRNA datasets.
        # Subsample deterministically; keep only if at least two clusters remain.
        max_q = int(getattr(args, "exposed_cluster_quality_max_samples", 3000))
        if max_q > 0 and x.shape[0] > max_q:
            rng = np.random.RandomState(int(seed) + 303917)
            qidx = np.sort(rng.choice(x.shape[0], size=max_q, replace=False)).astype(np.int64)
            xq = x[qidx]
            pq = pred[qidx]
        else:
            xq = x
            pq = pred

        uq = np.unique(pq)
        if len(uq) >= 2 and len(uq) < xq.shape[0]:
            try:
                from sklearn.metrics import silhouette_score
                sil = float(silhouette_score(xq, pq, metric="euclidean"))
            except Exception:
                sil = float("nan")
            try:
                from sklearn.metrics import calinski_harabasz_score
                ch = float(calinski_harabasz_score(xq, pq))
            except Exception:
                ch = float("nan")
            try:
                from sklearn.metrics import davies_bouldin_score
                db = float(davies_bouldin_score(xq, pq))
            except Exception:
                db = float("nan")

    return {
        "quality_valid": 1.0,
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
        "balance_entropy": float(entropy),
        "min_cluster_frac": float(min_frac),
        "max_cluster_frac": float(max_frac),
        "cluster_size_cv": float(size_cv),
        "collapse_penalty": float(collapse_penalty),
        "uniq_clusters": float(len(uniq)),
    }


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




# Qwen prior helpers
class ExprSemiDatasetWithIndex(torch.utils.data.Dataset):
    """Return (x, y, is_labeled, idx) for sample-aligned Qwen prior lookup."""
    def __init__(self, x_np, y_id=None, labeled_mask=None, unlabeled_label_value=-1):
        self.x = torch.as_tensor(np.asarray(x_np, dtype=np.float32), dtype=torch.float32)
        n = int(self.x.shape[0])
        if y_id is None:
            y = np.full(n, int(unlabeled_label_value), dtype=np.int64)
        else:
            y = np.asarray(y_id, dtype=np.int64).copy()
            if y.shape[0] != n:
                raise ValueError(f"[QWEN-DATASET] y_id length={y.shape[0]} != n={n}")
        if labeled_mask is None:
            is_lab = np.ones(n, dtype=bool) if y_id is not None else np.zeros(n, dtype=bool)
        else:
            is_lab = np.asarray(labeled_mask, dtype=bool).copy()
            if is_lab.shape[0] != n:
                raise ValueError(f"[QWEN-DATASET] labeled_mask length={is_lab.shape[0]} != n={n}")
        y[~is_lab] = int(unlabeled_label_value)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.is_lab = torch.as_tensor(is_lab, dtype=torch.bool)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.is_lab[idx], torch.tensor(idx, dtype=torch.long)


def _decode_npz_string_array(x):
    arr = np.asarray(x)
    out = []
    for v in arr.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def load_qwen_context_prior_npz(npz_path: str, sample_index, num_classes: int) -> Optional[Dict[str, Any]]:
    """Load module priors, confidence scores, and sample IDs from an NPZ file.

    Accepted keys:
      - module_prior_probability or prior_prob: [N, M]
      - sample_prior_confidence or prior_confidence: [N]
      - sample_ids or sample_index: [N]
      - module_ids: [M]

    Map numeric module class IDs to [0, num_classes); ignore unknown IDs.
    """
    if not npz_path:
        return None
    p = Path(str(npz_path))
    if not p.exists():
        raise FileNotFoundError(f"[QWEN-PRIOR] npz not found: {p}")

    z = np.load(str(p), allow_pickle=True)
    keys = set(z.files)

    # probability key compatibility
    if "prior_prob" in keys:
        prob_key = "prior_prob"
    elif "module_prior_probability" in keys:
        prob_key = "module_prior_probability"
    elif "module_raw_probability" in keys:
        prob_key = "module_raw_probability"
        print(
            "[QWEN-PRIOR][WARN] module_prior_probability not found; "
            "fallback to module_raw_probability.",
            flush=True,
        )
    else:
        raise KeyError(
            f"[QWEN-PRIOR] no probability matrix found in {p}; "
            f"expected one of prior_prob/module_prior_probability/module_raw_probability; keys={list(z.keys())}"
        )

    prior_prob = np.asarray(z[prob_key], dtype=np.float32)
    if prior_prob.ndim != 2:
        raise ValueError(f"[QWEN-PRIOR] {prob_key} must be 2D, got {prior_prob.shape}")

    module_activity_raw = None
    if "module_activity" in keys:
        module_activity_raw = np.asarray(z["module_activity"], dtype=np.float32)
        if module_activity_raw.ndim != 2:
            raise ValueError(f"[QWEN-PRIOR] module_activity must be 2D, got {module_activity_raw.shape}")

    n_current = len(sample_index)
    current_samples = [str(x) for x in list(sample_index)]

    # sample id key compatibility
    sample_key = None
    if "sample_index" in keys:
        sample_key = "sample_index"
    elif "sample_ids" in keys:
        sample_key = "sample_ids"

    if sample_key is not None:
        prior_samples = _decode_npz_string_array(z[sample_key])
        if prior_samples != current_samples:
            pos = {s: i for i, s in enumerate(prior_samples)}
            miss = [s for s in current_samples if s not in pos]
            if miss:
                # Align prior sample IDs with gexpr_aligned.index; reject mismatches.
                raise ValueError(
                    f"[QWEN-PRIOR] {sample_key} mismatch: missing {len(miss)} current samples in prior, "
                    f"first_missing={miss[:5]}, prior_first={prior_samples[:5]}, current_first={current_samples[:5]}"
                )
            order = np.asarray([pos[s] for s in current_samples], dtype=np.int64)
            prior_prob = prior_prob[order]
            if module_activity_raw is not None:
                module_activity_raw = module_activity_raw[order]
        else:
            order = None
    else:
        if prior_prob.shape[0] != n_current:
            raise ValueError(
                f"[QWEN-PRIOR] no sample_index/sample_ids and N mismatch: "
                f"prior={prior_prob.shape[0]} current={n_current}"
            )
        order = None

    # confidence key compatibility
    if "sample_prior_confidence" in keys:
        conf_raw = np.asarray(z["sample_prior_confidence"], dtype=np.float32)
    elif "prior_confidence" in keys:
        conf_raw = np.asarray(z["prior_confidence"], dtype=np.float32)
    else:
        conf_raw = prior_prob.max(axis=1).astype(np.float32)

    if sample_key is not None and order is not None:
        conf_raw = conf_raw[order]

    if prior_prob.shape[0] != n_current:
        raise ValueError(f"[QWEN-PRIOR] prior N={prior_prob.shape[0]} != current N={n_current}")

    # module/class id alignment
    if "module_ids" in keys:
        module_ids = _decode_npz_string_array(z["module_ids"])
    else:
        module_ids = [str(i) for i in range(prior_prob.shape[1])]

    aligned_prob = np.zeros((n_current, int(num_classes)), dtype=np.float32)
    aligned_activity = np.zeros((n_current, int(num_classes)), dtype=np.float32)
    module_reliability = np.zeros((int(num_classes),), dtype=np.float32)

    # Align module gene weights to class IDs for visible-gene selection: [C, G].
    module_gene_weight_raw = None
    module_gene_mask_raw = None
    aligned_module_gene_weight = None
    aligned_module_gene_mask = None
    if "module_gene_weight" in keys:
        module_gene_weight_raw = np.asarray(z["module_gene_weight"], dtype=np.float32)
        if module_gene_weight_raw.ndim == 2 and module_gene_weight_raw.shape[0] == prior_prob.shape[1]:
            gdim = int(module_gene_weight_raw.shape[1])
            aligned_module_gene_weight = np.zeros((int(num_classes), gdim), dtype=np.float32)
            if "module_gene_mask" in keys:
                module_gene_mask_raw = np.asarray(z["module_gene_mask"], dtype=np.float32)
                if module_gene_mask_raw.shape != module_gene_weight_raw.shape:
                    module_gene_mask_raw = None
            if module_gene_mask_raw is None:
                module_gene_mask_raw = (np.abs(module_gene_weight_raw) > 0).astype(np.float32)
            aligned_module_gene_mask = np.zeros((int(num_classes), gdim), dtype=np.float32)
        else:
            print(
                f"[QWEN-VISIBLE][WARN] ignore module_gene_weight with shape "
                f"{getattr(module_gene_weight_raw, 'shape', None)}; expected [M,G] with M={prior_prob.shape[1]}",
                flush=True,
            )
            module_gene_weight_raw = None
            module_gene_mask_raw = None
    if "module_reliability" in keys:
        rel_raw = np.asarray(z["module_reliability"], dtype=np.float32).reshape(-1)
    else:
        rel_raw = np.ones((prior_prob.shape[1],), dtype=np.float32)

    used_modules = []
    for j, mid in enumerate(module_ids):
        try:
            cid = int(float(str(mid)))
        except Exception:
            continue
        if 0 <= cid < int(num_classes):
            aligned_prob[:, cid] += prior_prob[:, j]
            if module_activity_raw is not None and j < module_activity_raw.shape[1]:
                aligned_activity[:, cid] += module_activity_raw[:, j]
            if j < rel_raw.shape[0]:
                module_reliability[cid] = max(float(module_reliability[cid]), float(rel_raw[j]))
            if aligned_module_gene_weight is not None and module_gene_weight_raw is not None and j < module_gene_weight_raw.shape[0]:
                aligned_module_gene_weight[cid] += module_gene_weight_raw[j].astype(np.float32)
                aligned_module_gene_mask[cid] = np.maximum(
                    aligned_module_gene_mask[cid],
                    module_gene_mask_raw[j].astype(np.float32),
                )
            used_modules.append(str(mid))

    row_sum = aligned_prob.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 1e-12
    if valid.any():
        aligned_prob[valid] = aligned_prob[valid] / row_sum[valid]
    if (~valid).any():
        aligned_prob[~valid] = 1.0 / float(num_classes)

    conf = np.asarray(conf_raw, dtype=np.float32).reshape(-1)
    if conf.shape[0] != n_current:
        raise ValueError(f"[QWEN-PRIOR] confidence length={conf.shape[0]} != N={n_current}")
    conf = np.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)

    print(
        f"[QWEN-PRIOR] loaded {p} | prob_key={prob_key} sample_key={sample_key} "
        f"prior_prob={aligned_prob.shape} used_modules={used_modules} "
        f"mean_conf={float(conf.mean()):.6f} max_conf={float(conf.max()):.6f} "
        f"visible_gene_weight={None if aligned_module_gene_weight is None else aligned_module_gene_weight.shape}",
        flush=True,
    )
    return {
        "active": True,
        "path": str(p),
        "prior_prob": aligned_prob.astype(np.float32),
        "module_activity": aligned_activity.astype(np.float32),
        "module_reliability": module_reliability.astype(np.float32),
        "aligned_module_gene_weight": None if aligned_module_gene_weight is None else aligned_module_gene_weight.astype(np.float32),
        "aligned_module_gene_mask": None if aligned_module_gene_mask is None else aligned_module_gene_mask.astype(np.float32),
        "prior_confidence": conf.astype(np.float32),
        "used_modules": used_modules,
        "raw_module_ids": module_ids,
        "prob_key": prob_key,
        "sample_key": sample_key,
    }

def qwen_prior_weight_for_epoch(args, ep: int) -> float:
    base = float(getattr(args, "qwen_prior_loss_weight", 0.0))
    if base <= 0:
        return 0.0
    warm = int(getattr(args, "qwen_prior_warmup_epochs", 0))
    if warm <= 0:
        return base
    # Apply linear warmup using the zero-based epoch index.
    scale = min(1.0, float(ep + 1) / float(max(1, warm)))
    return base * scale


def make_qwen_prior_batch_mask(batch_is_lab: torch.Tensor, prior_conf: torch.Tensor, args) -> torch.Tensor:
    apply_to = str(getattr(args, "qwen_prior_apply_to", "unlabeled")).lower().strip()
    conf_thr = float(getattr(args, "qwen_prior_conf_threshold", 0.15))
    mask = prior_conf >= conf_thr
    if apply_to == "unlabeled":
        mask = mask & (~batch_is_lab.bool())
    elif apply_to == "labeled":
        mask = mask & batch_is_lab.bool()
    elif apply_to == "all":
        pass
    else:
        raise ValueError(f"[QWEN-PRIOR] unknown qwen_prior_apply_to={apply_to}")
    return mask


def qwen_prior_kl_loss_from_logits(logits: torch.Tensor, prior_prob: torch.Tensor, prior_conf: torch.Tensor, mask: torch.Tensor, args) -> torch.Tensor:
    if logits is None or prior_prob is None:
        return torch.zeros((), device=prior_conf.device, dtype=torch.float32)
    if mask is None or mask.numel() == 0 or not bool(mask.any().item()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    eps = 1e-8
    temp = max(float(getattr(args, "qwen_prior_temperature", 1.0)), eps)
    prior = prior_prob.clamp_min(eps)
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)
    logp = F.log_softmax(logits.float() / temp, dim=-1)
    kl = F.kl_div(logp, prior.float(), reduction="none").sum(dim=-1)
    w = prior_conf.float().clamp_min(float(getattr(args, "qwen_prior_min_conf", 1e-6))).clamp_max(1.0)
    kl = kl[mask] * w[mask]
    denom = w[mask].sum().clamp_min(eps)
    return (kl.sum() / denom).to(logits.dtype)


def qwen_prior_kl_loss_from_prototypes(z_embed: torch.Tensor, proto_layer: Optional[nn.Module], prior_prob: torch.Tensor, prior_conf: torch.Tensor, mask: torch.Tensor, args) -> torch.Tensor:
    if proto_layer is None or prior_prob is None:
        return torch.zeros((), device=z_embed.device, dtype=z_embed.dtype)
    if mask is None or mask.numel() == 0 or not bool(mask.any().item()):
        return torch.zeros((), device=z_embed.device, dtype=z_embed.dtype)
    protos = proto_layer()
    z_norm = F.normalize(z_embed.float(), dim=-1)
    p_norm = F.normalize(protos.float(), dim=-1)
    temp = max(float(getattr(args, "qwen_prior_proto_temperature", 0.10)), 1e-8)
    logits = torch.matmul(z_norm, p_norm.t()) / temp
    return qwen_prior_kl_loss_from_logits(logits, prior_prob, prior_conf, mask, args).to(z_embed.dtype)


class QwenModuleActivityHead(nn.Module):
    """Predict continuous Qwen module activities from the projection embedding."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, z_embed: torch.Tensor) -> torch.Tensor:
        return self.net(z_embed.float())


def qwen_module_activity_loss(
    z_embed: torch.Tensor,
    module_head: Optional[nn.Module],
    target_activity: Optional[torch.Tensor],
    prior_conf: Optional[torch.Tensor],
    mask: torch.Tensor,
    args,
) -> torch.Tensor:
    if module_head is None or target_activity is None:
        return torch.zeros((), device=z_embed.device, dtype=z_embed.dtype)
    if mask is None or mask.numel() == 0 or not bool(mask.any().item()):
        return torch.zeros((), device=z_embed.device, dtype=z_embed.dtype)

    pred = module_head(z_embed).float()
    target = target_activity.float()
    clip = float(getattr(args, "qwen_activity_target_clip", 5.0))
    if clip > 0:
        target = target.clamp(-clip, clip)

    # Compare activity profiles after per-sample normalization.
    pred = F.layer_norm(pred, (pred.shape[-1],))
    target = F.layer_norm(target, (target.shape[-1],))

    per = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if bool(getattr(args, "qwen_activity_use_conf_weight", False)) and prior_conf is not None:
        min_w = float(getattr(args, "qwen_activity_min_sample_weight", 0.25))
        w = min_w + (1.0 - min_w) * prior_conf.float().clamp(0.0, 1.0)
    else:
        w = torch.ones_like(per, dtype=per.dtype, device=per.device)
    per = per[mask] * w[mask]
    denom = w[mask].sum().clamp_min(1e-8)
    return (per.sum() / denom).to(z_embed.dtype)



# Qwen-guided visible gene selection
def _ensure_qwen_visible_tensors(qwen_prior_pack: Optional[Dict[str, Any]], device: torch.device):
    """Cache tensors for visible-gene selection on the target device.

    Required fields:
      - aligned_module_gene_weight: [C, G]
      - aligned_module_gene_mask: [C, G]
      - module_activity: [N, C]

    These tensors are separate from the prior-loss tensors.
    """
    if qwen_prior_pack is None or not qwen_prior_pack.get("active", False):
        return None
    if "aligned_module_gene_weight" not in qwen_prior_pack or qwen_prior_pack.get("aligned_module_gene_weight") is None:
        return None

    cache = qwen_prior_pack.setdefault("_visible_tensor_cache", {})
    key = str(device)
    if key in cache:
        return cache[key]

    weight_np = np.asarray(qwen_prior_pack.get("aligned_module_gene_weight"), dtype=np.float32)
    if weight_np.ndim != 2 or weight_np.shape[1] <= 0:
        return None

    mask_np = qwen_prior_pack.get("aligned_module_gene_mask", None)
    if mask_np is None:
        mask_np = (np.abs(weight_np) > 0).astype(np.float32)
    else:
        mask_np = np.asarray(mask_np, dtype=np.float32)
        if mask_np.shape != weight_np.shape:
            mask_np = (np.abs(weight_np) > 0).astype(np.float32)

    activity_np = qwen_prior_pack.get("module_activity", None)
    if activity_np is not None:
        activity_np = np.asarray(activity_np, dtype=np.float32)
        if activity_np.ndim != 2 or activity_np.shape[1] != weight_np.shape[0]:
            activity_np = None

    reliability_np = qwen_prior_pack.get("module_reliability", None)
    if reliability_np is None:
        reliability_np = np.ones((weight_np.shape[0],), dtype=np.float32)
    else:
        reliability_np = np.asarray(reliability_np, dtype=np.float32).reshape(-1)
        if reliability_np.shape[0] != weight_np.shape[0]:
            reliability_np = np.ones((weight_np.shape[0],), dtype=np.float32)

    tensors = {
        "weight": torch.as_tensor(weight_np, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(mask_np, dtype=torch.float32, device=device),
        "activity": torch.as_tensor(activity_np, dtype=torch.float32, device=device) if activity_np is not None else None,
        "reliability": torch.as_tensor(reliability_np, dtype=torch.float32, device=device),
    }
    cache[key] = tensors
    return tensors


def _resolve_qwen_visible_quota_for_sample(x_gene: torch.Tensor, gene_len: int, args) -> int:
    """Compute the Qwen-visible gene quota from configured limits.

    Fixed mode caps the quota at gene_len. Auto and adaptive modes scale
    the quota by valid gene count and apply the configured bounds.
    """
    base_quota = max(0, int(getattr(args, "qwen_visible_quota", 0)))
    if base_quota <= 0 or gene_len <= 0:
        return 0
    mode = str(getattr(args, "qwen_visible_quota_mode", "fixed")).lower().strip()
    if mode == "fixed":
        return int(min(base_quota, gene_len))
    if mode not in {"auto", "adaptive"}:
        return int(min(base_quota, gene_len))

    eps = float(getattr(args, "qwen_visible_eps", 1e-8))
    valid = torch.isfinite(x_gene[:gene_len])
    if bool(getattr(args, "qwen_visible_require_nonzero", True)):
        valid = valid & (x_gene[:gene_len].abs() > eps)
    nonzero = int(valid.sum().item())
    if nonzero <= 0:
        return 0

    frac = max(0.0, float(getattr(args, "qwen_visible_quota_fraction", 0.12)))
    qmin = max(0, int(getattr(args, "qwen_visible_quota_min", 80)))
    qmax = max(1, int(getattr(args, "qwen_visible_quota_max", base_quota)))
    q = int(round(nonzero * frac))
    q = max(qmin, q)
    q = min(q, qmax, base_quota, nonzero, gene_len)
    return int(max(0, q))

def _select_qwen_genes_for_one_sample(
    *,
    x_gene: torch.Tensor,
    sample_index: Optional[int],
    tensors: Dict[str, torch.Tensor],
    args,
) -> torch.Tensor:
    """Return Qwen-visible gene indices as a 1D LongTensor in [0, G).

    Selection uses module gene weights and sample module activities.
    """
    weight = tensors["weight"]
    mask = tensors["mask"]
    activity = tensors.get("activity", None)
    reliability = tensors.get("reliability", None)

    gene_len = min(int(x_gene.shape[0]), int(weight.shape[1]))
    quota = _resolve_qwen_visible_quota_for_sample(x_gene=x_gene, gene_len=gene_len, args=args)
    if quota <= 0 or gene_len <= 0:
        return torch.empty((0,), dtype=torch.long, device=x_gene.device)

    mode = str(getattr(args, "qwen_visible_mode", "sample_module")).lower().strip()
    score = None

    if mode == "sample_module" and activity is not None and sample_index is not None:
        if 0 <= int(sample_index) < int(activity.shape[0]):
            act = activity[int(sample_index)].float().abs()
            if reliability is not None:
                act = act * reliability.float().clamp_min(0.0)
            m_top = min(max(1, int(getattr(args, "qwen_visible_module_topk", 2))), int(weight.shape[0]))
            if torch.isfinite(act).all() and float(act.max().item()) > 0:
                _, mod_idx = torch.topk(act, k=m_top, largest=True)
                module_weight = weight.index_select(0, mod_idx)[:, :gene_len].abs()
                module_mask = mask.index_select(0, mod_idx)[:, :gene_len].clamp_min(0.0)
                module_scale = act.index_select(0, mod_idx).view(-1, 1).clamp_min(0.0)
                score = (module_weight * module_mask * module_scale).max(dim=0).values

    if score is None:
        # Fallback/global mode: use the strongest Qwen gene weight across modules.
        score = (weight[:, :gene_len].abs() * mask[:, :gene_len].clamp_min(0.0)).max(dim=0).values

    score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)

    if bool(getattr(args, "qwen_visible_require_nonzero", True)):
        eps = float(getattr(args, "qwen_visible_eps", 1e-8))
        valid = torch.isfinite(x_gene[:gene_len]) & (x_gene[:gene_len].abs() > eps)
        score = torch.where(valid, score, torch.full_like(score, -float("inf")))

    finite = torch.isfinite(score)
    if not bool(finite.any().item()):
        return torch.empty((0,), dtype=torch.long, device=x_gene.device)

    k = min(quota, int(finite.sum().item()))
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=x_gene.device)

    _, idx = torch.topk(score, k=k, largest=True)
    return idx.long()


def apply_qwen_visible_gene_injection(
    *,
    x_all: torch.Tensor,
    qwen_prior_pack: Optional[Dict[str, Any]],
    sample_indices: Optional[torch.Tensor],
    args,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Boost selected gene positions in a copy of x_all for token selection.

    Restore encoder expression values from the original x_all after
    getEncoerDecoderData returns.
    """
    tensors = _ensure_qwen_visible_tensors(qwen_prior_pack, x_all.device)
    if tensors is None:
        return x_all, None

    gene_len = min(int(x_all.shape[1]) - 2, int(tensors["weight"].shape[1]))
    if gene_len <= 0:
        return x_all, None

    bsz = int(x_all.shape[0])
    if sample_indices is None:
        sample_indices_list = [None for _ in range(bsz)]
    else:
        sample_indices_list = [int(v) for v in sample_indices.detach().cpu().numpy().reshape(-1).tolist()]
        if len(sample_indices_list) != bsz:
            sample_indices_list = [None for _ in range(bsz)]

    x_sel = x_all.clone()
    boost_value = float(getattr(args, "qwen_visible_boost_value", 1000000.0))
    counts = []
    for b in range(bsz):
        idx = _select_qwen_genes_for_one_sample(
            x_gene=x_all[b, :gene_len],
            sample_index=sample_indices_list[b],
            tensors=tensors,
            args=args,
        )
        counts.append(int(idx.numel()))
        if idx.numel() <= 0:
            continue
        sign = torch.sign(x_all[b, idx])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        x_sel[b, idx] = sign * boost_value

    info = {
        "active": True,
        "gene_len": int(gene_len),
        "quota": int(getattr(args, "qwen_visible_quota", 0)),
        "quota_mode": str(getattr(args, "qwen_visible_quota_mode", "fixed")),
        "quota_fraction": float(getattr(args, "qwen_visible_quota_fraction", 0.12)),
        "mode": str(getattr(args, "qwen_visible_mode", "sample_module")),
        "module_topk": int(getattr(args, "qwen_visible_module_topk", 1)),
        "selected_min": int(min(counts) if counts else 0),
        "selected_max": int(max(counts) if counts else 0),
        "selected_mean": float(np.mean(counts) if counts else 0.0),
    }

    if bool(getattr(args, "qwen_visible_debug", True)) and not bool(getattr(args, "_qwen_visible_debug_printed", False)):
        print(
            f"[QWEN-VISIBLE] enabled | mode={info['mode']} quota={info['quota']} "
            f"quota_mode={info.get('quota_mode','fixed')} module_topk={info.get('module_topk','NA')} "
            f"selected_mean={info['selected_mean']:.2f} min={info['selected_min']} max={info['selected_max']} "
            f"keep_total_len={getattr(args, 'encoder_visible_max_len', 'NA')}",
            flush=True,
        )
        setattr(args, "_qwen_visible_debug_printed", True)

    return x_sel, info


def restore_encoder_values_after_qwen_visible_injection(
    *,
    encoder_data: torch.Tensor,
    encoder_labels: torch.Tensor,
    encoder_pos_ids: torch.Tensor,
    encoder_pad: torch.Tensor,
    x_all_original: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Restore encoder values from x_all using encoder_pos_ids.

    Clamp out-of-range positions and preserve padding values.
    """
    if encoder_data is None or encoder_pos_ids is None or x_all_original is None:
        return encoder_data, encoder_labels
    if encoder_data.ndim != 2 or encoder_pos_ids.shape != encoder_data.shape:
        return encoder_data, encoder_labels

    pos = encoder_pos_ids.long().clamp(0, int(x_all_original.shape[1]) - 1)
    restored = torch.gather(x_all_original.float(), dim=1, index=pos).to(dtype=encoder_data.dtype)
    nonpad = ~encoder_pad.bool() if encoder_pad is not None else torch.ones_like(encoder_data, dtype=torch.bool)
    encoder_data = torch.where(nonpad, restored, encoder_data)
    if encoder_labels is not None and encoder_labels.shape == encoder_data.shape:
        encoder_labels = torch.where(nonpad, restored.to(dtype=encoder_labels.dtype), encoder_labels)
    return encoder_data, encoder_labels

# Train
def train_one(
    pretrainmodel,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    pred_head: nn.Module,
    cls_head: Optional[nn.Module],
    proj_head: nn.Module,
    proto_layer: Optional[nn.Module],
    x_np: np.ndarray,
    y_id: Optional[np.ndarray],
    labeled_mask: Optional[np.ndarray],
    exposed_ckpt_mask: Optional[np.ndarray],
    label2id: Optional[Dict[str, int]],
    gexpr_aligned,
    device: torch.device,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
):
    dataset = ExprSemiDatasetWithIndex(
        x_np,
        y_id=y_id,
        labeled_mask=labeled_mask,
        unlabeled_label_value=args.unlabeled_label_value,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))

    def seed_worker(worker_id):
        worker_seed = int(args.seed) + int(worker_id) + 1000
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )
    set_trainable(pretrainmodel, args)
    pretrainmodel.train(True)
    adapter.train(bool(args.use_adapter))
    pred_head.train(True)
    proj_head.train(True)
    if cls_head is not None:
        cls_head.train(True)
    if proto_layer is not None:
        proto_layer.train(True)

    qwen_module_head = None
    qwen_prior_type_for_head = str(getattr(args, "qwen_prior_loss_type", "none")).lower().strip()
    if (
        bool(getattr(args, "use_qwen_context_prior", False))
        and qwen_prior_pack is not None
        and qwen_prior_pack.get("active", False)
        and qwen_prior_type_for_head in {"activity", "activity_mse", "module_activity", "module_mse"}
    ):
        module_dim = int(np.asarray(qwen_prior_pack.get("module_activity")).shape[1])
        if module_dim <= 0:
            raise RuntimeError("[QWEN-ACTIVITY] module_activity has zero modules")
        qwen_module_head = QwenModuleActivityHead(
            in_dim=int(args.proj_dim),
            out_dim=module_dim,
            hidden_dim=int(getattr(args, "qwen_activity_hidden_dim", 128)),
            dropout=float(getattr(args, "qwen_activity_dropout", 0.10)),
        ).to(device)
        qwen_module_head.train(True)
        print(
            f"[QWEN-ACTIVITY] auxiliary head enabled | in_dim={args.proj_dim} "
            f"out_dim={module_dim} hidden={getattr(args, 'qwen_activity_hidden_dim', 128)}",
            flush=True,
        )

    backbone_params = [p for p in pretrainmodel.parameters() if p.requires_grad]

    groups = [{"params": adapter.parameters(), "lr": float(args.lr)}]
    groups.append({"params": pred_head.parameters(), "lr": float(args.lr)})
    groups.append({"params": proj_head.parameters(), "lr": float(args.lr)})
    if cls_head is not None:
        groups.append({"params": cls_head.parameters(), "lr": float(args.lr)})
    if proto_layer is not None:
        groups.append({"params": proto_layer.parameters(), "lr": float(args.lr)})
    if qwen_module_head is not None:
        groups.append({"params": qwen_module_head.parameters(), "lr": float(args.lr)})
    if len(backbone_params) > 0:
        groups.append({"params": backbone_params, "lr": float(args.lr) * float(args.backbone_lr_scale)})

    optimizer = torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))

    use_fp16 = (args.amp and device.type == "cuda" and str(args.amp_dtype).lower().strip() == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    accum_steps = max(1, int(args.accum_steps))
    optimizer.zero_grad(set_to_none=True)

    did_dbg = False
    enc_len_max_global = 0
    global_step = 0
    eval_mode = normalize_infer_cluster_eval_mode(getattr(args, "infer_cluster_eval_mode", "final"))

    best_val_score = -float("inf")
    best_ckpt_state = None
    best_val_epoch = -1
    first_self_loss_for_ckpt = None

    max_exposed_cluster_score = -float("inf")
    prev_all_cluster_pred_for_ckpt = None
    stability_idx_for_ckpt = None

    # Store epoch-level checkpoint records for post-hoc selection.
    ckpt_selection_records = []

    qwen_prior_prob_t = None
    qwen_prior_conf_t = None
    qwen_module_activity_t = None
    if bool(getattr(args, "use_qwen_context_prior", False)) and qwen_prior_pack is not None and qwen_prior_pack.get("active", False):
        qwen_prior_prob_t = torch.as_tensor(qwen_prior_pack["prior_prob"], dtype=torch.float32, device=device)
        qwen_prior_conf_t = torch.as_tensor(qwen_prior_pack["prior_confidence"], dtype=torch.float32, device=device)
        qwen_module_activity_t = torch.as_tensor(qwen_prior_pack.get("module_activity", qwen_prior_pack["prior_prob"]), dtype=torch.float32, device=device)
        if qwen_prior_prob_t.shape[0] != int(x_np.shape[0]):
            raise RuntimeError(f"[QWEN-PRIOR] N mismatch: prior={qwen_prior_prob_t.shape} x_np={x_np.shape}")
        if qwen_module_activity_t.shape[0] != int(x_np.shape[0]):
            raise RuntimeError(f"[QWEN-ACTIVITY] N mismatch: activity={qwen_module_activity_t.shape} x_np={x_np.shape}")
        print(
            f"[QWEN-PRIOR] enabled in training | type={args.qwen_prior_loss_type} "
            f"apply_to={args.qwen_prior_apply_to} weight={args.qwen_prior_loss_weight} "
            f"conf_thr={args.qwen_prior_conf_threshold} warmup={args.qwen_prior_warmup_epochs}",
            flush=True,
        )

    for ep in range(int(args.epochs)):
        if bool(getattr(args, "strict_reproducible", True)):
            epoch_seed = int(args.seed) + int(ep) * 100003
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        pbar = tqdm(loader, desc=f"[TRAIN] epoch {ep+1}/{args.epochs}")
        running = 0.0
        running_mae = 0.0
        running_ce = 0.0
        running_supcon = 0.0
        running_proto = 0.0
        running_cons = 0.0
        running_pseudo = 0.0
        running_qwen = 0.0
        sup_cnt = 0
        steps = 0
        enc_len_max_epoch = 0

        for batch_x, batch_y, batch_is_lab, batch_idx in pbar:
            global_step += 1

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_is_lab = batch_is_lab.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True).long()

            out1 = encode_and_project_full(
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                cls_head=cls_head,
                batch_x=batch_x,
                device=device,
                args=args,
                qwen_prior_pack=qwen_prior_pack,
                sample_indices=batch_idx,
            )

            batch_x_aug = augment_expression(
                batch_x,
                noise_std=float(args.aug_noise_std),
                drop_prob=float(args.aug_drop_prob),
                input_clip=float(args.input_clip),
            )

            out2 = encode_and_project_light(
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                cls_head=cls_head,
                batch_x=batch_x_aug,
                device=device,
                args=args,
                qwen_prior_pack=qwen_prior_pack,
                sample_indices=batch_idx,
            )

            enc_nonpad = out1["enc_nonpad"]
            enc_len_max_epoch = max(enc_len_max_epoch, enc_nonpad)
            enc_len_max_global = max(enc_len_max_global, enc_nonpad)

            mae_loss = out1["mae_loss"]

            ce_loss = torch.zeros((), device=device, dtype=mae_loss.dtype)
            supcon_loss = torch.zeros((), device=device, dtype=mae_loss.dtype)
            proto_loss_v = torch.zeros((), device=device, dtype=mae_loss.dtype)
            cons_loss_v = consistency_loss(out1["z"], out2["z"]).to(mae_loss.dtype)
            pseudo_loss_v = torch.zeros((), device=device, dtype=mae_loss.dtype)
            qwen_prior_loss_v = torch.zeros((), device=device, dtype=mae_loss.dtype)

            has_sup = (cls_head is not None) and bool(batch_is_lab.any().item())

            if has_sup:
                idx = torch.nonzero(batch_is_lab, as_tuple=False).squeeze(1)
                z1_l = out1["z"].index_select(0, idx)
                z2_l = out2["z"].index_select(0, idx)
                y_l = batch_y.index_select(0, idx)

                logits_l = out1["logits"].index_select(0, idx)
                ce_loss = F.cross_entropy(logits_l, y_l)

                z_sup = torch.cat([z1_l, z2_l], dim=0)
                y_sup = torch.cat([y_l, y_l], dim=0)
                supcon_loss = supervised_contrastive_loss(
                    z_sup,
                    y_sup,
                    temperature=float(args.supcon_temperature)
                ).to(mae_loss.dtype)

                if proto_layer is not None:
                    protos = proto_layer()
                    proto_loss_v = prototype_loss(
                        z1_l,
                        y_l,
                        protos,
                        temperature=float(args.proto_temperature)
                    ).to(mae_loss.dtype)

                sup_cnt += int(batch_is_lab.sum().item())

            if (
                cls_head is not None
                and proto_layer is not None
                and ep >= int(args.pseudo_label_start_epoch)
                and bool((~batch_is_lab).any().item())
            ):
                idx_u = torch.nonzero(~batch_is_lab, as_tuple=False).squeeze(1)
                if idx_u.numel() > 0:
                    logits_u = out1["logits"].index_select(0, idx_u)
                    prob_u = F.softmax(logits_u, dim=-1)
                    conf, pseudo_y = prob_u.max(dim=-1)
                    keep = conf > float(args.pseudo_label_thresh)
                    if bool(keep.any().item()):
                        logits_u_keep = logits_u[keep]
                        pseudo_y_keep = pseudo_y[keep]
                        pseudo_loss_v = F.cross_entropy(logits_u_keep, pseudo_y_keep).to(mae_loss.dtype)

            qwen_prior_weight_ep = qwen_prior_weight_for_epoch(args, ep)
            if (
                qwen_prior_conf_t is not None
                and qwen_prior_weight_ep > 0.0
            ):
                prior_conf_b = qwen_prior_conf_t.index_select(0, batch_idx)
                prior_mask_b = make_qwen_prior_batch_mask(batch_is_lab, prior_conf_b, args)
                prior_type = str(getattr(args, "qwen_prior_loss_type", "none")).lower().strip()

                if prior_type in {"none", "off", "false", "0"}:
                    pass
                elif prior_type in {"activity", "activity_mse", "module_activity", "module_mse"}:
                    if qwen_module_activity_t is None:
                        raise RuntimeError("[QWEN-ACTIVITY] module_activity is missing from prior pack")
                    activity_b = qwen_module_activity_t.index_select(0, batch_idx)
                    qwen_prior_loss_v = qwen_prior_loss_v + qwen_module_activity_loss(
                        out1["z"], qwen_module_head, activity_b, prior_conf_b, prior_mask_b, args
                    ).to(mae_loss.dtype)
                else:
                    if qwen_prior_prob_t is None:
                        raise RuntimeError("[QWEN-PRIOR] prior_prob is missing from prior pack")
                    prior_prob_b = qwen_prior_prob_t.index_select(0, batch_idx)
                    if prior_type in {"kl", "both"}:
                        if cls_head is None:
                            raise RuntimeError("[QWEN-PRIOR] kl loss requires cls_head")
                        qwen_prior_loss_v = qwen_prior_loss_v + qwen_prior_kl_loss_from_logits(
                            out1["logits"], prior_prob_b, prior_conf_b, prior_mask_b, args
                        ).to(mae_loss.dtype)
                    if prior_type in {"proto", "both"}:
                        qwen_prior_loss_v = qwen_prior_loss_v + qwen_prior_kl_loss_from_prototypes(
                            out1["z"], proto_layer, prior_prob_b, prior_conf_b, prior_mask_b, args
                        ).to(mae_loss.dtype)
                    if prior_type not in {"kl", "proto", "both"}:
                        raise ValueError(f"[QWEN-PRIOR] unknown qwen_prior_loss_type={prior_type}")

            loss = (
                float(args.w_mae) * mae_loss
                + float(args.w_ce) * ce_loss
                + float(args.w_supcon) * supcon_loss
                + float(args.w_proto) * proto_loss_v
                + float(args.w_cons) * cons_loss_v
                + float(args.w_pseudo) * pseudo_loss_v
                + float(qwen_prior_weight_ep) * qwen_prior_loss_v
            )

            if not torch.isfinite(loss):
                raise RuntimeError("[NAN] loss became NaN/Inf")

            loss_to_backward = loss / float(accum_steps)
            if use_fp16:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if (global_step % accum_steps) == 0:
                if float(args.grad_clip) > 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        float(args.grad_clip),
                    )
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if args.print_debug_first_batch and (not did_dbg):
                with torch.no_grad():
                    x_all = out1["x_all"]
                    mask_labels = out1["mask_labels"]
                    gene_len = max(1, int(x_all.shape[1]) - 2)
                    valid_gene = (x_all[:, :gene_len] > float(args.mask_eps)) if str(args.mask_mode) == "pos" else torch.isfinite(x_all[:, :gene_len])
                    vcnt = int(valid_gene.sum().item())
                    mcnt = int(mask_labels[:, :gene_len].sum().item())
                    print(
                        f"[DBG] mask_mode={args.mask_mode} mask_eps={args.mask_eps} "
                        f"valid_cnt~={vcnt} mask_cnt={mcnt} enc_nonpad={enc_nonpad} "
                        f"N={x_all.shape[1]} mae_max_enc_len={args.mae_encoder_max_seq_len} "
                        f"encoder_visible_max_len={args.encoder_visible_max_len} topk_by={args.encoder_topk_by} "
                        f"semi={args.semi_supervised} labeled_ratio={args.labeled_ratio} "
                        f"w_mae={args.w_mae} w_ce={args.w_ce} w_supcon={args.w_supcon} "
                        f"w_proto={args.w_proto} w_cons={args.w_cons} w_pseudo={args.w_pseudo} "
                        f"use_qwen_prior={getattr(args, 'use_qwen_context_prior', False)} "
                        f"w_qwen_prior={getattr(args, 'qwen_prior_loss_weight', 0.0)}"
                    )
                did_dbg = True

            running += float(loss.item())
            running_mae += float(mae_loss.item())
            running_ce += float(ce_loss.item())
            running_supcon += float(supcon_loss.item())
            running_proto += float(proto_loss_v.item())
            running_cons += float(cons_loss_v.item())
            running_pseudo += float(pseudo_loss_v.item())
            running_qwen += float(qwen_prior_loss_v.item())
            steps += 1

            pbar.set_postfix(
                loss=running / max(1, steps),
                mae=running_mae / max(1, steps),
                ce=running_ce / max(1, steps),
                supcon=running_supcon / max(1, steps),
                proto=running_proto / max(1, steps),
                cons=running_cons / max(1, steps),
                pseudo=running_pseudo / max(1, steps),
                qwen=running_qwen / max(1, steps),
                sup_cnt=sup_cnt,
                enc_len=enc_nonpad,
                enc_len_max=enc_len_max_epoch,
            )

            if int(args.monitor_every_steps) > 0 and (global_step % int(args.monitor_every_steps) == 0):
                try:
                    embm = compute_embeddings_for_monitor(
                        pretrainmodel, pretrainconfig, adapter, proj_head,
                        x_np=x_np,
                        device=device,
                        max_samples=int(args.monitor_max_samples),
                        args=args,
                        qwen_prior_pack=qwen_prior_pack,
                        sample_indices=np.arange(int(x_np.shape[0]), dtype=np.int64),
                    )
                    k_monitor = int(args.monitor_kmeans_k)
                    if label2id is not None:
                        k_monitor = int(len(label2id))
                    stat = kmeans_monitor(embm, k=k_monitor, seed=int(getattr(args, "seed", 0)))
                    print(f"[MON] step={global_step} emb_shape={embm.shape} "
                          f"kmeans_inertia={stat['kmeans_inertia']:.4e} uniq={int(stat['uniq_clusters'])}")
                except Exception as e:
                    print("[MON][WARN] monitor failed:", repr(e))

        avg_mae_epoch = float(running_mae / max(1, steps))
        avg_cons_epoch = float(running_cons / max(1, steps))

        epoch_self_loss = (
            float(args.w_mae) * avg_mae_epoch
            + float(args.w_cons) * avg_cons_epoch
        )

        loss_eps = float(getattr(args, "exposed_cluster_loss_eps", 1e-8))
        if first_self_loss_for_ckpt is None:
            first_self_loss_for_ckpt = max(float(epoch_self_loss), loss_eps)

        norm_self_loss = float(epoch_self_loss) / max(float(first_self_loss_for_ckpt), loss_eps)

        if (
            bool(getattr(args, "use_exposed_cluster_checkpoint", False))
            and (exposed_ckpt_mask is not None)
            and (y_id is not None)
        ):
            ckpt_idx = np.where(np.asarray(exposed_ckpt_mask, dtype=bool))[0]
            if len(ckpt_idx) <= 0:
                raise RuntimeError("[EXPOSED-CKPT] exposed_ckpt_mask has 0 samples.")

            # 1. Exposed-only clustering metric
            emb_ckpt = compute_embeddings_for_monitor(
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                x_np=x_np[ckpt_idx],
                device=device,
                max_samples=len(ckpt_idx),
                args=args,
                qwen_prior_pack=qwen_prior_pack,
                sample_indices=ckpt_idx,
            )

            k_ckpt = int(len(label2id)) if label2id is not None else int(args.k_fixed)

            exposed_monitor_dir = os.path.join(
                str(args.local_model_dir),
                "exposed_cluster_checkpoint_monitor"
            )
            os.makedirs(exposed_monitor_dir, exist_ok=True)

            ckpt_stat = cluster_and_eval(
                emb=emb_ckpt,
                y_true=y_id[ckpt_idx],
                save_dir=exposed_monitor_dir,
                prefix=f"epoch_{ep+1:03d}_EXPOSED_CKPT",
                k_fixed=k_ckpt,
                cluster_prep=str(args.cluster_prep),
                pca_dim=int(args.pca_dim),
            )

            metric_name = str(getattr(args, "exposed_cluster_metric", "ARI")).upper().strip()
            if metric_name not in {"ACC", "NMI", "ARI", "PUR"}:
                raise ValueError(
                    f"[EXPOSED-CKPT] exposed_cluster_metric must be ACC/NMI/ARI/PUR, got {metric_name}"
                )

            cluster_score = float(ckpt_stat[metric_name])
            if cluster_score > max_exposed_cluster_score:
                max_exposed_cluster_score = float(cluster_score)

            candidate_eps = float(getattr(args, "exposed_cluster_candidate_eps", 0.005))
            is_candidate = bool(cluster_score >= max_exposed_cluster_score - candidate_eps)

            # 2. Label-free temporal clustering stability over all samples
            if stability_idx_for_ckpt is None:
                n_total = int(x_np.shape[0])
                max_stab = int(getattr(args, "exposed_cluster_stability_max_samples", 0))

                if max_stab > 0 and max_stab < n_total:
                    rng = np.random.RandomState(int(args.seed) + 202406)
                    stability_idx_for_ckpt = np.sort(
                        rng.choice(n_total, size=max_stab, replace=False)
                    ).astype(np.int64)
                else:
                    stability_idx_for_ckpt = np.arange(n_total, dtype=np.int64)

                print(
                    f"[STABILITY] use {len(stability_idx_for_ckpt)}/{x_np.shape[0]} "
                    f"samples for label-free temporal clustering stability"
                )

            emb_stability = compute_embeddings_for_monitor(
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                x_np=x_np[stability_idx_for_ckpt],
                device=device,
                max_samples=len(stability_idx_for_ckpt),
                args=args,
                qwen_prior_pack=qwen_prior_pack,
                sample_indices=stability_idx_for_ckpt,
            )

            all_cluster_pred = cluster_predict_for_checkpoint_stability(
                emb=emb_stability,
                k=k_ckpt,
                seed=int(args.seed),
                cluster_prep=str(args.cluster_prep),
                pca_dim=int(args.pca_dim),
            )

            label_free_quality = compute_label_free_cluster_quality_for_checkpoint(
                emb=emb_stability,
                pred=all_cluster_pred,
                k=k_ckpt,
                seed=int(args.seed),
                cluster_prep=str(args.cluster_prep),
                pca_dim=int(args.pca_dim),
                args=args,
            )

            if prev_all_cluster_pred_for_ckpt is None:
                temporal_stability = 0.0
            else:
                from sklearn.metrics import adjusted_rand_score
                temporal_stability = float(
                    adjusted_rand_score(prev_all_cluster_pred_for_ckpt, all_cluster_pred)
                )

            prev_all_cluster_pred_for_ckpt = all_cluster_pred.copy()

            # 3. Composite checkpoint score
            #    exposed metric + temporal stability - self-loss penalty
            stability_weight = float(getattr(args, "exposed_cluster_stability_weight", 0.25))
            loss_weight = float(getattr(args, "exposed_cluster_loss_weight", 0.03))
            loss_penalty = float(loss_weight * norm_self_loss)

            raw_composite_score = float(
                cluster_score
                + stability_weight * temporal_stability
                - loss_penalty
            )

            if is_candidate:
                val_score = raw_composite_score
            else:
                val_score = -float("inf")

            print(
                f"[EXPOSED-CKPT] epoch={ep+1:03d} "
                f"N={ckpt_stat['N']} K={ckpt_stat['K']} "
                f"ACC={ckpt_stat['ACC']:.6f} "
                f"NMI={ckpt_stat['NMI']:.6f} "
                f"ARI={ckpt_stat['ARI']:.6f} "
                f"PUR={ckpt_stat['PUR']:.6f} "
                f"cluster_{metric_name}={cluster_score:.6f} "
                f"max_cluster_{metric_name}={max_exposed_cluster_score:.6f} "
                f"candidate={is_candidate} "
                f"temporal_stability={temporal_stability:.6f} "
                f"self_loss={epoch_self_loss:.6f} "
                f"norm_self_loss={norm_self_loss:.6f} "
                f"loss_penalty={loss_penalty:.6f} "
                f"sil={label_free_quality.get('silhouette', float('nan')):.6f} "
                f"ch={label_free_quality.get('calinski_harabasz', float('nan')):.6f} "
                f"db={label_free_quality.get('davies_bouldin', float('nan')):.6f} "
                f"bal={label_free_quality.get('balance_entropy', float('nan')):.6f} "
                f"collapse={label_free_quality.get('collapse_penalty', float('nan')):.6f} "
                f"raw_score={raw_composite_score:.6f} "
                f"select_score={val_score:.6f}"
            )

            # 4. Save this epoch checkpoint and record for post-hoc reranking
            epoch_ckpt_state = {
                "epoch": int(ep + 1),
                "best_val_score": float(val_score),
                "selection_metric": metric_name,
                "selection_mode": "online_exposed_candidate_plus_label_free_temporal_stability_minus_self_loss",
                "cluster_score": float(cluster_score),
                "max_exposed_cluster_score": float(max_exposed_cluster_score),
                "candidate_eps": float(candidate_eps),
                "is_candidate": bool(is_candidate),
                "temporal_stability": float(temporal_stability),
                "stability_weight": float(stability_weight),
                "epoch_self_loss": float(epoch_self_loss),
                "norm_self_loss": float(norm_self_loss),
                "loss_weight": float(loss_weight),
                "loss_penalty": float(loss_penalty),
                "raw_composite_score": float(raw_composite_score),
                "label_free_quality": dict(label_free_quality),
                "silhouette": float(label_free_quality.get("silhouette", float("nan"))),
                "calinski_harabasz": float(label_free_quality.get("calinski_harabasz", float("nan"))),
                "davies_bouldin": float(label_free_quality.get("davies_bouldin", float("nan"))),
                "balance_entropy": float(label_free_quality.get("balance_entropy", 0.0)),
                "collapse_penalty_quality": float(label_free_quality.get("collapse_penalty", 0.0)),
                "exposed_cluster_stat": ckpt_stat,
                "pretrainmodel": _module_state_cpu(pretrainmodel),
                "adapter": _module_state_cpu(adapter),
                "pred_head": _module_state_cpu(pred_head),
                "cls_head": _module_state_cpu(cls_head),
                "proj_head": _module_state_cpu(proj_head),
                "proto_layer": _module_state_cpu(proto_layer),
                "qwen_module_head": _module_state_cpu(qwen_module_head),
            }

            epoch_ckpt_dir = os.path.join(
                str(args.local_model_dir),
                "epoch_ckpts_for_posthoc_selection"
            )
            os.makedirs(epoch_ckpt_dir, exist_ok=True)

            epoch_ckpt_path = os.path.join(
                epoch_ckpt_dir,
                f"epoch_{ep+1:03d}_checkpoint.pt"
            )
            torch.save(epoch_ckpt_state, epoch_ckpt_path)

            ckpt_selection_records.append(
                {
                    "epoch": int(ep + 1),
                    "ckpt_path": epoch_ckpt_path,
                    "metric_name": metric_name,
                    "cluster_score": float(cluster_score),
                    "max_exposed_cluster_score_at_epoch": float(max_exposed_cluster_score),
                    "candidate_eps": float(candidate_eps),
                    "is_candidate_online": bool(is_candidate),
                    "temporal_stability_prev": float(temporal_stability),
                    "epoch_self_loss": float(epoch_self_loss),
                    "norm_self_loss": float(norm_self_loss),
                    "loss_weight": float(loss_weight),
                    "stability_weight": float(stability_weight),
                    "raw_composite_score_online": float(raw_composite_score),
                    "silhouette": float(label_free_quality.get("silhouette", float("nan"))),
                    "calinski_harabasz": float(label_free_quality.get("calinski_harabasz", float("nan"))),
                    "davies_bouldin": float(label_free_quality.get("davies_bouldin", float("nan"))),
                    "balance_entropy": float(label_free_quality.get("balance_entropy", 0.0)),
                    "min_cluster_frac": float(label_free_quality.get("min_cluster_frac", 0.0)),
                    "max_cluster_frac": float(label_free_quality.get("max_cluster_frac", 1.0)),
                    "cluster_size_cv": float(label_free_quality.get("cluster_size_cv", float("inf"))),
                    "collapse_penalty_quality": float(label_free_quality.get("collapse_penalty", 0.0)),
                    "uniq_clusters": float(label_free_quality.get("uniq_clusters", 0.0)),
                    "all_cluster_pred": all_cluster_pred.copy(),
                    "exposed_cluster_stat": ckpt_stat,
                }
            )

            if is_candidate and val_score > best_val_score:
                best_val_score = float(val_score)
                best_val_epoch = int(ep + 1)

                best_ckpt_state = dict(epoch_ckpt_state)
                best_ckpt_state["best_val_score"] = float(best_val_score)
                best_ckpt_state["selection_mode"] = (
                    "online_exposed_candidate_plus_label_free_temporal_stability_minus_self_loss"
                )

                best_path = os.path.join(
                    str(args.local_model_dir),
                    "best_exposed_cluster_checkpoint_online.pt"
                )
                os.makedirs(str(args.local_model_dir), exist_ok=True)
                torch.save(best_ckpt_state, best_path)

                print(
                    f"[BEST-EXPOSED-CKPT][ONLINE] update best checkpoint: "
                    f"epoch={best_val_epoch} "
                    f"cluster_{metric_name}={cluster_score:.6f} "
                    f"temporal_stability={temporal_stability:.6f} "
                    f"norm_self_loss={norm_self_loss:.6f} "
                    f"select_score={best_val_score:.6f} "
                    f"path={best_path}"
                )





        if eval_mode in {"every_epoch", "both"}:
            save_local(
                pretrainmodel,
                pretrainconfig,
                adapter,
                pred_head,
                cls_head,
                proj_head,
                proto_layer,
                feature_list=[],
                local_dir=args.local_model_dir,
                extra_meta={
                    "epoch": int(ep + 1),
                    "enc_len_max_epoch": int(enc_len_max_epoch),
                    "global_step": int(global_step),
                    "sup_cnt_epoch": int(sup_cnt),
                },
                label2id=label2id,
                train_args=args,
            )

        if bool(args.run_infer_after_train) and eval_mode in {"every_epoch", "both"}:
            run_epoch_end_eval(
                epoch_idx=ep,
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                cls_head=cls_head,
                proto_layer=proto_layer,
                gexpr_aligned=gexpr_aligned,
                device=device,
                args=args,
                qwen_prior_pack=qwen_prior_pack,
            )

    # Rerank checkpoints using exposed scores, temporal stability, and self-loss.
    if (
        bool(getattr(args, "use_exposed_cluster_checkpoint", False))
        and bool(getattr(args, "exposed_cluster_posthoc_rerank", True))
        and len(ckpt_selection_records) > 0
    ):
        from sklearn.metrics import adjusted_rand_score

        global_max_cluster_score = max(
            float(r["cluster_score"]) for r in ckpt_selection_records
        )

        candidate_eps = float(getattr(args, "exposed_cluster_candidate_eps", 0.005))
        stability_window = int(getattr(args, "exposed_cluster_stability_window", 2))
        stability_weight = float(getattr(args, "exposed_cluster_stability_weight", 0.35))
        loss_weight = float(getattr(args, "exposed_cluster_loss_weight", 0.02))
        tie_eps = float(getattr(args, "exposed_cluster_tie_eps", 0.003))
        tie_break = str(getattr(args, "exposed_cluster_tie_break", "earlier")).lower().strip()

        posthoc_rows = []
        best_record = None

        for i, r in enumerate(ckpt_selection_records):
            cluster_score_i = float(r["cluster_score"])

            # Candidate is decided by global max exposed score over all epochs.
            is_candidate_global = bool(
                cluster_score_i >= global_max_cluster_score - candidate_eps
            )

            # Two-sided label-free temporal stability:
            # compare epoch i with neighbor epochs i±1, i±2 ...
            neighbor_scores = []
            for offset in range(1, max(1, stability_window) + 1):
                j_left = i - offset
                j_right = i + offset

                if j_left >= 0:
                    neighbor_scores.append(
                        float(adjusted_rand_score(
                            r["all_cluster_pred"],
                            ckpt_selection_records[j_left]["all_cluster_pred"]
                        ))
                    )

                if j_right < len(ckpt_selection_records):
                    neighbor_scores.append(
                        float(adjusted_rand_score(
                            r["all_cluster_pred"],
                            ckpt_selection_records[j_right]["all_cluster_pred"]
                        ))
                    )

            if len(neighbor_scores) > 0:
                smooth_temporal_stability = float(np.mean(neighbor_scores))
            else:
                smooth_temporal_stability = 0.0

            norm_self_loss_i = float(r["norm_self_loss"])
            loss_penalty_i = float(loss_weight * norm_self_loss_i)

            posthoc_score = float(
                cluster_score_i
                + stability_weight * smooth_temporal_stability
                - loss_penalty_i
            )

            if not is_candidate_global:
                select_score = -float("inf")
            else:
                select_score = posthoc_score

            row = {
                "epoch": int(r["epoch"]),
                "ckpt_path": str(r["ckpt_path"]),
                "metric_name": str(r["metric_name"]),
                "cluster_score": float(cluster_score_i),
                "global_max_cluster_score": float(global_max_cluster_score),
                "candidate_eps": float(candidate_eps),
                "is_candidate_global": bool(is_candidate_global),
                "smooth_temporal_stability": float(smooth_temporal_stability),
                "norm_self_loss": float(norm_self_loss_i),
                "loss_penalty": float(loss_penalty_i),
                "silhouette": float(r.get("silhouette", float("nan"))),
                "calinski_harabasz": float(r.get("calinski_harabasz", float("nan"))),
                "davies_bouldin": float(r.get("davies_bouldin", float("nan"))),
                "balance_entropy": float(r.get("balance_entropy", 0.0)),
                "min_cluster_frac": float(r.get("min_cluster_frac", 0.0)),
                "max_cluster_frac": float(r.get("max_cluster_frac", 1.0)),
                "cluster_size_cv": float(r.get("cluster_size_cv", float("inf"))),
                "collapse_penalty_quality": float(r.get("collapse_penalty_quality", 0.0)),
                "uniq_clusters": float(r.get("uniq_clusters", 0.0)),
                "posthoc_score_old": float(posthoc_score),
                "select_score_old": float(select_score),
                "posthoc_score": float(posthoc_score),
                "select_score": float(select_score),
            }
            posthoc_rows.append(row)

        # Select a checkpoint from the epoch-level records.
        # Compute structure-quality scores for exposed-score candidates.
        def _finite_or_nan(v):
            try:
                v = float(v)
                return v if np.isfinite(v) else float("nan")
            except Exception:
                return float("nan")

        def _robust01(values, higher_is_better=True):
            arr = np.asarray([_finite_or_nan(v) for v in values], dtype=np.float64)
            finite = np.isfinite(arr)
            out = np.full(arr.shape, 0.5, dtype=np.float64)
            if finite.sum() >= 2:
                lo = float(np.nanpercentile(arr[finite], 5))
                hi = float(np.nanpercentile(arr[finite], 95))
                if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                    lo = float(np.nanmin(arr[finite]))
                    hi = float(np.nanmax(arr[finite]))
                if np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) >= 1e-12:
                    out[finite] = (arr[finite] - lo) / (hi - lo)
                    out = np.clip(out, 0.0, 1.0)
            if not higher_is_better:
                out = 1.0 - out
            return out.astype(np.float64)

        sil01 = _robust01([r.get("silhouette", float("nan")) for r in posthoc_rows], True)
        ch01 = _robust01([r.get("calinski_harabasz", float("nan")) for r in posthoc_rows], True)
        db01 = _robust01([r.get("davies_bouldin", float("nan")) for r in posthoc_rows], False)
        bal01 = _robust01([r.get("balance_entropy", 0.0) for r in posthoc_rows], True)

        w_sil = float(getattr(args, "exposed_cluster_quality_silhouette_weight", 0.40))
        w_ch = float(getattr(args, "exposed_cluster_quality_ch_weight", 0.20))
        w_db = float(getattr(args, "exposed_cluster_quality_db_weight", 0.20))
        w_bal = float(getattr(args, "exposed_cluster_quality_balance_weight", 0.20))
        w_sum = max(w_sil + w_ch + w_db + w_bal, 1e-12)

        quality_weight = float(getattr(args, "exposed_cluster_quality_weight", 0.35))
        collapse_weight = float(getattr(args, "exposed_cluster_collapse_weight", 0.20))
        endpoint_weight = float(getattr(args, "exposed_cluster_endpoint_weight", 0.03))
        endpoint_start = float(getattr(args, "exposed_cluster_endpoint_start", 0.90))
        endpoint_start = min(max(endpoint_start, 0.0), 0.999)

        n_epochs_total = max(1, int(getattr(args, "epochs", len(posthoc_rows))))
        for ii, row in enumerate(posthoc_rows):
            unsup_quality_01 = float((w_sil * sil01[ii] + w_ch * ch01[ii] + w_db * db01[ii] + w_bal * bal01[ii]) / w_sum)
            collapse_penalty_i = float(row.get("collapse_penalty_quality", 0.0))
            ep_i = int(row["epoch"])
            progress = float(ep_i - 1) / float(max(1, n_epochs_total - 1))
            endpoint_penalty_i = 0.0
            if progress > endpoint_start:
                endpoint_penalty_i = float((progress - endpoint_start) / max(1e-8, 1.0 - endpoint_start))
            endpoint_penalty_i *= endpoint_weight

            old_posthoc = float(row["posthoc_score_old"])
            quality_posthoc = float(
                float(row["cluster_score"])
                + stability_weight * float(row["smooth_temporal_stability"])
                + quality_weight * unsup_quality_01
                - float(row["loss_penalty"])
                - collapse_weight * collapse_penalty_i
                - endpoint_penalty_i
            )
            row["silhouette_01"] = float(sil01[ii])
            row["calinski_harabasz_01"] = float(ch01[ii])
            row["davies_bouldin_01"] = float(db01[ii])
            row["balance_entropy_01"] = float(bal01[ii])
            row["unsup_quality_01"] = float(unsup_quality_01)
            row["quality_weight"] = float(quality_weight)
            row["collapse_weight"] = float(collapse_weight)
            row["endpoint_penalty"] = float(endpoint_penalty_i)
            row["posthoc_score_old"] = float(old_posthoc)
            row["posthoc_score"] = float(quality_posthoc)
            if bool(row["is_candidate_global"]):
                row["select_score"] = float(quality_posthoc)
            else:
                row["select_score"] = -float("inf")

        candidate_rows = [r for r in posthoc_rows if bool(r["is_candidate_global"])]

        if len(candidate_rows) <= 0:
            best_record = None
        elif tie_break in {"quality", "quality_score", "unsup_quality"}:
            best_record = None
            for row in candidate_rows:
                if best_record is None:
                    best_record = row
                else:
                    better = False
                    if float(row["select_score"]) > float(best_record["select_score"]) + tie_eps:
                        better = True
                    elif abs(float(row["select_score"]) - float(best_record["select_score"])) <= tie_eps:
                        # When quality scores are effectively tied, prefer the epoch
                        # with stronger label-free quality; then prefer earlier epoch.
                        if float(row.get("unsup_quality_01", 0.0)) > float(best_record.get("unsup_quality_01", 0.0)) + 1e-6:
                            better = True
                        elif abs(float(row.get("unsup_quality_01", 0.0)) - float(best_record.get("unsup_quality_01", 0.0))) <= 1e-6:
                            better = int(row["epoch"]) < int(best_record["epoch"])
                    if better:
                        best_record = row

            if best_record is not None:
                best_record["selection_rule"] = "quality_aware_label_free_structure"
                best_record["plateau_segment_start"] = -1
                best_record["plateau_segment_end"] = -1
                best_record["plateau_segment_len"] = -1
                best_record["plateau_center_quantile"] = float("nan")
                best_record["plateau_center_epoch_raw"] = -1

            print(
                f"[POSTHOC-FAIR-CKPT][QUALITY] "
                f"candidate_n={len(candidate_rows)} "
                f"selected_epoch={best_record['epoch'] if best_record is not None else -1} "
                f"unsup_quality={best_record.get('unsup_quality_01', float('nan')) if best_record is not None else float('nan'):.6f} "
                f"select_score={best_record.get('select_score', float('nan')) if best_record is not None else float('nan'):.6f}"
            )
        elif tie_break == "plateau_center":
            plateau_stability_eps = float(
                getattr(args, "exposed_cluster_plateau_stability_eps", 0.03)
            )
            plateau_min_len = int(
                getattr(args, "exposed_cluster_plateau_min_len", 3)
            )
            plateau_center_radius = int(
                getattr(args, "exposed_cluster_plateau_center_radius", 1)
            )

            max_smooth_stability = max(
                float(r["smooth_temporal_stability"]) for r in candidate_rows
            )

            stable_rows = [
                r for r in candidate_rows
                if float(r["smooth_temporal_stability"]) >= max_smooth_stability - plateau_stability_eps
            ]

            stable_rows = sorted(stable_rows, key=lambda x: int(x["epoch"]))

            # Split stable rows into contiguous epoch segments.
            segments = []
            cur_seg = []
            prev_epoch = None

            for r in stable_rows:
                ep_i = int(r["epoch"])
                if prev_epoch is None or ep_i == prev_epoch + 1:
                    cur_seg.append(r)
                else:
                    if len(cur_seg) > 0:
                        segments.append(cur_seg)
                    cur_seg = [r]
                prev_epoch = ep_i

            if len(cur_seg) > 0:
                segments.append(cur_seg)

            # Prefer segments with at least plateau_min_len.
            valid_segments = [seg for seg in segments if len(seg) >= plateau_min_len]
            if len(valid_segments) <= 0:
                valid_segments = segments

            if len(valid_segments) <= 0:
                # Fall back to score-based selection when no stable segment is found.
                best_record = max(candidate_rows, key=lambda r: float(r["select_score"]))
                best_record["selection_rule"] = "fallback_score"
                best_record["plateau_segment_start"] = -1
                best_record["plateau_segment_end"] = -1
                best_record["plateau_segment_len"] = -1
                best_record["plateau_center_quantile"] = float("nan")
                best_record["plateau_center_epoch_raw"] = -1
            else:
                # Choose the longest segment.
                # If tie, choose segment with higher mean posthoc_score.
                def _segment_key(seg):
                    return (
                        len(seg),
                        float(np.mean([float(x["posthoc_score"]) for x in seg])),
                        float(np.mean([float(x["smooth_temporal_stability"]) for x in seg])),
                    )

                best_segment = sorted(valid_segments, key=_segment_key, reverse=True)[0]

                # Select the configured quantile within the stable plateau.
                plateau_center_quantile = float(
                    getattr(args, "exposed_cluster_plateau_center_quantile", 0.65)
                )
                plateau_center_quantile = min(max(plateau_center_quantile, 0.0), 1.0)

                center_pos = int(round((len(best_segment) - 1) * plateau_center_quantile))
                center_pos = max(0, min(center_pos, len(best_segment) - 1))

                left = max(0, center_pos - plateau_center_radius)
                right = min(len(best_segment), center_pos + plateau_center_radius + 1)
                center_candidates = best_segment[left:right]

                # Maximize posthoc_score near the selected position; break ties by earlier epoch.
                best_record = None
                for r in center_candidates:
                    if best_record is None:
                        best_record = r
                    else:
                        better = False
                        if float(r["posthoc_score"]) > float(best_record["posthoc_score"]) + tie_eps:
                            better = True
                        elif abs(float(r["posthoc_score"]) - float(best_record["posthoc_score"])) <= tie_eps:
                            better = int(r["epoch"]) < int(best_record["epoch"])

                        if better:
                            best_record = r

                best_record["selection_rule"] = "plateau_center_quantile"
                best_record["plateau_max_smooth_stability"] = float(max_smooth_stability)
                best_record["plateau_stability_eps"] = float(plateau_stability_eps)
                best_record["plateau_min_len"] = int(plateau_min_len)
                best_record["plateau_center_radius"] = int(plateau_center_radius)
                best_record["plateau_center_quantile"] = float(plateau_center_quantile)
                best_record["plateau_segment_start"] = int(best_segment[0]["epoch"])
                best_record["plateau_segment_end"] = int(best_segment[-1]["epoch"])
                best_record["plateau_segment_len"] = int(len(best_segment))
                best_record["plateau_center_epoch_raw"] = int(best_segment[center_pos]["epoch"])

            print(
                f"[POSTHOC-FAIR-CKPT][PLATEAU] "
                f"candidate_n={len(candidate_rows)} "
                f"stable_n={len(stable_rows)} "
                f"selected_epoch={best_record['epoch'] if best_record is not None else -1} "
                f"plateau={best_record.get('plateau_segment_start', -1) if best_record is not None else -1}"
                f"-{best_record.get('plateau_segment_end', -1) if best_record is not None else -1} "
                f"center_raw={best_record.get('plateau_center_epoch_raw', -1) if best_record is not None else -1}"
            )

        else:
            # Score-based selection with earlier/lower_loss tie-breaking.
            best_record = None
            for row in candidate_rows:
                if best_record is None:
                    best_record = row
                else:
                    better = False
                    if row["select_score"] > best_record["select_score"] + tie_eps:
                        better = True
                    elif abs(row["select_score"] - best_record["select_score"]) <= tie_eps:
                        if tie_break == "lower_loss":
                            better = row["norm_self_loss"] < best_record["norm_self_loss"]
                        else:
                            better = row["epoch"] < best_record["epoch"]

                    if better:
                        best_record = row

        # Mark selected row in output table.
        for row in posthoc_rows:
            row["selected_by_posthoc"] = bool(
                best_record is not None and int(row["epoch"]) == int(best_record["epoch"])
            )
            row["posthoc_tie_break"] = str(tie_break)

        posthoc_dir = os.path.join(
            str(args.local_model_dir),
            "posthoc_fair_checkpoint_selection"
        )
        os.makedirs(posthoc_dir, exist_ok=True)

        pd.DataFrame(posthoc_rows).to_csv(
            os.path.join(posthoc_dir, "posthoc_fair_checkpoint_scores.csv"),
            index=False
        )

        with open(os.path.join(posthoc_dir, "posthoc_fair_checkpoint_scores.json"), "w", encoding="utf-8") as f:
            json.dump(posthoc_rows, f, ensure_ascii=False, indent=2)

        if best_record is not None:
            print(
                f"[POSTHOC-FAIR-CKPT] selected epoch={best_record['epoch']} "
                f"cluster_{best_record['metric_name']}={best_record['cluster_score']:.6f} "
                f"smooth_temporal_stability={best_record['smooth_temporal_stability']:.6f} "
                f"norm_self_loss={best_record['norm_self_loss']:.6f} "
                f"select_score={best_record['select_score']:.6f} "
                f"path={best_record['ckpt_path']}"
            )

            best_ckpt_state = torch.load(best_record["ckpt_path"], map_location="cpu")
            best_ckpt_state["best_val_score"] = float(best_record["select_score"])
            best_ckpt_state["selection_mode"] = (
                "posthoc_exposed_candidate_plus_plateau_center_label_free_temporal_stability_minus_self_loss"
            )
            best_ckpt_state["cluster_score"] = float(best_record["cluster_score"])
            best_ckpt_state["smooth_temporal_stability"] = float(best_record["smooth_temporal_stability"])
            best_ckpt_state["norm_self_loss"] = float(best_record["norm_self_loss"])
            best_ckpt_state["silhouette"] = float(best_record.get("silhouette", float("nan")))
            best_ckpt_state["calinski_harabasz"] = float(best_record.get("calinski_harabasz", float("nan")))
            best_ckpt_state["davies_bouldin"] = float(best_record.get("davies_bouldin", float("nan")))
            best_ckpt_state["balance_entropy"] = float(best_record.get("balance_entropy", 0.0))
            best_ckpt_state["unsup_quality_01"] = float(best_record.get("unsup_quality_01", float("nan")))
            best_ckpt_state["collapse_penalty_quality"] = float(best_record.get("collapse_penalty_quality", 0.0))
            best_ckpt_state["endpoint_penalty"] = float(best_record.get("endpoint_penalty", 0.0))
            best_ckpt_state["selection_rule"] = str(best_record.get("selection_rule", tie_break))
            best_ckpt_state["plateau_segment_start"] = int(best_record.get("plateau_segment_start", -1))
            best_ckpt_state["plateau_segment_end"] = int(best_record.get("plateau_segment_end", -1))
            best_ckpt_state["plateau_segment_len"] = int(best_record.get("plateau_segment_len", -1))
            best_ckpt_state["plateau_center_quantile"] = float(best_record.get("plateau_center_quantile", float("nan")))
            best_ckpt_state["plateau_center_epoch_raw"] = int(best_record.get("plateau_center_epoch_raw", -1))
            best_ckpt_state["posthoc_record"] = best_record

            final_best_path = os.path.join(
                str(args.local_model_dir),
                "best_exposed_cluster_checkpoint.pt"
            )
            torch.save(best_ckpt_state, final_best_path)

            print(f"[POSTHOC-FAIR-CKPT] saved final selected checkpoint to {final_best_path}")
        else:
            print("[POSTHOC-FAIR-CKPT][WARN] no global candidate checkpoint found; keep online-selected checkpoint.")

    print(f"[STAT] enc_len_max_global={enc_len_max_global} (max enc_len over all steps)")

    pretrainmodel.eval()
    adapter.eval()
    pred_head.eval()
    proj_head.eval()
    if cls_head is not None:
        cls_head.eval()
    if proto_layer is not None:
        proto_layer.eval()

    if best_ckpt_state is not None:
        print(
            f"[BEST-EXPOSED-CKPT] final selected checkpoint: "
            f"epoch={best_ckpt_state['epoch']} "
            f"mode={best_ckpt_state.get('selection_mode', 'NA')} "
            f"rule={best_ckpt_state.get('selection_rule', 'NA')} "
            f"cluster_{best_ckpt_state.get('selection_metric', 'METRIC')}="
            f"{best_ckpt_state.get('cluster_score', float('nan')):.6f} "
            f"smooth_temporal_stability={best_ckpt_state.get('smooth_temporal_stability', float('nan')):.6f} "
            f"norm_self_loss={best_ckpt_state.get('norm_self_loss', float('nan')):.6f} "
            f"unsup_quality={best_ckpt_state.get('unsup_quality_01', float('nan')):.6f} "
            f"sil={best_ckpt_state.get('silhouette', float('nan')):.6f} "
            f"db={best_ckpt_state.get('davies_bouldin', float('nan')):.6f} "
            f"plateau={best_ckpt_state.get('plateau_segment_start', -1)}"
            f"-{best_ckpt_state.get('plateau_segment_end', -1)} "
            f"plateau_center_quantile={best_ckpt_state.get('plateau_center_quantile', float('nan')):.2f} "
            f"plateau_center_raw={best_ckpt_state.get('plateau_center_epoch_raw', -1)} "
            f"select_score={best_ckpt_state['best_val_score']:.6f}"
        )
    else:
        print("[BEST-EXPOSED-CKPT][WARN] no exposed-cluster checkpoint was created.")

    return best_ckpt_state
# Inference / Clustering
@torch.no_grad()
def infer_and_cluster_after_train(
    pretrainmodel: nn.Module,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    proj_head: nn.Module,
    cls_head: Optional[nn.Module],
    proto_layer: Optional[nn.Module],
    gexpr_aligned,
    device: torch.device,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
):
    save_dir = str(args.infer_save_path)
    os.makedirs(save_dir, exist_ok=True)
    pretrainmodel.eval()
    adapter.eval()
    proj_head.eval()
    if cls_head is not None:
        cls_head.eval()
    if proto_layer is not None:
        proto_layer.eval()

    x_np = gexpr_aligned.values.astype(np.float32)
    sample_ids = [str(i) for i in range(len(x_np))]
    pooled_list, proj_list, cls_pred_list, proto_pred_list = [], [], [], []
    bs = max(1, int(args.infer_batch_size))
    for st in tqdm(range(0, len(x_np), bs), desc='[INFER] projected embedding'):
        ed = min(st + bs, len(x_np))
        batch_x = torch.tensor(x_np[st:ed], device=device, dtype=torch.float32)
        out = encode_and_project_light(
            pretrainmodel=pretrainmodel,
            pretrainconfig=pretrainconfig,
            adapter=adapter,
            proj_head=proj_head,
            cls_head=cls_head,
            batch_x=batch_x,
            device=device,
            args=args,
            qwen_prior_pack=qwen_prior_pack,
            sample_indices=torch.arange(st, ed, device=device, dtype=torch.long),
        )
        pooled_list.append(out["pooled"].detach().float().cpu().numpy())
        proj_list.append(out["z"].detach().float().cpu().numpy())
        if cls_head is not None and out["logits"] is not None:
            cls_pred_list.append(torch.argmax(out["logits"], dim=-1).detach().cpu().numpy())
        if proto_layer is not None:
            protos = proto_layer()
            proto_logits = torch.matmul(out["z"], protos.T)
            proto_pred_list.append(torch.argmax(proto_logits, dim=-1).detach().cpu().numpy())

    pooled_emb = np.concatenate(pooled_list, axis=0)
    proj_emb = np.concatenate(proj_list, axis=0)
    np.save(os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_pooled_embedding.npy"), pooled_emb)
    np.save(os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_projected_embedding.npy"), proj_emb)

    import pandas as pd
    pd.DataFrame(pooled_emb, index=sample_ids).to_csv(os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_pooled_embedding.csv"))
    pd.DataFrame(proj_emb, index=sample_ids).to_csv(os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_projected_embedding.csv"))
    pd.DataFrame({"sample_id": sample_ids}).to_csv(os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_sample_ids.csv"), index=False)
    np.save(os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_sample_ids.npy"), np.array(sample_ids, dtype=object))

    if len(cls_pred_list) > 0:
        cls_pred_all = np.concatenate(cls_pred_list, axis=0).reshape(-1)
        pd.DataFrame({"sample_id": sample_ids, "cls_pred": cls_pred_all}).to_csv(
            os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_cls_pred.csv"), index=False
        )
    if len(proto_pred_list) > 0:
        proto_pred_all = np.concatenate(proto_pred_list, axis=0).reshape(-1)
        pd.DataFrame({"sample_id": sample_ids, "proto_pred": proto_pred_all}).to_csv(
            os.path.join(save_dir, f"{args.infer_task_name}_{args.infer_ckpt_name}_proto_pred.csv"), index=False
        )

    y_true = load_labels_1col_numeric(args.label_path)
    y_true, mapping = remap_labels_to_0k(y_true)
    if mapping is not None:
        print('[LABEL][INFO] remapped labels to 0..K-1. mapping head:', list(mapping.items())[:10])
    if len(y_true) != len(sample_ids):
        raise RuntimeError(f"[LABEL] y_true len {len(y_true)} != sample len {len(sample_ids)}")

    keep_mask = np.ones(len(sample_ids), dtype=bool)
    keep_mask_unexposed = np.ones(len(sample_ids), dtype=bool)
    if bool(args.use_exposed_filter):
        exposed_ids = load_exposed_sample_ids(str(args.exposed_save_dir))
        if exposed_ids is not None:
            print(f"[EVAL][DEBUG] sample_ids[:10] = {sample_ids[:10]}")
            print(f"[EVAL][DEBUG] exposed_ids[:10] = {list(exposed_ids)[:10]}")
            keep_mask_unexposed = build_keep_mask_from_exposed(sample_ids, exposed_ids)
            keep_mask = keep_mask_unexposed
            save_eval_split(save_dir, sample_ids, keep_mask_unexposed)
            print(f"[EVAL] filter exposed samples: keep {int(keep_mask_unexposed.sum())}/{len(keep_mask_unexposed)}")
            if keep_mask_unexposed.shape[0] != proj_emb.shape[0]:
                raise RuntimeError(
                    f"[EVAL] keep_mask len {keep_mask_unexposed.shape[0]} != proj_emb len {proj_emb.shape[0]}"
                )
        else:
            print('[EVAL][WARN] exposed ids not found, fallback to all samples')

    if bool(args.use_exposed_filter):
        if int(keep_mask_unexposed.sum()) <= 0:
            raise RuntimeError(
                "[EVAL] no unexposed samples left after filtering. "
                "Check exposed_ids / sample_ids ID system consistency."
            )
        emb_eval_unexposed = proj_emb[keep_mask_unexposed]
        y_eval_unexposed = y_true[keep_mask_unexposed]
        prefix_unexposed = f"{args.infer_task_name}_{args.infer_ckpt_name}_UNEXPOSED"
        cluster_and_eval(
            emb_eval_unexposed,
            y_eval_unexposed,
            save_dir=save_dir,
            prefix=prefix_unexposed,
            k_fixed=int(args.k_fixed),
            cluster_prep=str(args.cluster_prep),
            pca_dim=int(args.pca_dim),
        )

    emb_eval_all = proj_emb
    y_eval_all = y_true
    prefix_all = f"{args.infer_task_name}_{args.infer_ckpt_name}_ALL"
    cluster_and_eval(
        emb_eval_all,
        y_eval_all,
        save_dir=save_dir,
        prefix=prefix_all,
        k_fixed=int(args.k_fixed),
        cluster_prep=str(args.cluster_prep),
        pca_dim=int(args.pca_dim),
    )


# Main
def main():
    set_seed(
    int(args.seed),
    deterministic=bool(getattr(args, "deterministic_algorithms", True))
    )

    install_scfoundation_patches(
        load_mod,
        encoder_visible_max_len=int(args.encoder_visible_max_len),
        encoder_topk_by=str(args.encoder_topk_by),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        if int(device.index) >= int(torch.cuda.device_count()):
            raise RuntimeError(
                f"[DEVICE] invalid device={device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"makes torch.cuda.device_count()={torch.cuda.device_count()}. "
                f"If using CUDA_VISIBLE_DEVICES=3 or auto_gpu, pass --device cuda:0."
            )
        torch.cuda.set_device(device.index)
        free, total = torch.cuda.mem_get_info(device.index)
        print(
            f"[DEVICE] using {device} | visible_count={torch.cuda.device_count()} "
            f"free_MB={free // 1024 // 1024} total_MB={total // 1024 // 1024} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )

    gene_list = load_gene_list(args.gene_index_path)
    if len(gene_list) != int(args.master_fixed_dim):
        raise RuntimeError(f"[GENE] gene_list len {len(gene_list)} != master_fixed_dim {args.master_fixed_dim}")

    raw_df = read_any_to_df_raw(args.data_path)
    gexpr = ensure_samples_by_rows_no_label(
        raw_df,
        gene_list=gene_list,
        orient_overlap_min=int(args.orient_overlap_min),
        orient_ratio=float(args.orient_ratio),
    )

    gexpr = clean_numeric_df(
        gexpr,
        input_clip=float(args.input_clip),
        dedup_policy=str(args.dedup_policy),
    )
    print(f"[DATA] shape(samples,features)={gexpr.shape}")

    overlap, _, _ = gene_align_diagnostics(gexpr.columns, gene_list)

    print("[GENE] align to gene_list(19264): pad missing with 0 + reorder by gene_list")
    t0 = time.time()
    gexpr_aligned, missing = align_to_gene_list(gexpr, gene_list)
    gexpr_aligned = clean_numeric_df(
        gexpr_aligned,
        input_clip=float(args.input_clip),
        dedup_policy=str(args.dedup_policy),
    )
    print(f"[GENE] after align: shape={gexpr_aligned.shape} | overlap={overlap} | padded_missing={len(missing)}")
    print("[TIME] gene_align seconds:", time.time() - t0)

    y_id = None
    labeled_mask = None             # Mask of exposed labeled samples.
    train_labeled_mask = None       # Exposed samples used for supervised training.
    exposed_ckpt_mask = None        # Exposed samples used for checkpoint selection.
    label2id = None

    if bool(args.semi_supervised):
        y_id, label2id = load_labels(args.label_path, sample_index=gexpr_aligned.index)

        # Load or create the exposed-sample mask.
        labeled_mask = load_or_create_fixed_labeled_mask(
            y_id=y_id,
            sample_index=gexpr_aligned.index,
            save_dir=args.exposed_save_dir,
            ratio=float(args.labeled_ratio),
            seed=int(args.seed),
            stratified=bool(args.stratified_sample),
            label_path=str(args.label_path),
        )

        print(
            f"[SEMI] exposed labeled_ratio={args.labeled_ratio} "
            f"exposed_cnt={int(labeled_mask.sum())}/{len(labeled_mask)} "
            f"stratified={args.stratified_sample} num_classes={len(label2id)}"
        )
        if int(getattr(args, "k_fixed", 0)) <= 0:
            args.k_fixed = int(len(label2id))
            print(f"[SEMI] k_fixed inferred from labels: K={args.k_fixed}")
        if int(getattr(args, "monitor_kmeans_k", 0)) <= 0:
            args.monitor_kmeans_k = int(len(label2id))
            print(f"[SEMI] monitor_kmeans_k inferred from labels: K={args.monitor_kmeans_k}")


        # Use the exposed subset for both supervised training and checkpoint selection.
        train_labeled_mask = np.asarray(labeled_mask, dtype=bool).copy()

        if bool(getattr(args, "use_exposed_cluster_checkpoint", False)):
            exposed_ckpt_mask = np.asarray(labeled_mask, dtype=bool).copy()
        else:
            exposed_ckpt_mask = None

        print(
            f"[SEMI] exposed labels are NOT split. "
            f"train_labeled={int(train_labeled_mask.sum())} "
            f"exposed_ckpt={int(exposed_ckpt_mask.sum()) if exposed_ckpt_mask is not None else 0} "
            f"exposed_total={int(labeled_mask.sum())} "
            f"total={len(labeled_mask)}"
        )

    pretrainmodel, pretrainconfig = load_model_frommmf(args.ckpt_path, args.key)
    pretrainmodel = pretrainmodel.to(device)

    pretrainconfig["valid_eps"] = float(args.valid_eps)
    pretrainconfig["mask_prob"] = float(args.mask_prob)
    pretrainconfig["mae_encoder_max_seq_len"] = int(args.mae_encoder_max_seq_len)
    pretrainconfig["mask_mode"] = str(args.mask_mode)
    pretrainconfig["mask_eps"] = float(args.mask_eps)

    pretrainconfig["encoder_visible_max_len"] = int(args.encoder_visible_max_len)
    pretrainconfig["encoder_topk_by"] = str(args.encoder_topk_by)
    pretrainconfig["qwen_visible_inject"] = bool(getattr(args, "qwen_visible_inject", False))
    pretrainconfig["qwen_visible_quota"] = int(getattr(args, "qwen_visible_quota", 0))
    pretrainconfig["qwen_visible_quota_mode"] = str(getattr(args, "qwen_visible_quota_mode", "fixed"))
    pretrainconfig["qwen_visible_quota_fraction"] = float(getattr(args, "qwen_visible_quota_fraction", 0.12))
    pretrainconfig["qwen_visible_quota_min"] = int(getattr(args, "qwen_visible_quota_min", 80))
    pretrainconfig["qwen_visible_quota_max"] = int(getattr(args, "qwen_visible_quota_max", 300))
    pretrainconfig["qwen_visible_mode"] = str(getattr(args, "qwen_visible_mode", "sample_module"))
    pretrainconfig["master_fixed_dim"] = int(args.master_fixed_dim)
    pretrainconfig["adapter_hidden_dim"] = int(args.adapter_hidden_dim)
    pretrainconfig["adapter_alpha"] = float(args.adapter_alpha)
    pretrainconfig["adapter_dropout"] = float(args.adapter_dropout)
    pretrainconfig["semi_supervised"] = bool(args.semi_supervised)
    pretrainconfig["labeled_ratio"] = float(args.labeled_ratio)
    pretrainconfig["unfreeze_mode"] = str(args.unfreeze_mode)
    pretrainconfig["unfreeze_last_n"] = int(args.unfreeze_last_n)

    pretrainconfig["proj_dim"] = int(args.proj_dim)
    pretrainconfig["proj_hidden_dim"] = int(args.proj_hidden_dim)
    pretrainconfig["cls_hidden_dim"] = int(args.cls_hidden_dim)
    pretrainconfig["supcon_temperature"] = float(args.supcon_temperature)
    pretrainconfig["proto_temperature"] = float(args.proto_temperature)
    pretrainconfig["aug_noise_std"] = float(args.aug_noise_std)
    pretrainconfig["aug_drop_prob"] = float(args.aug_drop_prob)
    pretrainconfig["w_mae"] = float(args.w_mae)
    pretrainconfig["w_ce"] = float(args.w_ce)
    pretrainconfig["w_supcon"] = float(args.w_supcon)
    pretrainconfig["w_proto"] = float(args.w_proto)
    pretrainconfig["w_cons"] = float(args.w_cons)
    pretrainconfig["w_pseudo"] = float(args.w_pseudo)

    print(
        f"[CFG] mask_mode={pretrainconfig['mask_mode']} mask_eps={pretrainconfig['mask_eps']:.2e} "
        f"mask_prob={pretrainconfig['mask_prob']} mae_max_enc_len={pretrainconfig['mae_encoder_max_seq_len']} "
        f"encoder_visible_max_len={args.encoder_visible_max_len} topk_by={args.encoder_topk_by} "
        f"qwen_visible_inject={getattr(args, 'qwen_visible_inject', False)} "
        f"qwen_visible_quota={getattr(args, 'qwen_visible_quota', 0)} "
        f"qwen_visible_quota_mode={getattr(args, 'qwen_visible_quota_mode', 'fixed')} "
        f"qwen_visible_mode={getattr(args, 'qwen_visible_mode', 'sample_module')} "
        f"semi={args.semi_supervised} labeled_ratio={args.labeled_ratio} "
        f"unfreeze_mode={args.unfreeze_mode} unfreeze_last_n={args.unfreeze_last_n}"
    )

    adapter = ResidualFeatureAdapter(
        dim=int(args.master_fixed_dim),
        hidden_dim=int(args.adapter_hidden_dim),
        alpha=float(args.adapter_alpha),
        dropout=float(args.adapter_dropout),
    ).to(device)

    pred_head = nn.Identity().to(device)

    try:
        if hasattr(pretrainmodel, "token_emb") and hasattr(pretrainmodel.token_emb, "dim"):
            d_model = int(pretrainmodel.token_emb.dim)
        else:
            tmp = torch.zeros((1, 4), device=device, dtype=torch.float32)
            tmp_tok = pretrainmodel.token_emb(torch.unsqueeze(tmp, 2).float(), output_weight=0)
            d_model = int(tmp_tok.shape[-1])
    except Exception:
        raise RuntimeError("[MAIN] cannot infer encoder hidden dim.")

    emb_dim = 4 * d_model

    proj_head = ProjectionHead(
        in_dim=emb_dim,
        proj_dim=int(args.proj_dim),
        hidden_dim=int(args.proj_hidden_dim),
        dropout=float(args.proj_dropout),
    ).to(device)

    cls_head = None
    proto_layer = None
    if bool(args.semi_supervised):
        if label2id is None:
            raise RuntimeError("[SEMI] label2id is None but semi_supervised=True")

        cls_head = ClassifierHead(
            in_dim=emb_dim,
            num_classes=int(len(label2id)),
            hidden_dim=int(args.cls_hidden_dim),
            dropout=float(args.cls_dropout),
        ).to(device)

        proto_layer = PrototypeLayer(
            num_classes=int(len(label2id)),
            dim=int(args.proj_dim),
        ).to(device)

        print(f"[SEMI] cls_head built: emb_dim={emb_dim} num_classes={len(label2id)}")
        print(f"[SEMI] proj_head built: proj_dim={args.proj_dim}")
        print(f"[SEMI] proto_layer built: num_classes={len(label2id)} dim={args.proj_dim}")

    _ = try_resume(
        pretrainmodel=pretrainmodel,
        adapter=adapter,
        pred_head=pred_head,
        cls_head=cls_head,
        proj_head=proj_head,
        proto_layer=proto_layer,
        local_dir=args.local_model_dir,
        resume_if_present=bool(args.resume_if_present),
    )

    x_np = gexpr_aligned.values.astype(np.float32)
    print("[DATA] shape(samples,19264):", x_np.shape)

    qwen_prior_pack = None
    if bool(getattr(args, "use_qwen_context_prior", False)):
        if not bool(args.semi_supervised):
            raise RuntimeError("[QWEN-PRIOR] use_qwen_context_prior=True requires semi_supervised=True")
        if label2id is None:
            raise RuntimeError("[QWEN-PRIOR] label2id is None; cannot align prior probabilities to classes")
        qwen_prior_pack = load_qwen_context_prior_npz(
            npz_path=str(getattr(args, "qwen_prior_npz", "")),
            sample_index=gexpr_aligned.index,
            num_classes=int(len(label2id)),
        )
        pretrainconfig["use_qwen_context_prior"] = True
        pretrainconfig["qwen_prior_npz"] = str(getattr(args, "qwen_prior_npz", ""))
        pretrainconfig["qwen_prior_loss_type"] = str(getattr(args, "qwen_prior_loss_type", "kl"))
        pretrainconfig["qwen_prior_loss_weight"] = float(getattr(args, "qwen_prior_loss_weight", 0.0))
        pretrainconfig["qwen_prior_apply_to"] = str(getattr(args, "qwen_prior_apply_to", "unlabeled"))
        pretrainconfig["qwen_visible_inject"] = bool(getattr(args, "qwen_visible_inject", False))
        pretrainconfig["qwen_visible_quota"] = int(getattr(args, "qwen_visible_quota", 0))
        pretrainconfig["qwen_visible_quota_mode"] = str(getattr(args, "qwen_visible_quota_mode", "fixed"))
        pretrainconfig["qwen_visible_quota_fraction"] = float(getattr(args, "qwen_visible_quota_fraction", 0.12))
        pretrainconfig["qwen_visible_quota_min"] = int(getattr(args, "qwen_visible_quota_min", 80))
        pretrainconfig["qwen_visible_quota_max"] = int(getattr(args, "qwen_visible_quota_max", 300))
        pretrainconfig["qwen_visible_mode"] = str(getattr(args, "qwen_visible_mode", "sample_module"))

    eval_mode = normalize_infer_cluster_eval_mode(getattr(args, "infer_cluster_eval_mode", "final"))

    if bool(args.semi_supervised) and (labeled_mask is not None):
        save_exposed_ids(
            save_dir=args.exposed_save_dir,
            sample_index=gexpr_aligned.index,
            labeled_mask=labeled_mask,
            seed=int(args.seed),
            labeled_ratio=float(args.labeled_ratio),
            stratified=bool(args.stratified_sample),
            label_path=str(args.label_path),
        )

    best_ckpt_state = train_one(
        pretrainmodel=pretrainmodel,
        pretrainconfig=pretrainconfig,
        adapter=adapter,
        pred_head=pred_head,
        cls_head=cls_head,
        proj_head=proj_head,
        proto_layer=proto_layer,
        x_np=x_np,
        y_id=y_id,
        labeled_mask=train_labeled_mask,
        exposed_ckpt_mask=exposed_ckpt_mask,
        label2id=label2id,
        gexpr_aligned=gexpr_aligned,
        device=device,
        args=args,
        qwen_prior_pack=qwen_prior_pack,
    )

    if bool(args.semi_supervised) and (labeled_mask is not None):
        save_exposed_ids(
            save_dir=args.exposed_save_dir,
            sample_index=gexpr_aligned.index,
            labeled_mask=labeled_mask,
            seed=int(args.seed),
            labeled_ratio=float(args.labeled_ratio),
            stratified=bool(args.stratified_sample),
            label_path=str(args.label_path),
        )

    if (
        bool(getattr(args, "use_exposed_cluster_checkpoint", False))
        and best_ckpt_state is not None
    ):
        print(
            f"[BEST-EXPOSED-CKPT] loading exposed-cluster-selected checkpoint "
            f"before final save/inference: epoch={best_ckpt_state['epoch']}"
        )

        _load_state_to_modules(
            ckpt_state=best_ckpt_state,
            pretrainmodel=pretrainmodel,
            adapter=adapter,
            pred_head=pred_head,
            cls_head=cls_head,
            proj_head=proj_head,
            proto_layer=proto_layer,
            device=device,
        )

        args.infer_ckpt_name = (
            f"{args.infer_ckpt_name}_bestexposed_epoch{int(best_ckpt_state['epoch']):03d}"
        )

    save_local(
        pretrainmodel,
        pretrainconfig,
        adapter,
        pred_head,
        cls_head,
        proj_head,
        proto_layer,
        gene_list,
        args.local_model_dir,
        extra_meta={
            "data_path": str(args.data_path),
            "gene_index_path": str(args.gene_index_path),
            "gene_overlap": int(overlap),
            "padded_missing": int(len(missing)),
            "use_qwen_context_prior": bool(getattr(args, "use_qwen_context_prior", False)),
            "qwen_prior_npz": str(getattr(args, "qwen_prior_npz", "")),
            "qwen_prior_loss_type": str(getattr(args, "qwen_prior_loss_type", "kl")),
            "qwen_prior_loss_weight": float(getattr(args, "qwen_prior_loss_weight", 0.0)),
            "qwen_prior_apply_to": str(getattr(args, "qwen_prior_apply_to", "unlabeled")),
            "qwen_visible_inject": bool(getattr(args, "qwen_visible_inject", False)),
            "qwen_visible_quota": int(getattr(args, "qwen_visible_quota", 0)),
            "qwen_visible_quota_mode": str(getattr(args, "qwen_visible_quota_mode", "fixed")),
            "qwen_visible_quota_fraction": float(getattr(args, "qwen_visible_quota_fraction", 0.12)),
            "qwen_visible_quota_min": int(getattr(args, "qwen_visible_quota_min", 80)),
            "qwen_visible_quota_max": int(getattr(args, "qwen_visible_quota_max", 300)),
            "qwen_visible_mode": str(getattr(args, "qwen_visible_mode", "sample_module")),
            "qwen_visible_module_topk": int(getattr(args, "qwen_visible_module_topk", 0)),
            "qwen_visible_require_nonzero": bool(getattr(args, "qwen_visible_require_nonzero", True)),
            "final_labeled_cnt": int(labeled_mask.sum()) if labeled_mask is not None else 0,
            "exposed_save_dir": str(args.exposed_save_dir),
            "checkpoint_selection": "posthoc_exposed_candidate_plus_plateau_center_label_free_temporal_stability_minus_self_loss",
            "best_ckpt_epoch": int(best_ckpt_state["epoch"]) if best_ckpt_state is not None else -1,
            "best_ckpt_score": float(best_ckpt_state["best_val_score"]) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_cluster_score": float(best_ckpt_state.get("cluster_score", float("nan"))) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_temporal_stability": float(best_ckpt_state.get("temporal_stability", float("nan"))) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_smooth_temporal_stability": float(best_ckpt_state.get("smooth_temporal_stability", float("nan"))) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_norm_self_loss": float(best_ckpt_state.get("norm_self_loss", float("nan"))) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_candidate_eps": float(best_ckpt_state.get("candidate_eps", float("nan"))) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_selection_rule": str(best_ckpt_state.get("selection_rule", "NA")) if best_ckpt_state is not None else "NA",
            "best_ckpt_plateau_start": int(best_ckpt_state.get("plateau_segment_start", -1)) if best_ckpt_state is not None else -1,
            "best_ckpt_plateau_end": int(best_ckpt_state.get("plateau_segment_end", -1)) if best_ckpt_state is not None else -1,
            "best_ckpt_plateau_len": int(best_ckpt_state.get("plateau_segment_len", -1)) if best_ckpt_state is not None else -1,
            "best_ckpt_plateau_center_quantile": float(best_ckpt_state.get("plateau_center_quantile", float("nan"))) if best_ckpt_state is not None else float("nan"),
            "best_ckpt_plateau_center_raw": int(best_ckpt_state.get("plateau_center_epoch_raw", -1)) if best_ckpt_state is not None else -1,
        },
        label2id=label2id,
        train_args=args,
    )

    if bool(args.run_infer_after_train) and eval_mode in {"final", "both"}:
        print("\n" + "=" * 80)
        print("[POST] training finished, start inference + clustering ...")
        print("=" * 80)
        infer_and_cluster_after_train(
            pretrainmodel=pretrainmodel,
            pretrainconfig=pretrainconfig,
            adapter=adapter,
            proj_head=proj_head,
            cls_head=cls_head,
            proto_layer=proto_layer,
            gexpr_aligned=gexpr_aligned,
            device=device,
            args=args,
            qwen_prior_pack=qwen_prior_pack,
        )


if __name__ == "__main__":
    main()