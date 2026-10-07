"""train.py definitions moved here without algorithm changes."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from load import load_model_frommmf
import load as load_mod
from base import set_seed, install_scfoundation_patches, try_resume, save_local, save_exposed_ids, gene_align_diagnostics, load_gene_list, ensure_samples_by_rows_no_label, read_any_to_df_raw, clean_numeric_df, align_to_gene_list, ResidualFeatureAdapter, ProjectionHead, ClassifierHead, PrototypeLayer, load_labels, load_or_create_fixed_labeled_mask
from ..logging_utils import diagnostic_print
from .config import (
    _load_state_to_modules,
    args,
    normalize_infer_cluster_eval_mode,
)
from .evaluation import (
    infer_and_cluster_after_train,
)
from .prior_loading import (
    load_qwen_context_prior_npz,
)
from .training_loop import (
    train_one,
)


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

    diagnostic_print("[GENE] align to gene_list(19264): pad missing with 0 + reorder by gene_list")
    t0 = time.time()
    gexpr_aligned, missing = align_to_gene_list(gexpr, gene_list)
    gexpr_aligned = clean_numeric_df(
        gexpr_aligned,
        input_clip=float(args.input_clip),
        dedup_policy=str(args.dedup_policy),
    )
    print(f"[GENE] after align: shape={gexpr_aligned.shape} | overlap={overlap} | padded_missing={len(missing)}")
    diagnostic_print("[TIME] gene_align seconds:", time.time() - t0)

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

    diagnostic_print(
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

        diagnostic_print(f"[SEMI] cls_head built: emb_dim={emb_dim} num_classes={len(label2id)}")
        diagnostic_print(f"[SEMI] proj_head built: proj_dim={args.proj_dim}")
        diagnostic_print(f"[SEMI] proto_layer built: num_classes={len(label2id)} dim={args.proj_dim}")

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
    diagnostic_print("[DATA] shape(samples,19264):", x_np.shape)

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
        # print(
        #     f"[BEST-EXPOSED-CKPT] loading exposed-cluster-selected checkpoint "
        #     f"before final save/inference: epoch={best_ckpt_state['epoch']}"
        # )

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
        diagnostic_print("\n" + "=" * 80)
        print("[POST] training finished, start inference + clustering ...")
        diagnostic_print("=" * 80)
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
