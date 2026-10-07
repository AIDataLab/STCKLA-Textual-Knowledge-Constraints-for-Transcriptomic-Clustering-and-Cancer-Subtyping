"""train.py definitions moved here without algorithm changes."""

import os
from typing import Optional, Dict, Any
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pandas as pd
from base import save_local, augment_expression, supervised_contrastive_loss, prototype_loss, consistency_loss, cluster_and_eval
from ..logging_utils import diagnostic_print
from .config import (
    _module_state_cpu,
    normalize_infer_cluster_eval_mode,
)
from .encoder import (
    encode_and_project_full,
    encode_and_project_light,
    set_trainable,
)
from .evaluation import (
    run_epoch_end_eval,
)
from .monitoring import (
    cluster_predict_for_checkpoint_stability,
    compute_embeddings_for_monitor,
    compute_label_free_cluster_quality_for_checkpoint,
    kmeans_monitor,
)
from .prior_guidance import (
    QwenModuleActivityHead,
    make_qwen_prior_batch_mask,
    qwen_module_activity_loss,
    qwen_prior_kl_loss_from_logits,
    qwen_prior_kl_loss_from_prototypes,
    qwen_prior_weight_for_epoch,
)
from .prior_loading import (
    ExprSemiDatasetWithIndex,
)


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
                    diagnostic_print(
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

            # print(
            #     f"[EXPOSED-CKPT] epoch={ep+1:03d} "
            #     f"N={ckpt_stat['N']} K={ckpt_stat['K']} "
            #     f"ACC={ckpt_stat['ACC']:.6f} "
            #     f"NMI={ckpt_stat['NMI']:.6f} "
            #     f"ARI={ckpt_stat['ARI']:.6f} "
            #     f"PUR={ckpt_stat['PUR']:.6f} "
            #     f"cluster_{metric_name}={cluster_score:.6f} "
            #     f"max_cluster_{metric_name}={max_exposed_cluster_score:.6f} "
            #     f"candidate={is_candidate} "
            #     f"temporal_stability={temporal_stability:.6f} "
            #     f"self_loss={epoch_self_loss:.6f} "
            #     f"norm_self_loss={norm_self_loss:.6f} "
            #     f"loss_penalty={loss_penalty:.6f} "
            #     f"sil={label_free_quality.get('silhouette', float('nan')):.6f} "
            #     f"ch={label_free_quality.get('calinski_harabasz', float('nan')):.6f} "
            #     f"db={label_free_quality.get('davies_bouldin', float('nan')):.6f} "
            #     f"bal={label_free_quality.get('balance_entropy', float('nan')):.6f} "
            #     f"collapse={label_free_quality.get('collapse_penalty', float('nan')):.6f} "
            #     f"raw_score={raw_composite_score:.6f} "
            #     f"select_score={val_score:.6f}"
            # )

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

                # print(
                #     f"[BEST-EXPOSED-CKPT][ONLINE] update best checkpoint: "
                #     f"epoch={best_val_epoch} "
                #     f"cluster_{metric_name}={cluster_score:.6f} "
                #     f"temporal_stability={temporal_stability:.6f} "
                #     f"norm_self_loss={norm_self_loss:.6f} "
                #     f"select_score={best_val_score:.6f} "
                #     f"path={best_path}"
                # )





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

            # print(
            #     f"[POSTHOC-FAIR-CKPT][QUALITY] "
            #     f"candidate_n={len(candidate_rows)} "
            #     f"selected_epoch={best_record['epoch'] if best_record is not None else -1} "
            #     f"unsup_quality={best_record.get('unsup_quality_01', float('nan')) if best_record is not None else float('nan'):.6f} "
            #     f"select_score={best_record.get('select_score', float('nan')) if best_record is not None else float('nan'):.6f}"
            # )
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

            # print(
            #     f"[POSTHOC-FAIR-CKPT][PLATEAU] "
            #     f"candidate_n={len(candidate_rows)} "
            #     f"stable_n={len(stable_rows)} "
            #     f"selected_epoch={best_record['epoch'] if best_record is not None else -1} "
            #     f"plateau={best_record.get('plateau_segment_start', -1) if best_record is not None else -1}"
            #     f"-{best_record.get('plateau_segment_end', -1) if best_record is not None else -1} "
            #     f"center_raw={best_record.get('plateau_center_epoch_raw', -1) if best_record is not None else -1}"
            # )

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
            # print(
            #     f"[POSTHOC-FAIR-CKPT] selected epoch={best_record['epoch']} "
            #     f"cluster_{best_record['metric_name']}={best_record['cluster_score']:.6f} "
            #     f"smooth_temporal_stability={best_record['smooth_temporal_stability']:.6f} "
            #     f"norm_self_loss={best_record['norm_self_loss']:.6f} "
            #     f"select_score={best_record['select_score']:.6f} "
            #     f"path={best_record['ckpt_path']}"
            # )

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

            #print(f"[POSTHOC-FAIR-CKPT] saved final selected checkpoint to {final_best_path}")
        else:
            print("[POSTHOC-FAIR-CKPT][WARN] no global candidate checkpoint found; keep online-selected checkpoint.")

    diagnostic_print(f"[STAT] enc_len_max_global={enc_len_max_global} (max enc_len over all steps)")

    pretrainmodel.eval()
    adapter.eval()
    pred_head.eval()
    proj_head.eval()
    if cls_head is not None:
        cls_head.eval()
    if proto_layer is not None:
        proto_layer.eval()

    # if best_ckpt_state is not None:
    #     print(
    #         f"[BEST-EXPOSED-CKPT] final selected checkpoint: "
    #         f"epoch={best_ckpt_state['epoch']} "
    #         f"mode={best_ckpt_state.get('selection_mode', 'NA')} "
    #         f"rule={best_ckpt_state.get('selection_rule', 'NA')} "
    #         f"cluster_{best_ckpt_state.get('selection_metric', 'METRIC')}="
    #         f"{best_ckpt_state.get('cluster_score', float('nan')):.6f} "
    #         f"smooth_temporal_stability={best_ckpt_state.get('smooth_temporal_stability', float('nan')):.6f} "
    #         f"norm_self_loss={best_ckpt_state.get('norm_self_loss', float('nan')):.6f} "
    #         f"unsup_quality={best_ckpt_state.get('unsup_quality_01', float('nan')):.6f} "
    #         f"sil={best_ckpt_state.get('silhouette', float('nan')):.6f} "
    #         f"db={best_ckpt_state.get('davies_bouldin', float('nan')):.6f} "
    #         f"plateau={best_ckpt_state.get('plateau_segment_start', -1)}"
    #         f"-{best_ckpt_state.get('plateau_segment_end', -1)} "
    #         f"plateau_center_quantile={best_ckpt_state.get('plateau_center_quantile', float('nan')):.2f} "
    #         f"plateau_center_raw={best_ckpt_state.get('plateau_center_epoch_raw', -1)} "
    #         f"select_score={best_ckpt_state['best_val_score']:.6f}"
    #     )
    # else:
    #     print("[BEST-EXPOSED-CKPT][WARN] no exposed-cluster checkpoint was created.")

    return best_ckpt_state
