"""train.py definitions moved here without algorithm changes."""

import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from base import load_exposed_sample_ids, build_keep_mask_from_exposed, save_eval_split, load_labels_1col_numeric, remap_labels_to_0k, cluster_and_eval
from ..logging_utils import diagnostic_print
from .config import (
    get_rng_state_all,
    set_rng_state_all,
)
from .encoder import (
    encode_and_project_light,
)


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

    diagnostic_print("\n" + "=" * 80)
    print(f"[EVAL] epoch {epoch_idx+1} finished, start inference + clustering ...")
    diagnostic_print("=" * 80)

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
            diagnostic_print(f"[EVAL][DEBUG] sample_ids[:10] = {sample_ids[:10]}")
            diagnostic_print(f"[EVAL][DEBUG] exposed_ids[:10] = {list(exposed_ids)[:10]}")
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
