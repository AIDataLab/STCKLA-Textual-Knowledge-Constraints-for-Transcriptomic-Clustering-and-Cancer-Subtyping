"""base.py definitions moved here without algorithm changes."""

import os
import json
import time
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from ..logging_utils import diagnostic_print


def _local_paths(local_dir: str):
    return (
        os.path.join(local_dir, "finetuned_state_dict.pt"),
        os.path.join(local_dir, "pretrainconfig.json"),
        os.path.join(local_dir, "meta.json"),
        os.path.join(local_dir, "feature_adapter.pt"),
        os.path.join(local_dir, "feature_list.json"),
        os.path.join(local_dir, "pred_head.pt"),
        os.path.join(local_dir, "cls_head.pt"),
        os.path.join(local_dir, "label_meta.json"),
        os.path.join(local_dir, "proj_head.pt"),
        os.path.join(local_dir, "proto_layer.pt"),
    )


def has_finetuned(local_dir: str) -> bool:
    sd_path, cfg_path, _, adapter_path, _, _, _, _, proj_path, proto_path = _local_paths(local_dir)
    base_ok = os.path.isfile(sd_path) and os.path.isfile(cfg_path) and os.path.isfile(adapter_path)
    extra_ok = os.path.isfile(proj_path) and os.path.isfile(proto_path)
    return base_ok and extra_ok


def try_resume(
    pretrainmodel: nn.Module,
    adapter: nn.Module,
    pred_head: nn.Module,
    cls_head: Optional[nn.Module],
    proj_head: nn.Module,
    proto_layer: Optional[nn.Module],
    local_dir: str,
    resume_if_present: bool = True,
) -> bool:
    if not (resume_if_present and has_finetuned(local_dir)):
        return False

    (
        sd_path, cfg_path, _, adapter_path, _, head_path, cls_path,
        label_meta_path, proj_path, proto_path
    ) = _local_paths(local_dir)

    print(f"[RESUME] found finetuned artifacts in {local_dir} -> loading to resume")

    sd = torch.load(sd_path, map_location="cpu")
    missing_keys, unexpected_keys = pretrainmodel.load_state_dict(sd, strict=False)
    print(f"[RESUME] pretrainmodel loaded. missing={len(missing_keys)} unexpected={len(unexpected_keys)}")

    adapter_sd = torch.load(adapter_path, map_location="cpu")
    adapter.load_state_dict(adapter_sd, strict=True)
    diagnostic_print("[RESUME] adapter loaded.")

    if os.path.isfile(head_path):
        head_sd = torch.load(head_path, map_location="cpu")
        pred_head.load_state_dict(head_sd, strict=False)
        diagnostic_print("[RESUME] pred_head loaded.")

    if os.path.isfile(proj_path):
        proj_sd = torch.load(proj_path, map_location="cpu")
        proj_head.load_state_dict(proj_sd, strict=True)
        diagnostic_print("[RESUME] proj_head loaded.")

    if (cls_head is not None) and os.path.isfile(cls_path):
        cls_sd = torch.load(cls_path, map_location="cpu")
        cls_head.load_state_dict(cls_sd, strict=True)
        diagnostic_print("[RESUME] cls_head loaded.")
    elif (cls_head is not None) and (not os.path.isfile(cls_path)):
        print("[RESUME][INFO] cls_head.pt not found. cls_head starts fresh.")

    if (proto_layer is not None) and os.path.isfile(proto_path):
        proto_sd = torch.load(proto_path, map_location="cpu")
        proto_layer.load_state_dict(proto_sd, strict=True)
        diagnostic_print("[RESUME] proto_layer loaded.")
    elif (proto_layer is not None) and (not os.path.isfile(proto_path)):
        print("[RESUME][INFO] proto_layer.pt not found. proto_layer starts fresh.")

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        diagnostic_print("[RESUME] pretrainconfig.json loaded (info). keys head:", list(_cfg.keys())[:10])
    except Exception:
        pass

    if os.path.isfile(label_meta_path):
        try:
            with open(label_meta_path, "r", encoding="utf-8") as f:
                _lm = json.load(f)
            diagnostic_print("[RESUME] label_meta.json loaded (info).")
            diagnostic_print("         num_classes:", _lm.get("num_classes"), "mapping_head:", list(_lm.get("label2id", {}).items())[:5])
        except Exception:
            pass

    return True


def save_feature_list(feature_list: List[str], local_dir: str):
    _, _, _, _, featlist_path, _, _, _, _, _ = _local_paths(local_dir)
    with open(featlist_path, "w", encoding="utf-8") as f:
        json.dump({"feature_list": feature_list}, f, ensure_ascii=False, indent=2)
    diagnostic_print("[LOCAL] Saved feature_list:", featlist_path)


def save_label_meta(label2id: Dict[str, int], local_dir: str):
    label_meta_path = _local_paths(local_dir)[7]
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_classes": int(len(label2id)),
        "label2id": {str(k): int(v) for k, v in label2id.items()},
    }
    with open(label_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    diagnostic_print("[LOCAL] Saved label_meta:", label_meta_path)


def save_local(
    pretrainmodel,
    pretrainconfig: Dict[str, Any],
    adapter,
    pred_head,
    cls_head: Optional[nn.Module],
    proj_head: nn.Module,
    proto_layer: Optional[nn.Module],
    feature_list: List[str],
    local_dir: str,
    *,
    extra_meta=None,
    label2id: Optional[Dict[str, int]] = None,
    train_args: Optional[Any] = None,
):
    os.makedirs(local_dir, exist_ok=True)
    (
        sd_path, cfg_path, meta_path, adapter_path, _, head_path,
        cls_path, _, proj_path, proto_path
    ) = _local_paths(local_dir)

    torch.save(pretrainmodel.state_dict(), sd_path)
    torch.save(adapter.state_dict(), adapter_path)
    torch.save(pred_head.state_dict(), head_path)
    torch.save(proj_head.state_dict(), proj_path)

    if cls_head is not None:
        torch.save(cls_head.state_dict(), cls_path)
    if proto_layer is not None:
        torch.save(proto_layer.state_dict(), proto_path)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(pretrainconfig, f, ensure_ascii=False, indent=2)

    save_feature_list(feature_list, local_dir)

    if label2id is not None:
        save_label_meta(label2id, local_dir)

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "device_note": str(next(pretrainmodel.parameters()).device),
    }

    if train_args is not None:
        meta.update({
            "mask_prob": float(train_args.mask_prob),
            "valid_eps": float(train_args.valid_eps),
            "mask_mode": str(train_args.mask_mode),
            "mask_eps": float(train_args.mask_eps),
            "mae_encoder_max_seq_len": int(train_args.mae_encoder_max_seq_len),
            "encoder_visible_max_len": int(train_args.encoder_visible_max_len),
            "encoder_topk_by": str(train_args.encoder_topk_by),
            "lr": float(getattr(train_args, "lr", getattr(train_args, "sc_lr", 0.0))),
            "backbone_lr_scale": float(train_args.backbone_lr_scale),
            "batch_size": int(getattr(train_args, "batch_size", getattr(train_args, "sc_batch_size", None))),
            "accum_steps": int(train_args.accum_steps),
            "epochs": int(getattr(train_args, "epochs", getattr(train_args, "sc_epochs", None))),
            "objective": "MAE + semi-supervised CE + SupCon + Prototype + Consistency + optional PseudoLabel",
            "freeze_scfoundation": bool(train_args.freeze_scfoundation),
            "unfreeze_mode": str(train_args.unfreeze_mode),
            "unfreeze_last_n": int(train_args.unfreeze_last_n),
            "semi_supervised": bool(train_args.semi_supervised),
            "labeled_ratio": float(train_args.labeled_ratio),
            "proj_dim": int(train_args.proj_dim),
            "supcon_temperature": float(train_args.supcon_temperature),
            "proto_temperature": float(train_args.proto_temperature),
            "w_mae": float(train_args.w_mae),
            "w_ce": float(train_args.w_ce),
            "w_supcon": float(train_args.w_supcon),
            "w_proto": float(train_args.w_proto),
            "w_cons": float(train_args.w_cons),
            "w_pseudo": float(train_args.w_pseudo),
        })

    if extra_meta:
        meta.update(extra_meta)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[LOCAL] Saved to:", local_dir)
    diagnostic_print("  -", sd_path)
    diagnostic_print("  -", adapter_path)
    diagnostic_print("  -", head_path)
    diagnostic_print("  -", proj_path)
    if cls_head is not None:
        diagnostic_print("  -", cls_path)
    if proto_layer is not None:
        diagnostic_print("  -", proto_path)
    diagnostic_print("  -", cfg_path)
    diagnostic_print("  -", meta_path)
