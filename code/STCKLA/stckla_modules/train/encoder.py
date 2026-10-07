"""train.py definitions moved here without algorithm changes."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import load as load_mod
from base import sdp_kernel_ctx, amp_autocast_kwargs, build_x_all, pool_tokens_like_train
from .prior_guidance import (
    apply_qwen_visible_gene_injection,
    restore_encoder_values_after_qwen_visible_injection,
)


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
