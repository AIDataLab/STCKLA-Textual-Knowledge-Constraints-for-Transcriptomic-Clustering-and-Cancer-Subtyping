"""base.py definitions moved here without algorithm changes."""

from typing import Optional
import torch
from ..logging_utils import diagnostic_print


def gatherData_fixed_len_impl(
    orig_gather_fn,
    data: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    *,
    max_len: Optional[int] = None,
    **kwargs,
):
    try:
        if max_len is None:
            return orig_gather_fn(data, labels, pad_token_id, max_len=int(data.shape[1]))
        return orig_gather_fn(data, labels, pad_token_id, max_len=int(max_len))
    except TypeError:
        pass

    if data.dim() != 2:
        raise ValueError(f"gatherData expects [B,N], got {tuple(data.shape)}")
    assert labels.shape == data.shape
    B, N = data.shape

    keep = labels.bool().clone()
    for i in range(B):
        if int(keep[i].sum().item()) == 0:
            keep[i, 0] = True

    L = int(keep.sum(1).max().item())
    L = max(1, L)
    if max_len is not None:
        L = min(L, int(max_len))

    score = keep.float()
    score[~keep] = float("-inf")
    bias = torch.arange(N, device=data.device).float()
    bias = (N - bias) * 20000.0
    score = score + bias

    topk_idx = score.topk(L, dim=1).indices
    new_data = torch.gather(data, 1, topk_idx)

    if new_data.dtype.is_floating_point:
        padding_labels = (new_data == float(pad_token_id))
    else:
        padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def _topk_keep_mask(scores_1d: torch.Tensor, k: int) -> torch.Tensor:
    if k >= scores_1d.numel():
        return torch.ones_like(scores_1d, dtype=torch.bool)
    idx = torch.topk(scores_1d, k=k, largest=True, sorted=False).indices
    m = torch.zeros_like(scores_1d, dtype=torch.bool)
    m[idx] = True
    return m


def build_get_encoder_decoder_patch_fn(
    *,
    encoder_visible_max_len: int,
    encoder_topk_by: str,
):
    def getEncoerDecoderData_encoder_sees_valid(
        data: torch.Tensor,
        data_raw: torch.Tensor,
        config: dict
    ):
        device = data.device
        B, N = data.shape

        pad_token_id = int(config.get("pad_token_id", 103))
        mask_token_id = int(config.get("mask_token_id", 102))
        seq_len = int(config.get("seq_len", N))
        max_enc_len = int(config.get("mae_encoder_max_seq_len", max(1, N - 1)))

        valid_eps = float(config.get("valid_eps", 1e-12))
        mask_prob = float(config.get("mask_prob", 0.30))
        mask_mode = str(config.get("mask_mode", "abs_eps")).lower().strip()
        mask_eps = float(config.get("mask_eps", valid_eps))

        enc_budget = int(encoder_visible_max_len)
        if enc_budget <= 0:
            enc_budget = max_enc_len

        gene_len = max(1, N - 2)
        gene_vals = torch.nan_to_num(data_raw[:, :gene_len], nan=0.0, posinf=0.0, neginf=0.0)

        if mask_mode == "finite":
            valid_gene = torch.isfinite(gene_vals)
            score_vals = gene_vals.abs()
        elif mask_mode == "pos":
            valid_gene = torch.isfinite(gene_vals) & (gene_vals > mask_eps)
            score_vals = gene_vals if str(encoder_topk_by).lower().strip() == "raw" else gene_vals.abs()
        elif mask_mode in ["abs_eps", "abs"]:
            valid_gene = torch.isfinite(gene_vals) & (gene_vals.abs() > mask_eps)
            score_vals = gene_vals.abs()
        else:
            raise ValueError(f"Unknown mask_mode={mask_mode}, expected finite | pos | abs_eps")

        data_mask_labels = torch.zeros((B, N), dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.nonzero(valid_gene[i], as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                data_mask_labels[i, 0] = True
                continue
            m = max(1, int(round(idx.numel() * mask_prob)))
            perm = idx[torch.randperm(idx.numel(), device=device)]
            pick = perm[:m]
            data_mask_labels[i, pick] = True

        encoder_keep = torch.zeros((B, N), dtype=torch.bool, device=device)
        encoder_keep[:, :gene_len] = valid_gene
        if N >= 2:
            encoder_keep[:, -2:] = True

        gene_budget = max(1, int(enc_budget) - 2)
        topk_by = str(encoder_topk_by).lower().strip()

        for i in range(B):
            vis_idx = torch.nonzero(encoder_keep[i, :gene_len], as_tuple=False).squeeze(1)
            if vis_idx.numel() <= gene_budget:
                continue

            if topk_by == "raw":
                scores = gene_vals[i, vis_idx]
            else:
                scores = score_vals[i, vis_idx]
            keep_local = _topk_keep_mask(scores, k=gene_budget)
            new_vis_idx = vis_idx[keep_local]
            encoder_keep[i, :gene_len] = False
            encoder_keep[i, new_vis_idx] = True

        decoder_data = torch.nan_to_num(data.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        decoder_data[data_mask_labels] = float(mask_token_id)
        decoder_data_padding = torch.zeros_like(decoder_data, dtype=torch.bool, device=device)

        encoder_source = torch.nan_to_num(data.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        encoder_data, encoder_data_padding = gatherData_fixed_len_impl(
            config["_orig_gather_fn"], encoder_source, encoder_keep, pad_token_id, max_len=max_enc_len
        )

        data_gene_ids = torch.arange(N, device=device).repeat(B, 1)
        encoder_position_gene_ids, _ = gatherData_fixed_len_impl(
            config["_orig_gather_fn"], data_gene_ids, encoder_keep, pad_token_id, max_len=max_enc_len
        )

        decoder_position_gene_ids = data_gene_ids
        encoder_labels = encoder_keep

        new_data_raw = torch.nan_to_num(data_raw, nan=0.0, posinf=0.0, neginf=0.0)
        encoder_position_gene_ids[encoder_data_padding] = seq_len
        decoder_position_gene_ids[decoder_data_padding] = seq_len

        return (
            encoder_data,
            encoder_position_gene_ids,
            encoder_data_padding,
            encoder_labels,
            decoder_data,
            decoder_data_padding,
            new_data_raw,
            data_mask_labels,
            decoder_position_gene_ids,
        )

    return getEncoerDecoderData_encoder_sees_valid


def install_scfoundation_patches(load_mod, *, encoder_visible_max_len: int, encoder_topk_by: str):
    orig_gather = getattr(load_mod, "gatherData", None)
    if orig_gather is None:
        raise RuntimeError("[PATCH] load_mod.gatherData not found. Check your load.py.")

    def patched_gatherData(data, labels, pad_token_id, *, max_len=None, **kwargs):
        return gatherData_fixed_len_impl(
            orig_gather,
            data,
            labels,
            pad_token_id,
            max_len=max_len,
            **kwargs,
        )

    orig_get_ed = getattr(load_mod, "getEncoerDecoderData", None)
    if orig_get_ed is None:
        raise RuntimeError("[PATCH] load_mod.getEncoerDecoderData not found. Check your load.py.")

    patched_get_ed = build_get_encoder_decoder_patch_fn(
        encoder_visible_max_len=encoder_visible_max_len,
        encoder_topk_by=encoder_topk_by,
    )

    load_mod.gatherData = patched_gatherData

    def wrapped_get_ed(data, data_raw, config):
        config = dict(config)
        config["_orig_gather_fn"] = orig_gather
        return patched_get_ed(data, data_raw, config)

    load_mod.getEncoerDecoderData = wrapped_get_ed
    diagnostic_print("[PATCH] gatherData patched.")
    diagnostic_print("[PATCH] getEncoerDecoderData patched: encoder sees VALID (topk-capped), mask only for loss/decoder.")
