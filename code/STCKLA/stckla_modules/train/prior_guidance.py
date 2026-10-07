"""train.py definitions moved here without algorithm changes."""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
