"""base.py definitions moved here without algorithm changes."""

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.ndim != 2 or labels.ndim != 1 or features.size(0) != labels.size(0):
        raise ValueError(f"[SupCon] shape mismatch: features={features.shape}, labels={labels.shape}")

    device = features.device
    N = features.size(0)
    if N <= 1:
        return torch.zeros((), device=device, dtype=features.dtype)

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / float(temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(N, device=device)
    mask = mask * logits_mask

    pos_cnt = mask.sum(dim=1)
    valid_rows = pos_cnt > 0
    if not bool(valid_rows.any().item()):
        return torch.zeros((), device=device, dtype=features.dtype)

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_cnt + 1e-12)
    loss = -mean_log_prob_pos[valid_rows].mean()
    return loss


def prototype_loss(z_l: torch.Tensor, y_l: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if z_l.numel() == 0:
        return torch.zeros((), device=z_l.device, dtype=z_l.dtype)
    logits = torch.matmul(z_l, prototypes.T) / float(temperature)
    return F.cross_entropy(logits, y_l)


def consistency_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    if z1.numel() == 0 or z2.numel() == 0:
        return torch.zeros((), device=z1.device, dtype=z1.dtype)
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()
