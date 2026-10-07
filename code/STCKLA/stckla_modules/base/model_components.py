"""base.py definitions moved here without algorithm changes."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 1024, alpha: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ln = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, int(hidden_dim))
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), self.dim)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        y = x + self.alpha * h
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrototypeLayer(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)


def build_x_all(gene_x: torch.Tensor, *, pre_normalized: str, totalcount_mode: str, eps: float) -> torch.Tensor:
    if totalcount_mode == "sumabs":
        s = gene_x.abs().sum(dim=1)
    else:
        s = gene_x.sum(dim=1)
    s = torch.clamp(s, min=eps)

    if pre_normalized == "T":
        totalcount = s
    elif pre_normalized == "F":
        totalcount = torch.log10(s)
    else:
        raise ValueError("pre_normalized must be 'T' or 'F'")

    totalcount = torch.nan_to_num(totalcount, nan=float(eps), posinf=float(eps), neginf=float(eps))
    x_all = torch.cat([gene_x, totalcount[:, None], totalcount[:, None]], dim=1)
    x_all = torch.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
    return x_all


def pool_tokens_like_train(h: torch.Tensor) -> torch.Tensor:
    geneemb1 = h[:, -1, :]
    geneemb2 = h[:, -2, :] if h.shape[1] >= 2 else geneemb1
    if h.shape[1] > 2:
        core = h[:, :-2, :]
        geneemb3, _ = torch.max(core, dim=1)
        geneemb4 = torch.mean(core, dim=1)
    else:
        geneemb3, _ = torch.max(h, dim=1)
        geneemb4 = torch.mean(h, dim=1)
    return torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)


class ExprSemiDataset(Dataset):
    def __init__(self, x: np.ndarray, y_id: Optional[np.ndarray], labeled_mask: Optional[np.ndarray], unlabeled_label_value: int = -1):
        self.x = np.asarray(x, dtype=np.float32)
        self.y_id = None if y_id is None else np.asarray(y_id, dtype=np.int64)
        self.labeled_mask = None if labeled_mask is None else np.asarray(labeled_mask, dtype=bool)
        self.unlabeled_label_value = int(unlabeled_label_value)

        if self.y_id is not None and len(self.y_id) != len(self.x):
            raise RuntimeError(f"[DATA] label length {len(self.y_id)} != n_samples {len(self.x)}")
        if self.labeled_mask is not None and len(self.labeled_mask) != len(self.x):
            raise RuntimeError(f"[DATA] labeled_mask length {len(self.labeled_mask)} != n_samples {len(self.x)}")

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        x = torch.from_numpy(self.x[idx])
        if self.y_id is None or self.labeled_mask is None:
            y = torch.tensor(self.unlabeled_label_value, dtype=torch.long)
            is_lab = torch.tensor(False, dtype=torch.bool)
        else:
            if bool(self.labeled_mask[idx]):
                y = torch.tensor(int(self.y_id[idx]), dtype=torch.long)
                is_lab = torch.tensor(True, dtype=torch.bool)
            else:
                y = torch.tensor(self.unlabeled_label_value, dtype=torch.long)
                is_lab = torch.tensor(False, dtype=torch.bool)
        return x, y, is_lab


def augment_expression(x: torch.Tensor, noise_std: float = 0.01, drop_prob: float = 0.05, input_clip: float = 50.0) -> torch.Tensor:
    x = x.clone()
    if noise_std > 0:
        x = x + torch.randn_like(x) * float(noise_std)
    if drop_prob > 0:
        keep = (torch.rand_like(x) > float(drop_prob)).float()
        x = x * keep
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.clamp(-float(input_clip), float(input_clip))
    return x
