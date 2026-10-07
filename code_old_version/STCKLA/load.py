
import sys
sys.path.append('/data/LJL/scFoundationmain/scFoundationmain/model')
import os
import random
import math
import numpy as np
import torch
from pretrainmodels import select_model


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def convertconfig(ckpt):
    newconfig = {}
    newconfig["config"] = {}
    model_type = ckpt["config"]["model"]

    for key, val in ckpt["config"]["model_config"][model_type].items():
        newconfig["config"][key] = val

    for key, val in ckpt["config"]["dataset_config"]["rnaseq"].items():
        newconfig["config"][key] = val

    if model_type == "performergau_resolution":
        model_type = "performer_gau"

    import collections
    d = collections.OrderedDict()
    for key, val in ckpt["state_dict"].items():
        d[str(key).split("model.")[1]] = val

    newconfig["config"]["model_type"] = model_type
    newconfig["model_state_dict"] = d
    newconfig["config"]["pos_embed"] = False
    newconfig["config"]["device"] = "cuda"
    return newconfig


def load_model_frommmf(best_ckpt_path, key="gene"):
    model_data = torch.load(best_ckpt_path, map_location="cpu")
    model_data = model_data[key]
    model_data = convertconfig(model_data)

    config = model_data["config"]
    if "qv_dim" not in config:
        if config.get("model", "") != "mae_autobin":
            if "dim_head" in config:
                config["qv_dim"] = config["dim_head"]
            else:
                config["qv_dim"] = 64

    if "ppi_edge" not in config:
        config["ppi_edge"] = None

    model = select_model(config)
    model_state_dict = model_data["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model.cuda(), config


# -----------------------------
# MAE gatherData: cap length by mae_encoder_max_seq_len
# -----------------------------
def gatherData(data: torch.Tensor, labels: torch.Tensor, pad_token_id: int, *, max_len: int):
    """
    data:   [B, N]
    labels: [B, N] bool, True means "keep"
    output: [B, L] where L = min(max(labels.sum), max_len), padded by pad_token_id
    """
    assert data.dim() == 2
    assert labels.shape == data.shape
    B, N = data.shape
    device = data.device

    value_nums = labels.sum(1)
    L = int(value_nums.max().item())
    L = max(L, 1)
    L = min(L, int(max_len))

    none_labels = ~labels
    score = labels.float()
    score[none_labels] = torch.tensor(-float("Inf"), device=device)

    # 偏置：让 topk 更偏向前面的 idx（稳定）
    bias = torch.arange(N, device=device).float()
    bias = (N - bias) * 20000.0
    score = score + bias

    topk_idx = score.topk(L, dim=1).indices  # [B, L]
    new_data = torch.gather(data, 1, topk_idx)

    if new_data.dtype.is_floating_point:
        padding_labels = (new_data == float(pad_token_id))
    else:
        padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def getEncoerDecoderData(data: torch.Tensor, data_raw: torch.Tensor, config: dict):
    """
    MAE（方案2 兼容）：
    - valid token：abs(x)>valid_eps（保留负数意义）
    - mask：在 valid token 中按 mask_prob 采样（每样本至少 1 个 mask）
    - encoder 输入：只保留 visible（valid 且未 mask）+ 最后两个 extra token
    - decoder 输入：全长，mask 位替换为 mask_token_id
    - encoder_labels：全长 visible 标签（True=visible）
    - data_mask_labels：全长 mask 标签（True=mask）
    """
    device = data.device
    B, N = data.shape

    pad_token_id = int(config.get("pad_token_id", 103))
    mask_token_id = int(config.get("mask_token_id", 102))
    seq_len = int(config.get("seq_len", N))
    max_enc_len = int(config.get("mae_encoder_max_seq_len", max(1, N - 1)))

    valid_eps = float(config.get("valid_eps", 1e-12))
    mask_prob = float(config.get("mask_prob", 0.30))

    decoder_data = data.clone()
    decoder_data = torch.nan_to_num(decoder_data, nan=0.0, posinf=0.0, neginf=0.0)
    decoder_data_padding = torch.zeros_like(decoder_data, dtype=torch.bool, device=device)

    # gene_len：只 mask 前 N-2（最后两个 extra token 不 mask）
    gene_len = max(1, N - 2)

    gene_vals = torch.nan_to_num(data_raw[:, :gene_len], nan=0.0, posinf=0.0, neginf=0.0)
    valid_gene = (gene_vals.abs() > valid_eps)

    data_mask_labels = torch.zeros((B, N), dtype=torch.bool, device=device)

    # 逐样本采样 mask（至少 1 个）
    for i in range(B):
        idx = torch.nonzero(valid_gene[i], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            # 全 0：强制 mask 0 号位，避免 n_valid=0
            data_mask_labels[i, 0] = True
            continue

        m = max(1, int(round(idx.numel() * mask_prob)))
        perm = idx[torch.randperm(idx.numel(), device=device)]
        pick = perm[:m]
        data_mask_labels[i, pick] = True

    # decoder：mask 位替换为 mask_token_id（float）
    decoder_data[data_mask_labels] = float(mask_token_id)

    # encoder_keep：visible=valid & ~mask；最后两个 extra token 强制 visible
    encoder_keep = torch.zeros((B, N), dtype=torch.bool, device=device)
    encoder_keep[:, :gene_len] = valid_gene & (~data_mask_labels[:, :gene_len])
    if N >= 2:
        encoder_keep[:, -2:] = True

    # gather encoder data
    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_keep, pad_token_id, max_len=max_enc_len)

    # encoder position ids（按同样 keep gather）
    data_gene_ids = torch.arange(N, device=device).repeat(B, 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_keep, pad_token_id, max_len=max_enc_len)

    decoder_position_gene_ids = data_gene_ids

    # encoder_labels：全长 visible 标签（给 mae_autobin 写回 visible token）
    encoder_labels = encoder_keep

    new_data_raw = torch.nan_to_num(data_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # positional pad 修正
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
