

import os
import json
import time
import random
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment


# Utils
def set_seed(seed: int = 0, deterministic: bool = True):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)


@contextmanager
def sdp_kernel_ctx(device: torch.device, force_sdp: bool):
    if device.type != "cuda" or (not force_sdp):
        yield
        return
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=True,
            enable_math=True,
        ):
            yield
    except TypeError:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        yield


def amp_autocast_kwargs(amp: bool, amp_dtype: str):
    if (not amp) or (not torch.cuda.is_available()):
        return dict(enabled=False)
    dt = str(amp_dtype).lower().strip()
    if dt == "bf16":
        return dict(enabled=True, dtype=torch.bfloat16)
    return dict(enabled=True, dtype=torch.float16)


# GatherData / getEncoderDecoder patch
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
    print("[PATCH] gatherData patched.")
    print("[PATCH] getEncoerDecoderData patched: encoder sees VALID (topk-capped), mask only for loss/decoder.")


# Local save/resume helpers
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
    print("[RESUME] adapter loaded.")

    if os.path.isfile(head_path):
        head_sd = torch.load(head_path, map_location="cpu")
        pred_head.load_state_dict(head_sd, strict=False)
        print("[RESUME] pred_head loaded.")

    if os.path.isfile(proj_path):
        proj_sd = torch.load(proj_path, map_location="cpu")
        proj_head.load_state_dict(proj_sd, strict=True)
        print("[RESUME] proj_head loaded.")

    if (cls_head is not None) and os.path.isfile(cls_path):
        cls_sd = torch.load(cls_path, map_location="cpu")
        cls_head.load_state_dict(cls_sd, strict=True)
        print("[RESUME] cls_head loaded.")
    elif (cls_head is not None) and (not os.path.isfile(cls_path)):
        print("[RESUME][INFO] cls_head.pt not found. cls_head starts fresh.")

    if (proto_layer is not None) and os.path.isfile(proto_path):
        proto_sd = torch.load(proto_path, map_location="cpu")
        proto_layer.load_state_dict(proto_sd, strict=True)
        print("[RESUME] proto_layer loaded.")
    elif (proto_layer is not None) and (not os.path.isfile(proto_path)):
        print("[RESUME][INFO] proto_layer.pt not found. proto_layer starts fresh.")

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        print("[RESUME] pretrainconfig.json loaded (info). keys head:", list(_cfg.keys())[:10])
    except Exception:
        pass

    if os.path.isfile(label_meta_path):
        try:
            with open(label_meta_path, "r", encoding="utf-8") as f:
                _lm = json.load(f)
            print("[RESUME] label_meta.json loaded (info).")
            print("         num_classes:", _lm.get("num_classes"), "mapping_head:", list(_lm.get("label2id", {}).items())[:5])
        except Exception:
            pass

    return True


def save_feature_list(feature_list: List[str], local_dir: str):
    _, _, _, _, featlist_path, _, _, _, _, _ = _local_paths(local_dir)
    with open(featlist_path, "w", encoding="utf-8") as f:
        json.dump({"feature_list": feature_list}, f, ensure_ascii=False, indent=2)
    print("[LOCAL] Saved feature_list:", featlist_path)


def save_label_meta(label2id: Dict[str, int], local_dir: str):
    label_meta_path = _local_paths(local_dir)[7]
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_classes": int(len(label2id)),
        "label2id": {str(k): int(v) for k, v in label2id.items()},
    }
    with open(label_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[LOCAL] Saved label_meta:", label_meta_path)


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
    print("  -", sd_path)
    print("  -", adapter_path)
    print("  -", head_path)
    print("  -", proj_path)
    if cls_head is not None:
        print("  -", cls_path)
    if proto_layer is not None:
        print("  -", proto_path)
    print("  -", cfg_path)
    print("  -", meta_path)


# save / load exposed sample ids
def save_exposed_ids(
    save_dir: str,
    sample_index: pd.Index,
    labeled_mask: np.ndarray,
    *,
    seed: int,
    labeled_ratio: float,
    stratified: bool,
    label_path: str,
):
    os.makedirs(save_dir, exist_ok=True)

    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    if labeled_mask.ndim != 1:
        raise RuntimeError(f"[EXPOSED] labeled_mask must be 1D, got shape={labeled_mask.shape}")

    n_total = int(len(sample_index))
    if n_total != len(labeled_mask):
        raise RuntimeError(f"[EXPOSED] sample_index len {n_total} != labeled_mask len {len(labeled_mask)}")

    rows = np.where(labeled_mask)[0].astype(np.int64)
    exposed_rows = rows.tolist()
    # Use row indices to keep sample IDs consistent across file formats.
    exposed_ids = [str(i) for i in exposed_rows]

    with open(os.path.join(save_dir, "exposed_ids.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "exposed_ids": exposed_ids,
                "exposed_rows": exposed_rows,
                "id_type": "row_index",
                "n_exposed": int(len(exposed_ids)),
                "n_total": n_total,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.save(os.path.join(save_dir, "exposed_rows.npy"), rows)
    np.save(os.path.join(save_dir, "exposed_mask.npy"), labeled_mask.astype(np.bool_))

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "save_dir": str(save_dir),
        "label_path": str(label_path),
        "seed": int(seed),
        "labeled_ratio": float(labeled_ratio),
        "stratified_sample": bool(stratified),
        "sampling_mode": "class-balanced random sampling" if bool(stratified) else "global random sampling",
        "id_type": "row_index",
        "n_total": n_total,
        "n_exposed": int(len(exposed_ids)),
        "files": ["exposed_ids.json", "exposed_rows.npy", "exposed_mask.npy", "exposed_meta.json"],
    }
    with open(os.path.join(save_dir, "exposed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[EXPOSED] saved exposed set to: {save_dir}")
    print(f"          n_exposed={meta['n_exposed']} / n_total={meta['n_total']} | id_type=row_index")


def load_fixed_exposed_mask_if_exists(save_dir: str, n_samples: int) -> Optional[np.ndarray]:
    path = os.path.join(save_dir, "exposed_mask.npy")
    if not os.path.isfile(path):
        return None
    try:
        mask = np.load(path)
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != int(n_samples):
            print(f"[EXPOSED][WARN] existing exposed_mask.npy shape mismatch: got {mask.shape}, expect ({n_samples},)")
            return None
        print(f"[EXPOSED] loaded fixed exposed_mask from {path}, labeled_cnt={int(mask.sum())}/{len(mask)}")
        return mask
    except Exception as e:
        print("[EXPOSED][WARN] failed to load exposed_mask.npy:", repr(e))
        return None


def load_exposed_sample_ids(train_meta_dir: str) -> Optional[List[str]]:
    if not train_meta_dir:
        return None
    cand = [
        os.path.join(train_meta_dir, "exposed_ids.json"),
        os.path.join(train_meta_dir, "exposed_samples.json"),
        os.path.join(train_meta_dir, "exposed_sample_ids.json"),
    ]
    for p in cand:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if isinstance(meta, dict):
            # Prefer row-index IDs.
            for k in ["exposed_rows", "rows", "exposed_row_indices"]:
                if k in meta and isinstance(meta[k], list):
                    ids = [str(int(x)) for x in meta[k]]
                    print(f"[EXPOSED] loaded {len(ids)} exposed ids from {p} (key={k}, interpreted=row_index)")
                    return ids
            for k in ["exposed_sample_ids", "exposed_ids", "exposed"]:
                if k in meta and isinstance(meta[k], list):
                    ids = [str(x) for x in meta[k]]
                    print(f"[EXPOSED] loaded {len(ids)} exposed ids from {p} (key={k})")
                    return ids
        elif isinstance(meta, list):
            ids = [str(x) for x in meta]
            print(f"[EXPOSED] loaded {len(ids)} exposed ids(list) from {p}")
            return ids
    print(f"[EXPOSED][WARN] no exposed ids json found in: {train_meta_dir}")
    return None


def build_keep_mask_from_exposed(sample_ids: List[str], exposed_ids: List[str]) -> np.ndarray:
    sample_ids_str = [str(x) for x in sample_ids]
    exposed_set = set(str(x) for x in exposed_ids)
    hit = sum(sid in exposed_set for sid in sample_ids_str)
    keep_mask = np.array([sid not in exposed_set for sid in sample_ids_str], dtype=bool)
    print(f"[EXPOSED][CHECK] sample_ids={len(sample_ids_str)} exposed_ids={len(exposed_set)} hit={hit} keep={int(keep_mask.sum())}")
    return keep_mask


def save_eval_split(save_dir: str, sample_ids: List[str], keep_mask: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    eval_ids = [str(sample_ids[i]) for i in np.where(keep_mask)[0].tolist()]
    exposed_ids = [str(sample_ids[i]) for i in np.where(~keep_mask)[0].tolist()]
    with open(os.path.join(save_dir, "eval_split_used.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total": int(len(sample_ids)),
                "n_eval_keep": int(len(eval_ids)),
                "n_exposed_filtered": int(len(exposed_ids)),
                "eval_keep_sample_ids": eval_ids[:5000],
                "exposed_filtered_sample_ids": exposed_ids[:5000]
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    pd.DataFrame({
        "sample_id": [str(x) for x in sample_ids],
        "row_id": np.arange(len(sample_ids), dtype=np.int64),
        "is_eval_keep": keep_mask.astype(int)
    }).to_csv(os.path.join(save_dir, "eval_split_used.csv"), index=False)


# Gene align helpers
def _clean_gene_names(cols):
    return [str(c).strip() for c in cols]


def gene_align_diagnostics(df_cols, gene_list):
    cols = set(_clean_gene_names(df_cols))
    gl = set(_clean_gene_names(gene_list))
    overlap = len(cols & gl)
    missing = len(gl - cols)
    extra = len(cols - gl)
    print(f"[GENE][CHECK] raw_cols={len(cols)} | gene_list={len(gl)} | overlap={overlap} | missing_to_pad={missing} | extra_not_in_list={extra}")
    if overlap < 500:
        print("[GENE][WARN] overlap < 500: very likely NOT gene-symbol aligned -> huge zero padding -> embedding degrades.")
    elif overlap < 5000:
        print("[GENE][WARN] overlap is relatively low; alignment may be partial.")
    else:
        print("[GENE][OK] overlap looks reasonable.")
    return overlap, missing, extra


def load_gene_list(gene_index_path: str) -> List[str]:
    gene_list_df = pd.read_csv(gene_index_path, header=0, delimiter="\t")
    gl = [str(x) for x in list(gene_list_df["gene_name"])]
    seen = set()
    out = []
    for g in gl:
        g = str(g).strip()
        if g and (g not in seen):
            out.append(g)
            seen.add(g)
    return out


def ensure_samples_by_rows_no_label(
    df: pd.DataFrame,
    gene_list: List[str],
    *,
    orient_overlap_min: int,
    orient_ratio: float,
) -> pd.DataFrame:
    cols = set(_clean_gene_names(df.columns))
    idxs = set(_clean_gene_names(df.index))
    gl = set(_clean_gene_names(gene_list))

    overlap_cols = len(cols & gl)
    overlap_idx = len(idxs & gl)

    print(f"[ALIGN-NOLABEL] overlap(columns,gene_list)={overlap_cols} | overlap(index,gene_list)={overlap_idx}")

    if overlap_cols >= orient_overlap_min or overlap_idx >= orient_overlap_min:
        if overlap_cols > overlap_idx * orient_ratio:
            print("[ALIGN-NOLABEL] assume df is (samples, genes) based on columns overlap.")
            return df
        if overlap_idx > overlap_cols * orient_ratio:
            print("[ALIGN-NOLABEL] assume df is (genes, samples) -> transpose to (samples, genes).")
            return df.T

    print("[ALIGN-NOLABEL][WARN] cannot confidently infer orientation; keep as-is.")
    return df


# Data IO / clean
def dedup_columns(df: pd.DataFrame, policy: str = "sum") -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(str)
    if not df.columns.duplicated().any():
        return df
    if policy == "first":
        return df.loc[:, ~df.columns.duplicated(keep="first")]
    if policy == "mean":
        return df.T.groupby(df.columns).mean().T
    return df.T.groupby(df.columns).sum().T


def read_any_to_df_raw(path: str) -> pd.DataFrame:
    if path.endswith("npz"):
        mat = scipy.sparse.load_npz(path)
        return pd.DataFrame(mat.toarray())
    if path.endswith("h5ad"):
        ad = sc.read_h5ad(path)
        idx = ad.obs_names.tolist()
        try:
            col = ad.var.gene_name.tolist()
        except Exception:
            col = ad.var_names.tolist()
        mat = ad.X.toarray() if issparse(ad.X) else ad.X
        return pd.DataFrame(mat, index=idx, columns=col)
    if path.endswith("npy"):
        return pd.DataFrame(np.load(path))
    return pd.read_csv(path, index_col=0)


def clean_numeric_df(df: pd.DataFrame, *, input_clip: float, dedup_policy: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index([str(x).strip() for x in df.columns.map(str)])
    bad = (df.columns == "") | (pd.Series(df.columns).str.lower().values == "nan")
    if bad.any():
        df = df.loc[:, ~bad]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.clip(lower=-float(input_clip), upper=float(input_clip))
    df = dedup_columns(df, policy=str(dedup_policy))
    return df


def align_to_gene_list(df: pd.DataFrame, gene_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = df.columns.map(str)
    gene_list = [str(x) for x in gene_list]

    missing = list(set(gene_list) - set(df.columns))
    if len(missing) > 0:
        pad = pd.DataFrame(
            np.zeros((df.shape[0], len(missing)), dtype=np.float32),
            index=df.index,
            columns=missing,
        )
        df = pd.concat([df, pad], axis=1)
    df = df[gene_list]
    return df, missing


# Adapter / Heads
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


# Label IO
def load_labels(label_path: str, sample_index: Optional[pd.Index] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    df = pd.read_csv(label_path)
    if df.shape[1] < 1:
        raise RuntimeError(f"[LABEL] label csv has no columns: {label_path}")

    y_raw = df.iloc[:, 0].astype(str).values

    if sample_index is not None:
        candidates = []
        for c in df.columns[1:]:
            lc = str(c).lower()
            if ("sample" in lc) or (lc in ["id", "sid", "sample_id", "case_id"]):
                candidates.append(c)
        if df.shape[1] >= 2:
            candidates = [df.columns[1]] + candidates

        used = None
        for c in candidates:
            try:
                sids = df[c].astype(str)
                if len(set(sids)) == len(sids):
                    tmp = df.copy()
                    tmp.index = sids.values
                    inter = len(set(tmp.index) & set(sample_index.astype(str)))
                    if inter >= int(0.8 * len(sample_index)):
                        tmp = tmp.reindex(sample_index.astype(str))
                        if tmp.iloc[:, 0].isna().any():
                            continue
                        y_raw = tmp.iloc[:, 0].astype(str).values
                        used = c
                        break
            except Exception:
                continue
        if used is not None:
            print(f"[LABEL] aligned by sample id column: {used}")
        else:
            print("[LABEL] fallback: align labels by row order (requires same length)")

    if sample_index is not None and len(y_raw) != len(sample_index):
        raise RuntimeError(f"[LABEL] label length {len(y_raw)} != n_samples {len(sample_index)}.")

    uniq = sorted(list({str(x) for x in y_raw}))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    y_id = np.array([label2id[str(x)] for x in y_raw], dtype=np.int64)

    print(f"[LABEL] loaded labels: n={len(y_id)} num_classes={len(label2id)} head={list(label2id.items())[:5]}")
    return y_id, label2id


def pick_labeled_indices(y_id: np.ndarray, ratio: float, seed: int = 0, stratified: bool = True) -> np.ndarray:
    n = int(len(y_id))
    rng = np.random.RandomState(seed)
    ratio = float(ratio)
    ratio = min(max(ratio, 0.0), 1.0)

    m = int(round(n * ratio))
    m = max(1, m) if ratio > 0 else 0
    labeled_mask = np.zeros(n, dtype=bool)

    if m <= 0:
        return labeled_mask

    classes = np.unique(y_id)
    num_classes = len(classes)

    if (not stratified) or (num_classes <= 1):
        idx = rng.permutation(n)[:m]
        labeled_mask[idx] = True
        return labeled_mask

    class_to_indices = {int(c): np.where(y_id == c)[0] for c in classes}
    capacities = {int(c): len(class_to_indices[int(c)]) for c in classes}

    if m < num_classes:
        chosen_classes = rng.permutation(classes)[:m]
        for c in chosen_classes:
            idx_c = class_to_indices[int(c)]
            pick = rng.permutation(idx_c)[:1]
            labeled_mask[pick] = True
        return labeled_mask

    quotas = {int(c): 0 for c in classes}
    base = m // num_classes
    remainder = m % num_classes

    for c in classes:
        quotas[int(c)] = min(base, capacities[int(c)])

    assigned = sum(quotas.values())
    leftover = m - assigned

    if remainder > 0 and leftover > 0:
        eligible = [int(c) for c in rng.permutation(classes) if quotas[int(c)] < capacities[int(c)]]
        for c in eligible[:min(remainder, leftover)]:
            quotas[c] += 1
            leftover -= 1

    while leftover > 0:
        eligible = [int(c) for c in classes if quotas[int(c)] < capacities[int(c)]]
        if len(eligible) == 0:
            break
        rng.shuffle(eligible)
        progressed = False
        for c in eligible:
            if leftover <= 0:
                break
            if quotas[c] < capacities[c]:
                quotas[c] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break

    picks = []
    for c in classes:
        c = int(c)
        k = int(quotas[c])
        if k <= 0:
            continue
        idx_c = class_to_indices[c]
        pick = rng.permutation(idx_c)[:k]
        picks.append(pick)

    picks = np.concatenate(picks) if len(picks) else np.array([], dtype=int)

    if len(picks) < m:
        rest = np.setdiff1d(np.arange(n), picks, assume_unique=False)
        add = rng.permutation(rest)[: (m - len(picks))]
        picks = np.concatenate([picks, add])
    elif len(picks) > m:
        picks = rng.permutation(picks)[:m]

    labeled_mask[picks] = True
    return labeled_mask


def load_or_create_fixed_labeled_mask(
    y_id: np.ndarray,
    sample_index: pd.Index,
    save_dir: str,
    *,
    ratio: float,
    seed: int,
    stratified: bool,
    label_path: str,
) -> np.ndarray:
    fixed = load_fixed_exposed_mask_if_exists(save_dir, n_samples=len(sample_index))
    if fixed is not None:
        return fixed

    labeled_mask = pick_labeled_indices(
        y_id,
        ratio=ratio,
        seed=seed,
        stratified=stratified,
    )
    save_exposed_ids(
        save_dir=save_dir,
        sample_index=sample_index,
        labeled_mask=labeled_mask,
        seed=seed,
        labeled_ratio=ratio,
        stratified=stratified,
        label_path=label_path,
    )
    return labeled_mask

def split_exposed_train_val_masks(
    y_id: np.ndarray,
    exposed_mask: np.ndarray,
    *,
    val_ratio: float = 0.20,
    seed: int = 0,
    stratified: bool = True,
    min_val_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split exposed samples into disjoint training and validation masks.

    Both outputs are subsets of exposed_mask. Validation samples are
    reserved for checkpoint selection. Evaluation of unexposed samples
    excludes the entire exposed_mask.
    """
    y_id = np.asarray(y_id, dtype=np.int64).reshape(-1)
    exposed_mask = np.asarray(exposed_mask, dtype=bool).reshape(-1)

    if len(y_id) != len(exposed_mask):
        raise RuntimeError(
            f"[VALSPLIT] y_id len {len(y_id)} != exposed_mask len {len(exposed_mask)}"
        )

    val_ratio = float(val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"[VALSPLIT] val_ratio must be in (0,1), got {val_ratio}")

    rng = np.random.RandomState(int(seed))

    train_mask = np.zeros_like(exposed_mask, dtype=bool)
    val_mask = np.zeros_like(exposed_mask, dtype=bool)

    exposed_idx = np.where(exposed_mask)[0]
    if exposed_idx.size == 0:
        raise RuntimeError("[VALSPLIT] exposed_mask has 0 samples.")

    if not stratified:
        perm = rng.permutation(exposed_idx)
        n_val = int(round(len(perm) * val_ratio))
        n_val = max(1, n_val)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_mask[val_idx] = True
        train_mask[train_idx] = True
    else:
        classes = np.unique(y_id[exposed_idx])
        for c in classes:
            idx_c = exposed_idx[y_id[exposed_idx] == c]
            idx_c = rng.permutation(idx_c)

            if len(idx_c) <= 1:
                # Assign singleton classes to the training split.
                train_mask[idx_c] = True
                continue

            n_val_c = int(round(len(idx_c) * val_ratio))
            n_val_c = max(int(min_val_per_class), n_val_c)
            n_val_c = min(n_val_c, len(idx_c) - 1)

            val_idx_c = idx_c[:n_val_c]
            train_idx_c = idx_c[n_val_c:]

            val_mask[val_idx_c] = True
            train_mask[train_idx_c] = True

    if bool(np.any(train_mask & val_mask)):
        raise RuntimeError("[VALSPLIT] train_mask and val_mask overlap.")

    if not bool(np.all((train_mask | val_mask) <= exposed_mask)):
        raise RuntimeError("[VALSPLIT] train/val mask must be subset of exposed_mask.")

    print(
        f"[VALSPLIT] exposed={int(exposed_mask.sum())} "
        f"train_labeled={int(train_mask.sum())} "
        f"val_labeled={int(val_mask.sum())} "
        f"val_ratio={val_ratio} stratified={stratified}"
    )

    return train_mask, val_mask


def save_labeled_train_val_split(
    save_dir: str,
    sample_index: pd.Index,
    exposed_mask: np.ndarray,
    train_labeled_mask: np.ndarray,
    val_labeled_mask: np.ndarray,
    *,
    seed: int,
    val_ratio: float,
):
    """Save the training and validation masks within the exposed subset."""
    os.makedirs(save_dir, exist_ok=True)

    exposed_mask = np.asarray(exposed_mask, dtype=bool)
    train_labeled_mask = np.asarray(train_labeled_mask, dtype=bool)
    val_labeled_mask = np.asarray(val_labeled_mask, dtype=bool)

    n_total = len(sample_index)
    for name, m in [
        ("exposed_mask", exposed_mask),
        ("train_labeled_mask", train_labeled_mask),
        ("val_labeled_mask", val_labeled_mask),
    ]:
        if m.ndim != 1 or len(m) != n_total:
            raise RuntimeError(f"[VALSPLIT] {name} shape={m.shape}, expect ({n_total},)")

    if bool(np.any(train_labeled_mask & val_labeled_mask)):
        raise RuntimeError("[VALSPLIT] train_labeled_mask overlaps val_labeled_mask.")

    if not bool(np.all((train_labeled_mask | val_labeled_mask) <= exposed_mask)):
        raise RuntimeError("[VALSPLIT] train/val masks must be subsets of exposed_mask.")

    train_rows = np.where(train_labeled_mask)[0].astype(np.int64)
    val_rows = np.where(val_labeled_mask)[0].astype(np.int64)
    exposed_rows = np.where(exposed_mask)[0].astype(np.int64)

    np.save(os.path.join(save_dir, "train_labeled_mask.npy"), train_labeled_mask.astype(np.bool_))
    np.save(os.path.join(save_dir, "val_labeled_mask.npy"), val_labeled_mask.astype(np.bool_))
    np.save(os.path.join(save_dir, "train_labeled_rows.npy"), train_rows)
    np.save(os.path.join(save_dir, "val_labeled_rows.npy"), val_rows)

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id_type": "row_index",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "n_total": int(n_total),
        "n_exposed": int(exposed_mask.sum()),
        "n_train_labeled": int(train_labeled_mask.sum()),
        "n_val_labeled": int(val_labeled_mask.sum()),
        "exposed_rows": exposed_rows.tolist(),
        "train_labeled_rows": train_rows.tolist(),
        "val_labeled_rows": val_rows.tolist(),
    }

    with open(os.path.join(save_dir, "labeled_train_val_split.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[VALSPLIT] saved train/val labeled split to {save_dir} | "
        f"train={len(train_rows)} val={len(val_rows)} exposed={len(exposed_rows)}"
    )


def load_fixed_labeled_train_val_split_if_exists(
    save_dir: str,
    n_samples: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load saved training and validation masks when both files exist."""
    train_path = os.path.join(save_dir, "train_labeled_mask.npy")
    val_path = os.path.join(save_dir, "val_labeled_mask.npy")

    if not (os.path.isfile(train_path) and os.path.isfile(val_path)):
        return None, None

    try:
        train_mask = np.asarray(np.load(train_path), dtype=bool)
        val_mask = np.asarray(np.load(val_path), dtype=bool)

        if train_mask.ndim != 1 or len(train_mask) != int(n_samples):
            print(f"[VALSPLIT][WARN] train_labeled_mask shape mismatch: {train_mask.shape}")
            return None, None

        if val_mask.ndim != 1 or len(val_mask) != int(n_samples):
            print(f"[VALSPLIT][WARN] val_labeled_mask shape mismatch: {val_mask.shape}")
            return None, None

        if bool(np.any(train_mask & val_mask)):
            print("[VALSPLIT][WARN] train/val masks overlap; ignore existing split.")
            return None, None

        print(
            f"[VALSPLIT] loaded fixed train/val labeled split from {save_dir}: "
            f"train={int(train_mask.sum())} val={int(val_mask.sum())}"
        )
        return train_mask, val_mask

    except Exception as e:
        print("[VALSPLIT][WARN] failed to load train/val split:", repr(e))
        return None, None

def load_labels_1col_numeric(label_path: str) -> np.ndarray:
    df = pd.read_csv(label_path)
    if df.shape[1] == 0:
        raise RuntimeError(f"[LABEL] no columns in {label_path}")
    col = None
    for cand in ["Label", "label", "LABEL"]:
        if cand in df.columns:
            col = cand
            break
    y = df[col].to_numpy() if col is not None else df.iloc[:, 0].to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    if np.isnan(y).any():
        bad = np.where(np.isnan(y))[0][:10]
        raise RuntimeError(f"[LABEL] NaN found in labels at rows {bad.tolist()} (check label csv)")
    return y.astype(np.int64).reshape(-1)


def remap_labels_to_0k(y: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[int, int]]]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    uniq = np.unique(y)
    if uniq.min() == 0 and np.array_equal(uniq, np.arange(len(uniq), dtype=np.int64)):
        return y, None
    mapping = {int(old): int(new) for new, old in enumerate(uniq.tolist())}
    y2 = np.vectorize(lambda v: mapping[int(v)])(y).astype(np.int64)
    return y2, mapping


# Dataset
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


# Augment / Losses
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


# Inference / Clustering
def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise RuntimeError(f"[CLUSTER] shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return float(w[row_ind, col_ind].sum()) / float(y_pred.size)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if idx.size == 0:
            continue
        _, counts = np.unique(y_true[idx], return_counts=True)
        total += int(counts.max())
    return float(total) / float(y_true.size)


def preprocess_for_clustering(emb: np.ndarray, method: str = "raw", pca_dim: int = 50) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    method = str(method).lower().strip()
    if method == "raw":
        return emb
    if method == "l2":
        return normalize(emb, norm="l2")
    if method in ["pca", "pca_l2"]:
        n_comp = int(min(max(1, pca_dim), emb.shape[0], emb.shape[1]))
        x = PCA(n_components=n_comp, random_state=0).fit_transform(emb)
        if method == "pca_l2":
            x = normalize(x, norm="l2")
        return x.astype(np.float32)
    raise ValueError(f"Unknown cluster_prep={method}")


def cluster_and_eval(
    emb: np.ndarray,
    y_true: np.ndarray,
    *,
    save_dir: str,
    prefix: str,
    k_fixed: int = 5,
    cluster_prep: str = "raw",
    pca_dim: int = 50
):
    os.makedirs(save_dir, exist_ok=True)
    emb = np.asarray(emb)
    y_true = np.asarray(y_true)
    if emb.ndim != 2:
        raise RuntimeError(f"[CLUSTER] {prefix} expects 2D embedding, got shape={emb.shape}")
    if len(emb) != len(y_true):
        raise RuntimeError(f"[CLUSTER] {prefix} emb len {len(emb)} != y_true len {len(y_true)}")
    if len(emb) == 0:
        raise RuntimeError(f"[CLUSTER] {prefix} has 0 samples after filtering; skip or fix exposed split.")
    emb_in = preprocess_for_clustering(emb, method=cluster_prep, pca_dim=pca_dim)
    if len(emb_in) == 0:
        raise RuntimeError(f"[CLUSTER] {prefix} has 0 samples after preprocess.")
    km = KMeans(n_clusters=int(k_fixed), random_state=0, n_init=50)
    pred = km.fit_predict(emb_in)
    nmi = normalized_mutual_info_score(y_true, pred)
    ari = adjusted_rand_score(y_true, pred)
    acc = cluster_acc(y_true, pred)
    pur = purity_score(y_true, pred)
    print(f"[CLUSTER] {prefix} | N={len(y_true)} | K={k_fixed} | prep={cluster_prep}")
    print(f"[CLUSTER] ACC={acc:.6f} NMI={nmi:.6f} ARI={ari:.6f} PUR={pur:.6f}")
    pd.DataFrame({"y_true": y_true.astype(int), "cluster_pred": pred.astype(int)}).to_csv(
        os.path.join(save_dir, f"{prefix}_cluster_pred.csv"), index=False
    )
    with open(os.path.join(save_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ACC": float(acc),
                "NMI": float(nmi),
                "ARI": float(ari),
                "Purity": float(pur),
                "N": int(len(y_true)),
                "K": int(k_fixed),
                "cluster_prep": str(cluster_prep)
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return {
        "ACC": float(acc),
        "NMI": float(nmi),
        "ARI": float(ari),
        "PUR": float(pur),
        "N": int(len(y_true)),
        "K": int(k_fixed),
        "cluster_prep": str(cluster_prep),
    }