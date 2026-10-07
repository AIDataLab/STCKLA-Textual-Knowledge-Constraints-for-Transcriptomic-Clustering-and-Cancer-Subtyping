"""train.py definitions moved here without algorithm changes."""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from .encoder import (
    encode_and_project_light,
)


def compute_embeddings_for_monitor(
    pretrainmodel: nn.Module,
    pretrainconfig: Dict[str, Any],
    adapter: nn.Module,
    proj_head: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    max_samples: int,
    args,
    qwen_prior_pack: Optional[Dict[str, Any]] = None,
    sample_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    pretrainmodel_was_training = pretrainmodel.training
    adapter_was_training = adapter.training
    proj_was_training = proj_head.training

    pretrainmodel.eval()
    adapter.eval()
    proj_head.eval()

    embs = []
    n = min(int(max_samples), int(x_np.shape[0]))
    if sample_indices is None:
        sample_indices_arr = np.arange(int(x_np.shape[0]), dtype=np.int64)
    else:
        sample_indices_arr = np.asarray(sample_indices, dtype=np.int64)
        if sample_indices_arr.shape[0] != int(x_np.shape[0]):
            raise RuntimeError(
                f"[MON] sample_indices length={sample_indices_arr.shape[0]} != x_np rows={x_np.shape[0]}"
            )

    with torch.no_grad():
        for i in range(n):
            gene_x = torch.tensor(x_np[i], device=device).unsqueeze(0)
            sample_idx_t = torch.tensor([int(sample_indices_arr[i])], device=device, dtype=torch.long)
            out = encode_and_project_light(
                pretrainmodel=pretrainmodel,
                pretrainconfig=pretrainconfig,
                adapter=adapter,
                proj_head=proj_head,
                cls_head=None,
                batch_x=gene_x,
                device=device,
                args=args,
                qwen_prior_pack=qwen_prior_pack,
                sample_indices=sample_idx_t,
            )
            embs.append(out["z"].detach().float().cpu().numpy())

    emb = np.squeeze(np.array(embs))
    if emb.ndim != 2:
        raise RuntimeError(f"[MON] embedding must be 2D, got {emb.shape}")

    if pretrainmodel_was_training:
        pretrainmodel.train(True)
    if adapter_was_training:
        adapter.train(True)
    if proj_was_training:
        proj_head.train(True)

    return emb.astype(np.float32)


def kmeans_monitor(emb: np.ndarray, k: int = 32, seed: int = 0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20, max_iter=300)
    pred = km.fit_predict(emb)
    inertia = float(km.inertia_)
    return {
        "kmeans_inertia": inertia,
        "n": float(emb.shape[0]),
        "d": float(emb.shape[1]),
        "k": float(k),
        "uniq_clusters": float(len(np.unique(pred)))
    }


def _prepare_embedding_for_cluster_metrics(
    emb: np.ndarray,
    *,
    seed: int,
    cluster_prep: str = "raw",
    pca_dim: int = 50,
) -> np.ndarray:
    """Apply checkpoint KMeans preprocessing to stability and quality metrics."""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise RuntimeError(f"[STABILITY] emb must be 2D, got shape={emb.shape}")

    x = emb
    prep = str(cluster_prep).lower().strip()

    if prep == "raw":
        pass
    elif prep == "l2":
        from sklearn.preprocessing import normalize
        x = normalize(x)
    elif prep == "pca":
        from sklearn.decomposition import PCA
        d = min(int(pca_dim), x.shape[1], x.shape[0] - 1)
        if d >= 2:
            x = PCA(n_components=d, random_state=int(seed)).fit_transform(x)
    elif prep == "pca_l2":
        from sklearn.preprocessing import normalize
        from sklearn.decomposition import PCA
        d = min(int(pca_dim), x.shape[1], x.shape[0] - 1)
        if d >= 2:
            x = PCA(n_components=d, random_state=int(seed)).fit_transform(x)
        x = normalize(x)
    else:
        raise ValueError(f"[STABILITY] unknown cluster_prep={cluster_prep}")

    return np.asarray(x, dtype=np.float32)


def cluster_predict_for_checkpoint_stability(
    emb: np.ndarray,
    *,
    k: int,
    seed: int,
    cluster_prep: str = "raw",
    pca_dim: int = 50,
) -> np.ndarray:

    x = _prepare_embedding_for_cluster_metrics(
        emb,
        seed=int(seed),
        cluster_prep=str(cluster_prep),
        pca_dim=int(pca_dim),
    )

    from sklearn.cluster import KMeans
    km = KMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=20,
        max_iter=300,
    )
    pred = km.fit_predict(x)
    return np.asarray(pred, dtype=np.int64)


def compute_label_free_cluster_quality_for_checkpoint(
    emb: np.ndarray,
    pred: np.ndarray,
    *,
    k: int,
    seed: int,
    cluster_prep: str,
    pca_dim: int,
    args,
) -> Dict[str, float]:
    """Compute cluster-quality metrics from embeddings and KMeans assignments."""
    emb = np.asarray(emb, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)

    if emb.ndim != 2 or pred.shape[0] != emb.shape[0] or emb.shape[0] <= 2:
        return {
            "quality_valid": 0.0,
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "balance_entropy": 0.0,
            "min_cluster_frac": 0.0,
            "max_cluster_frac": 1.0,
            "cluster_size_cv": float("inf"),
            "collapse_penalty": 1.0,
            "uniq_clusters": float(len(np.unique(pred)) if pred.size else 0),
        }

    n = int(emb.shape[0])
    k = int(k)
    uniq, counts = np.unique(pred, return_counts=True)
    probs = counts.astype(np.float64) / max(1, counts.sum())
    min_frac = float(probs.min()) if probs.size else 0.0
    max_frac = float(probs.max()) if probs.size else 1.0
    mean_size = float(np.mean(counts)) if counts.size else 0.0
    size_cv = float(np.std(counts) / max(mean_size, 1e-8)) if counts.size else float("inf")
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / max(np.log(max(2, len(probs))), 1e-12)) if probs.size > 1 else 0.0

    min_allowed = float(getattr(args, "exposed_cluster_min_cluster_frac", 0.005))
    max_allowed = float(getattr(args, "exposed_cluster_max_cluster_frac", 0.80))
    collapse_penalty = 0.0
    if min_frac < min_allowed:
        collapse_penalty += float((min_allowed - min_frac) / max(min_allowed, 1e-8))
    if max_frac > max_allowed:
        collapse_penalty += float((max_frac - max_allowed) / max(1.0 - max_allowed, 1e-8))
    if len(uniq) < k:
        collapse_penalty += float((k - len(uniq)) / max(1, k))

    sil = float("nan")
    ch = float("nan")
    db = float("nan")

    if len(uniq) >= 2 and len(uniq) < n:
        x = _prepare_embedding_for_cluster_metrics(
            emb,
            seed=int(seed),
            cluster_prep=str(cluster_prep),
            pca_dim=int(pca_dim),
        )
        # Quality metrics can be expensive for large scRNA datasets.
        # Subsample deterministically; keep only if at least two clusters remain.
        max_q = int(getattr(args, "exposed_cluster_quality_max_samples", 3000))
        if max_q > 0 and x.shape[0] > max_q:
            rng = np.random.RandomState(int(seed) + 303917)
            qidx = np.sort(rng.choice(x.shape[0], size=max_q, replace=False)).astype(np.int64)
            xq = x[qidx]
            pq = pred[qidx]
        else:
            xq = x
            pq = pred

        uq = np.unique(pq)
        if len(uq) >= 2 and len(uq) < xq.shape[0]:
            try:
                from sklearn.metrics import silhouette_score
                sil = float(silhouette_score(xq, pq, metric="euclidean"))
            except Exception:
                sil = float("nan")
            try:
                from sklearn.metrics import calinski_harabasz_score
                ch = float(calinski_harabasz_score(xq, pq))
            except Exception:
                ch = float("nan")
            try:
                from sklearn.metrics import davies_bouldin_score
                db = float(davies_bouldin_score(xq, pq))
            except Exception:
                db = float("nan")

    return {
        "quality_valid": 1.0,
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
        "balance_entropy": float(entropy),
        "min_cluster_frac": float(min_frac),
        "max_cluster_frac": float(max_frac),
        "cluster_size_cv": float(size_cv),
        "collapse_penalty": float(collapse_penalty),
        "uniq_clusters": float(len(uniq)),
    }
