"""base.py definitions moved here without algorithm changes."""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment


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
