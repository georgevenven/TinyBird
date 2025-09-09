# cluster_npz_hdbscan.py
# pip install hdbscan scikit-learn numpy
import argparse
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import v_measure_score

import hdbscan


def load_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        if "embeddings" not in data or "labels" not in data:
            raise KeyError("NPZ must contain 'embeddings' and 'labels'.")
        X = data["embeddings"]
        y = data["labels"]
    # keep only items with a real label
    mask_labeled = np.array([lbl is not None and str(lbl) != "" for lbl in y], dtype=bool)
    X = X[mask_labeled]
    y = np.asarray([str(lbl) for lbl in y[mask_labeled]], dtype=object)
    return X, y, mask_labeled.sum()


def preprocess(X, normalize_mode, pca_dim):
    if normalize_mode == "zscore":
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X).astype(np.float32)
    elif normalize_mode == "l2":
        X = normalize(X).astype(np.float32)
    else:
        X = X.astype(np.float32)

    if pca_dim and pca_dim > 0:
        # safe cap so PCA won't error on tiny N
        n_comp = int(min(pca_dim, X.shape[1], max(1, X.shape[0] - 1)))
        if n_comp >= 1 and n_comp < X.shape[1]:
            X = PCA(n_components=n_comp).fit_transform(X).astype(np.float32)
    return X


def run_hdbscan(X, min_cluster_size, min_samples, metric, eps, jobs):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=(None if min_samples is None or min_samples < 0 else min_samples),
        metric=metric,
        cluster_selection_epsilon=eps,
        core_dist_n_jobs=jobs,
    )
    labels = clusterer.fit_predict(X)          # -1 = noise
    probs = clusterer.probabilities_
    return labels, probs


def main():
    p = argparse.ArgumentParser(description="Cluster NPZ embeddings with HDBSCAN and compute V-measure.")
    p.add_argument("--npz", required=True, help="Path to npz produced by your pipeline.")
    p.add_argument("--min_cluster_size", type=int, default=5000)
    p.add_argument("--min_samples", type=int, default=-1, help="-1 -> auto None for HDBSCAN")
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan", "l1", "l2", "cosine"])
    p.add_argument("--eps", type=float, default=0.0, help="cluster_selection_epsilon")
    p.add_argument("--normalize", default="l2", choices=["none", "l2", "zscore"])
    p.add_argument("--pca", type=int, default=32, help="0 disables PCA")
    p.add_argument("--jobs", type=int, default=-1, help="threads for core distance")
    p.add_argument("--out_csv", default=None, help="Optional CSV to save assignments.")
    args = p.parse_args()

    X, y, n_labeled = load_npz(args.npz)
    if X.shape[0] == 0:
        raise ValueError("No labeled items in NPZ. Cannot compute V-measure.")

    Xp = preprocess(X, args.normalize, args.pca)
    pred, probs = run_hdbscan(Xp, args.min_cluster_size, args.min_samples, args.metric, args.eps, args.jobs)

    # Stats
    n = len(pred)
    n_noise = int((pred == -1).sum())
    clusters = np.unique(pred[pred != -1])
    n_clusters = len(clusters)

    # V-measure
    v_including_noise = float(v_measure_score(y, pred))
    keep = pred != -1
    v_excluding_noise = float(v_measure_score(y[keep], pred[keep])) if keep.any() else float("nan")

    print(f"NPZ labeled items: {n_labeled}")
    print(f"Used for clustering: {n}")
    print(f"Clusters (excl. noise): {n_clusters}")
    print(f"Noise points: {n_noise} ({n_noise/n:.2%})")
    print(f"V-measure (including noise as a cluster): {v_including_noise:.4f}")
    print(f"V-measure (excluding noise): {v_excluding_noise:.4f}")

    if args.out_csv:
        out = Path(args.out_csv)
        with out.open("w") as f:
            f.write("idx,true_label,cluster,probability\n")
            for i, (t, c, pr) in enumerate(zip(y, pred, probs)):
                f.write(f"{i},{t},{c},{pr:.6f}\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
