import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# ======================== Excercise 12.3 - Izermann ========================
def linkage(data,metric,method='single',standarize=False):
    n_samples = data.shape[0]
    if standarize:
        data = (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 1e-8)

    Dissimilarity = dissimilarity_matrix(data, metric)

    Z = []

    clusters = [[i] for i in range(n_samples)]
    labels = list(range(n_samples))

    for i in range(n_samples - 1):
        a,b,dis = closest_pair(Dissimilarity, clusters, method)
        new = clusters[a] + clusters[b]
        la,lb = labels[a], labels[b]

        clusters.pop(b)
        clusters.pop(a)
        labels.pop(b)
        labels.pop(a)

        clusters.append(new)
        labels.append(n_samples + i)
        Z.append([la,lb,dis,len(new)])
    return np.array(Z)

# ======================== Helper functions ========================
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def dissimilarity_matrix(X, metric) -> np.ndarray:
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            D[i, j] = metric(X[i], X[j])
            D[j, i] = D[i, j]
    return D

def cluster_distance(D, A, B, method) -> float:
    if method == 'single':
        return np.min(D[np.ix_(A, B)])
    elif method == 'complete':
        return np.max(D[np.ix_(A, B)])
    elif method == 'average':
        return np.mean(D[np.ix_(A, B)])
    else:
        raise ValueError("Invalid method")

def closest_pair (D, clusters, method) -> tuple[int, int, float]:
    """
    a < b
    """

    k = len(clusters)
    min_dist = np.inf
    a,b = -1, -1
    for i in range(k):
        for j in range(i+1,k):
            dist = cluster_distance(D, clusters[i], clusters[j], method)
            if dist < min_dist:
                min_dist = dist
                a, b = i, j
    return a, b, min_dist



# Plotting functions

def fcluster(Z, k, n_samples):
    """Cut the dendrogram to get exactly k clusters.
    Replays the first (n_samples - k) merges of Z and stops.
    Returns (assignments, members):
      assignments : (n,) int array, assignments[i] = cluster id 0..k-1
      members     : dict cluster_label -> list of point indices
    """
    if not 1 <= k <= n_samples:
        raise ValueError(f"k must be in [1, {n_samples}]")
    members = {i: [i] for i in range(n_samples)}
    for i in range(n_samples - k):
        la, lb = int(Z[i, 0]), int(Z[i, 1])
        members[n_samples + i] = members[la] + members[lb]
        del members[la], members[lb]
    assignments = np.empty(n_samples, dtype=int)
    for cid, label in enumerate(sorted(members)):
        for point in members[label]:
            assignments[point] = cid
    return assignments, members


def cut_height(Z, k):
    """Height at which a horizontal cut yields exactly k clusters."""
    n = Z.shape[0] + 1
    lo = Z[n - k - 1, 2] if k < n else 0.0
    hi = Z[n - k, 2] if k > 1 else Z[-1, 2] * 1.1
    return (lo + hi) / 2.0


def plot_dendrogram(Z, k=None, ax=None, title=None, labels=None):
    """Dendrogram, optionally with the k-cluster cut drawn and colored."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    kwargs = {}
    if k is not None:
        h = cut_height(Z, k)
        kwargs["color_threshold"] = h
        ax.axhline(h, ls="--", c="k", lw=1, zorder=0)

    dendrogram(Z, ax=ax, labels=labels, above_threshold_color="0.6", **kwargs)
    ax.set_ylabel("merge height")
    ax.set_xlabel("observation")
    if title:
        ax.set_title(title)
    return ax


def plot_clusters(X, assignments, ax=None, title=None):
    """Scatter colored by cluster. 1D -> on a line, 2D -> as is, p>2 -> first 2 PCs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    X = np.asarray(X, dtype=float)

    if X.shape[1] == 1:
        xs, ys = X[:, 0], np.zeros(len(X))
        ax.set_yticks([])
        ax.set_xlabel("x")
    elif X.shape[1] == 2:
        xs, ys = X[:, 0], X[:, 1]
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    else:
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt[:2].T
        xs, ys = P[:, 0], P[:, 1]
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax.scatter(xs, ys, c=assignments, cmap="tab10", s=80,
               edgecolors="k", linewidths=0.5)
    if title:
        ax.set_title(title)
    return ax


def plot_merge_heights(Z, ax=None, title=None):
    """Merge heights vs number of clusters remaining. Big gaps suggest a good k."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    n = Z.shape[0] + 1
    ks = np.arange(n - 1, 0, -1)          # clusters remaining after each merge
    ax.plot(ks, Z[:, 2], "o-")
    ax.invert_xaxis()
    ax.set_xlabel("clusters remaining")
    ax.set_ylabel("merge height")
    if title:
        ax.set_title(title)
    return ax


def compare_methods(X, metric, k, methods=("single", "complete", "average"),
                    standarize=False, suptitle=None):
    """Dendrogram + colored scatter for each linkage method, side by side."""
    n = X.shape[0]
    fig, axes = plt.subplots(2, len(methods), figsize=(5 * len(methods), 8))
    if len(methods) == 1:
        axes = axes.reshape(2, 1)

    for j, m in enumerate(methods):
        Z = linkage(X, metric, method=m, standarize=standarize)
        assign, _ = fcluster(Z, k, n)
        plot_dendrogram(Z, k=k, ax=axes[0, j], title=f"{m} linkage")
        plot_clusters(X, assign, ax=axes[1, j], title=f"{m}, k={k}")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig