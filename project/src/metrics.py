"""
Fuzzy Clustering Evaluation Metrics
====================================
Implements fuzzy-specific validation indices for cluster quality assessment.

These metrics are specifically designed for fuzzy clustering and provide
better insights than traditional metrics like silhouette score alone.

References:
-----------
- Sivaguru, M. (2023). Granular Computing, Vol. 8, pp. 345-360.
- Xie, X. L., & Beni, G. (1991). IEEE Transactions on Pattern Analysis.

Author: FuzzySegment Pro Team
Date: December 2025
"""

import numpy as np
from sklearn.metrics import silhouette_score


def partition_coefficient(u):
    """
    Partition Coefficient (PC): measures 'fuzziness' of clustering.
    PC ∈ [1/c, 1], higher is better (less fuzzy, more crisp).
    """
    n = u.shape[1]
    return np.sum(u ** 2) / n


def modified_partition_coefficient(u):
    """
    Modified Partition Coefficient (MPC): normalized version of PC.
    MPC ∈ [0, 1], higher is better.
    """
    c = u.shape[0]
    pc = partition_coefficient(u)
    return (c * pc - 1) / (c - 1)


def partition_entropy(u):
    """
    Partition Entropy (PE): measures disorder in cluster assignments.
    PE ∈ [0, log(c)], lower is better.
    """
    n = u.shape[1]
    u_safe = np.clip(u, 1e-10, 1.0)  # avoid log(0)
    return -np.sum(u * np.log(u_safe)) / n


def xie_beni_index(X, u, centers):
    """
    Xie-Beni Index (XBI): ratio of compactness to separation.
    Lower is better.
    
    XBI = Σ(u_ik^m * ||x_k - v_i||^2) / (n * min_{i≠j} ||v_i - v_j||^2)
    """
    m = 2.0  # fuzzifier
    c, n = u.shape
    
    # Numerator: weighted sum of squared distances
    numerator = 0
    for i in range(c):
        for k in range(n):
            numerator += (u[i, k] ** m) * np.linalg.norm(X[k] - centers[i]) ** 2
    
    # Denominator: minimum squared distance between centers
    min_dist_sq = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            dist_sq = np.linalg.norm(centers[i] - centers[j]) ** 2
            min_dist_sq = min(min_dist_sq, dist_sq)
    
    return numerator / (n * min_dist_sq) if min_dist_sq > 0 else np.inf


def fuzzy_silhouette_index(X, u):
    """
    Fuzzy Silhouette Index (FSI): fuzzy extension of silhouette score.
    Assigns each sample to cluster with highest membership, then computes standard silhouette.
    FSI ∈ [-1, 1], higher is better.
    """
    hard_labels = np.argmax(u, axis=0)
    if len(np.unique(hard_labels)) < 2:
        return 0.0
    return silhouette_score(X, hard_labels)


def evaluate_fuzzy_clustering(X, u, centers):
    """
    Compute all fuzzy validation metrics.
    Returns dict of metric name -> value.
    """
    return {
        'PC': partition_coefficient(u),
        'MPC': modified_partition_coefficient(u),
        'PE': partition_entropy(u),
        'XBI': xie_beni_index(X, u, centers),
        'FSI': fuzzy_silhouette_index(X, u)
    }
