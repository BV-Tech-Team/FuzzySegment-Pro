"""
Fuzzy C-Means vs K-Means Comparison
Demonstrates the advantage of fuzzy clustering for multi-dimensional customer profiling.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
sys.path.append('..')
from src.fuzzy_clustering import FuzzyCMeansWrapper
from src.metrics import evaluate_fuzzy_clustering

# Load preprocessed data
df = pd.read_csv('../data/preprocessed_customers.csv')
X = df.drop(columns=['Customer ID']).values
customer_ids = df['Customer ID'].values
print(f"Loaded {len(X)} customers with {X.shape[1]} features")

# Hyperparameters
N_CLUSTERS = 3
M = 2.0  # fuzzifier

print(f"\n{'='*80}\nRunning Fuzzy C-Means (n_clusters={N_CLUSTERS}, m={M})\n{'='*80}")
fcm = FuzzyCMeansWrapper(n_clusters=N_CLUSTERS, m=M)
fcm.fit(X)

# Fuzzy metrics
fuzzy_metrics = evaluate_fuzzy_clustering(X, fcm.u, fcm.centers)
print("\nFuzzy C-Means Metrics:")
for metric, value in fuzzy_metrics.items():
    print(f"  {metric}: {value:.4f}")

print(f"\n{'='*80}\nRunning K-Means (n_clusters={N_CLUSTERS})\n{'='*80}")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)
print(f"\nK-Means Silhouette Score: {kmeans_silhouette:.4f}")

# Compare sample customers
print(f"\n{'='*80}\nSample Customers: Fuzzy Memberships vs K-Means Hard Labels\n{'='*80}")
sample_indices = np.random.choice(len(X), size=5, replace=False)
for idx in sample_indices:
    cust_id = customer_ids[idx]
    fuzzy_memberships = fcm.u[:, idx]
    kmeans_label = kmeans_labels[idx]
    print(f"\nCustomer {cust_id}:")
    print(f"  Fuzzy Memberships: " + " | ".join([f"C{i}: {m:.2%}" for i, m in enumerate(fuzzy_memberships)]))
    print(f"  K-Means Label: Cluster {kmeans_label} (100%)")
    
    # Highlight lost information
    sorted_memberships = sorted(enumerate(fuzzy_memberships), key=lambda x: x[1], reverse=True)
    if sorted_memberships[0][1] < 0.7:  # if top membership < 70%, multi-dimensional
        print(f"  ⚠️  Multi-dimensional customer! K-Means misses {1 - sorted_memberships[0][1]:.1%} of behavior")

# Visualization: Membership heatmap
print(f"\n{'='*80}\nGenerating Visualizations\n{'='*80}")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Fuzzy membership heatmap (first 30 customers)
n_show = min(30, len(X))
im = axes[0].imshow(fcm.u[:, :n_show], aspect='auto', cmap='YlOrRd', interpolation='nearest')
axes[0].set_title('Fuzzy C-Means: Membership Degrees')
axes[0].set_xlabel('Customers')
axes[0].set_ylabel('Cluster')
axes[0].set_yticks(range(N_CLUSTERS))
axes[0].set_yticklabels([f'Cluster {i}' for i in range(N_CLUSTERS)])
plt.colorbar(im, ax=axes[0], label='Membership')

# Plot 2: K-Means hard assignments (first 30 customers)
kmeans_matrix = np.zeros((N_CLUSTERS, n_show))
for i in range(n_show):
    kmeans_matrix[kmeans_labels[i], i] = 1
im2 = axes[1].imshow(kmeans_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
axes[1].set_title('K-Means: Hard Assignments')
axes[1].set_xlabel('Customers')
axes[1].set_ylabel('Cluster')
axes[1].set_yticks(range(N_CLUSTERS))
axes[1].set_yticklabels([f'Cluster {i}' for i in range(N_CLUSTERS)])
plt.colorbar(im2, ax=axes[1], label='Assignment')

plt.tight_layout()
plt.savefig('../data/fcm_vs_kmeans_comparison.png', dpi=150)
plt.show()
print("\nSaved: data/fcm_vs_kmeans_comparison.png")

# Distribution of maximum memberships (fuzzy) vs hard (k-means)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

max_memberships = np.max(fcm.u, axis=0)
axes[0].hist(max_memberships, bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0].set_title('Fuzzy C-Means: Max Membership Distribution')
axes[0].set_xlabel('Max Membership Degree')
axes[0].set_ylabel('Frequency')
axes[0].axvline(0.7, color='red', linestyle='--', label='70% threshold')
axes[0].legend()

axes[1].hist([1.0] * len(kmeans_labels), bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[1].set_title('K-Means: All Memberships = 1.0 (Hard Assignment)')
axes[1].set_xlabel('Membership Degree')
axes[1].set_ylabel('Frequency')
axes[1].set_xlim(0, 1.1)

plt.tight_layout()
plt.savefig('../data/membership_distribution.png', dpi=150)
plt.show()
print("\nSaved: data/membership_distribution.png")

print(f"\n{'='*80}\nComparison Complete!\n{'='*80}")
print("\nKey Findings:")
print(f"  - Fuzzy C-Means FSI (Fuzzy Silhouette): {fuzzy_metrics['FSI']:.4f}")
print(f"  - K-Means Silhouette: {kmeans_silhouette:.4f}")
print(f"  - Fuzzy PC (Partition Coefficient): {fuzzy_metrics['PC']:.4f}")
print(f"  - Fuzzy XBI (Xie-Beni Index, lower=better): {fuzzy_metrics['XBI']:.4f}")
multi_dim_count = np.sum(max_memberships < 0.7)
print(f"  - Multi-dimensional customers (max membership < 70%): {multi_dim_count} ({multi_dim_count/len(X):.1%})")
print("\n✓ Fuzzy C-Means captures the full spectrum of customer behavior!")
