# Afrin, Al-Amin & Tabassum (2015)

Title: Comparative Performance of Using PCA with K-Means and Fuzzy C-Means Clustering for Customer Segmentation

Publication: International Journal of Advanced Research in Computer Science and Software Engineering (2015).

Summary:

- Compares K-Means and Fuzzy C-Means on customer segmentation tasks, with and without PCA-based dimensionality reduction.
- Finds Fuzzy C-Means often produces more informative segment membership distributions, especially when features capture multi-aspect behavior.
- PCA can improve clustering by removing noise and reducing dimensionality, but interpretability of components should be considered.

Key takeaways for FuzzySegment Pro:

- Add optional PCA preprocessing step in pipeline and allow users to toggle it in the Streamlit UI.
- Report variance explained and component loadings so users understand transformed features.
- Use PCA for higher-dimensional feature sets (browsing vectors, embeddings).

Suggested actions:

- Implement PCA option in `src/data_pipeline.py` and demo in a notebook.
