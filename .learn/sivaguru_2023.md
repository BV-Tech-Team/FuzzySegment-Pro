# Sivaguru (2023)

Title: Dynamic Customer Segmentation: A Case Study Using the Modified Dynamic Fuzzy C-Means Clustering Algorithm

Publication: Granular Computing (2023), Vol. 8, pp. 345-360. DOI: 10.1007/s41066-022-00335-0

Summary:

- Integrates Fuzzy C-Means with Granular Computing principles to produce hierarchical, multi-resolution customer segments.
- Evaluates clustering using fuzzy-specific indices: Xie-Beni (XBI), Partition Coefficient (PC), Modified Partition Coefficient (MPC), Partition Entropy (PE), and Fuzzy Silhouette Index (FSI).
- Demonstrates case study on supermarket retail data with actionable segment definitions.

Key takeaways for FuzzySegment Pro:

- Adopt fuzzy evaluation metrics beyond silhouette, e.g., XBI, PC/MPC, PE, and FSI to validate soft clusters.
- Use granular (multi-level) aggregation to present hierarchical segment summaries in the UI.
- Provide interactive thresholds to view coarse → fine segments.

Suggested actions:

- Add implementation of Xie-Beni and PC/MPC metrics in `src/` and example notebooks in `notebooks/`.
- Add a `.learn` lesson about Granular Computing and hierarchical analysis.
