# Sivaguru & Punniyamoorthy (2020)

Title: Modified Dynamic Fuzzy C-Means Clustering Algorithm – Application in Dynamic Customer Segmentation

Publication: Applied Intelligence (2020), Vol. 50, Issue 6, pp. 1922-1942. DOI: 10.1007/s10489-019-01626-x

Summary:

- Proposes a Modified Dynamic Fuzzy C-Means (CDFCM) tailored for customer segmentation where memberships evolve over time.
- Handles temporal changes in customer behavior by allowing cluster centers and membership degrees to adapt dynamically.
- Validated on retail transaction data and shows improved segmentation stability and accuracy versus static FCM.

Key takeaways for FuzzySegment Pro:

- Implement time-aware membership updates for longitudinal customer data (e.g., rolling windows).
- Use dynamic adaptation to capture behavior shifts (seasonality, campaign effects).
- Consider incremental or online fuzzy c-means variants for streaming/batch updates.

Suggested actions:

- Add a lesson on dynamic fuzzy clustering to `.learn` and prototype a rolling-window implementation in `notebooks/`.
