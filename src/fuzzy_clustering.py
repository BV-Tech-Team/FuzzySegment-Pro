import numpy as np
import skfuzzy as fuzz


class FuzzyCMeansWrapper:
    def __init__(self, n_clusters=3, m=2.0, error=1e-5, maxiter=1000):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.centers = None
        self.u = None
        self.X_fitted = None

    def fit(self, X):
        """
        Fit Fuzzy C-Means on data matrix X.
        X: (n_samples, n_features)
        """
        self.X_fitted = X
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter, init=None)
        self.centers = cntr
        self.u = u
        self.fpc = fpc
        return self

    def predict_membership(self, X):
        """
        Predict membership matrix for new data X.
        Returns: membership matrix (n_clusters, n_samples)
        """
        if self.centers is None:
            raise RuntimeError("Model not fitted")
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            X.T, self.centers, self.m, error=self.error, maxiter=self.maxiter)
        return u

    def get_hard_labels(self):
        """
        Convert fuzzy memberships to hard cluster assignments.
        Returns: array of shape (n_samples,)
        """
        if self.u is None:
            raise RuntimeError("Model not fitted")
        return np.argmax(self.u, axis=0)
