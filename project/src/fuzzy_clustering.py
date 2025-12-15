"""
Fuzzy C-Means Clustering Module
================================
Wrapper for scikit-fuzzy library to provide sklearn-like interface.

Author: FuzzySegment Pro Team
Date: December 2025
"""

import numpy as np
import skfuzzy as fuzz


class FuzzyCMeansWrapper:
    """
    Fuzzy C-Means clustering wrapper with sklearn-like API.
    
    Unlike K-Means which assigns each point to exactly one cluster,
    Fuzzy C-Means assigns membership degrees to all clusters,
    capturing overlapping patterns in customer behavior.
    
    Parameters:
    -----------
    n_clusters : int, default=3
        Number of clusters to form
    m : float, default=2.0
        Fuzzifier parameter (controls overlap between clusters)
        Higher values = more fuzzy (more overlap)
        m=1 approaches hard clustering (K-Means)
    error : float, default=1e-5
        Convergence threshold for stopping criterion
    maxiter : int, default=1000
        Maximum number of iterations
    
    Attributes:
    -----------
    centers : ndarray of shape (n_clusters, n_features)
        Cluster centers after fitting
    u : ndarray of shape (n_clusters, n_samples)
        Membership matrix - u[i,j] is membership of sample j in cluster i
    fpc : float
        Fuzzy Partition Coefficient (FPC) - measures clustering quality
    """
    
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
        Fit Fuzzy C-Means clustering on data matrix X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data matrix
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        self.X_fitted = X
        # scikit-fuzzy expects (n_features, n_samples), so transpose
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, self.n_clusters, self.m, error=self.error, maxiter=self.maxiter, init=None)
        self.centers = cntr  # Cluster centers
        self.u = u  # Membership matrix
        self.fpc = fpc  # Fuzzy Partition Coefficient
        return self

    def predict_membership(self, X):
        """
        Predict fuzzy membership degrees for new samples.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        u : ndarray of shape (n_clusters, n_samples)
            Membership matrix where u[i,j] represents the degree to which
            sample j belongs to cluster i. All columns sum to 1.
        """
        if self.centers is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            X.T, self.centers, self.m, error=self.error, maxiter=self.maxiter)
        return u

    def get_hard_labels(self):
        """
        Convert fuzzy memberships to hard cluster assignments.
        Assigns each sample to the cluster with highest membership.
        
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels (0 to n_clusters-1)
        """
        if self.u is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.argmax(self.u, axis=0)
