"""
This module implements the K-SVD dictionary learning algorithm.
"""
import numpy as np
from numpy.typing import NDArray

from sklearn.linear_model import orthogonal_mp


class KSVD:
    """
    K-SVD dictionary learning algorithm.

    Attributes:
        k (int): Number of atoms in the dictionary.
        num_features (int): Number of features in the input data.
        num_coefs (int): Number of coefficients in the sparse representation.
        dictionary (NDArray): The learned dictionary. None if the fit method has not been run.
    """

    def __init__(
        self,
        k: int = 441,
        num_features: int = 64,
        num_coefs: int = 10,
        max_iter: int = 100,
        tol: float = 1e-3,
    ) -> None:
        """
        k (int): Number of atoms in the dictionary. Default: 441.
        num_features (int): Number of features in the input data. Default: 64.
        num_coefs (int): Number of coefficients in the sparse representation. Default: 10.
        max_iter (int): Maximum number of iterations. Default: 100.
        tol (float): Tolerance for the stopping criterion. Default: 1e-3.
        """
        self.k = k
        self.num_features = num_features
        self.num_coefs = num_coefs
        self.max_iter = max_iter
        self.tol = tol

        self.dictionary = None

    def fit(self, X: NDArray) -> None:
        """
        Learn the dictionary from the input data.

        X (NDArray): Input data. Shape: (num_samples, num_features).
        """
        assert X.shape[0] >= self.k, (
            "The number of samples in the training data is less than the number "
            "of atoms in the dictionary."
        )
        assert X.shape[1] == self.num_features, (
            "The number of features in the input data does not match the number "
            "of features in the dictionary."
        )

        # Initialize the dictionary with random normalized samples from the input data.
        self.dictionary = X[np.random.choice(X.shape[0], self.k, replace=False)]
        self.dictionary /= np.linalg.norm(self.dictionary, axis=1, keepdims=True)

        # Run the K-SVD algorithm.
        for _ in range(self.max_iter):
            X_hat = self._fit_step(X)
            if self._check_convergence(X, X_hat):
                break

    def _check_convergence(self, X: NDArray, X_hat: NDArray) -> bool:
        """
        Check if the algorithm has converged.

        X (NDArray): Input data. Shape: (num_samples, num_features).
        X_hat (NDArray): Reconstructed data. Shape: (num_samples, num_features).

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        return np.amax(np.linalg.norm(X - X_hat, axis=1)) < self.tol

    def _fit_step(self, X: NDArray) -> NDArray:
        """
        Perform a single iteration of the K-SVD algorithm.

        Returns:
            NDArray: Reconstructed data. Shape: (num_samples, num_features).
        """
        # Sparse Coding Stage
        # Compute the sparse representation of the input data using OMP with
        # the current dictionary.
        X_reconstructed, coefs = self.transform(X, return_coefs=True)

        # TODO: Codebook Update Stage

        return X_reconstructed

    def transform(self, X: NDArray, return_coefs: bool = False) -> NDArray:
        """
        Transform the input data using the OMP algorithm and the learned dictionary.

        X (NDArray): Input data. Shape: (num_samples, num_features).
        return_coefs (bool): If True, return the coefficients of the sparse representation.
            Default: False.

        Returns:
            X_reconstructed (NDArray): Sparse representation of the input data.
                Shape: (num_samples, num_coefs).
            coefs (NDArray): Coefficients of the sparse representation.
                Shape: (num_samples, num_coefs). Present only if return_coefs is True.
        """
        assert self.dictionary is not None, "The fit method has not been run."
        assert X.shape[1] == self.num_features, (
            "The number of features in the input data does not match the number "
            "of features in the dictionary."
        )

        coefs = orthogonal_mp(self.dictionary.T, X.T, n_nonzero_coefs=self.num_coefs).T
        X_reconstructed = coefs @ self.dictionary

        if return_coefs:
            return X_reconstructed, coefs
        return X_reconstructed
