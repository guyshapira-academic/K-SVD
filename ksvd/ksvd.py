"""
This module implements the K-SVD dictionary learning algorithm.
"""
import numpy as np
from numpy.typing import NDArray


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
            X_hat = self._fit_step()
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

    def _fit_step(self) -> NDArray:
        """
        Perform a single iteration of the K-SVD algorithm.

        Returns:
            NDArray: Reconstructed data. Shape: (num_samples, num_features).
        """
        pass
