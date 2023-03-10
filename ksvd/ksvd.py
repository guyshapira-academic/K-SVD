"""
This module implements the K-SVD dictionary learning algorithm.
"""
from typing import Tuple
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from sklearn.linear_model import orthogonal_mp

from tqdm.auto import trange, tqdm


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

    def fit(self, X: NDArray, verbose: int = 0) -> None:
        """
        Learn the dictionary from the input data.

        Parameters:
            X (NDArray): Input data. Shape: (num_samples, num_features).
            verbose (int): Verbosity level. Default: 0.
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

        random_samples = X[np.random.choice(X.shape[0], self.k, replace=False), :]
        self.dictionary = random_samples + 1e-6 * np.random.randn(
            self.k, self.num_features
        )
        self.dictionary /= np.linalg.norm(self.dictionary, axis=1, keepdims=True)

        # Run the K-SVD algorithm.
        it = trange(self.max_iter, disable=(verbose < 1), total=None)
        for _ in it:
            X_hat, err = self._fit_step(X, verbose=verbose)
            it.set_description(f"error: {Decimal(err):.4E}")
            if err < self.tol:
                break

    @staticmethod
    def _error(X: NDArray, X_hat: NDArray) -> float:
        """
        Compute the reconstruction error.

        Parameters:
            X (NDArray): Input data. Shape: (num_samples, num_features).
            X_hat (NDArray): Reconstructed data. Shape: (num_samples, num_features).

        Returns:
            float: Reconstruction error.
        """
        return (np.linalg.norm(X - X_hat, ord="fro") / np.linalg.norm(X, ord="fro")) ** 2

    def _fit_step(self, X: NDArray, verbose: int = 0) -> Tuple[NDArray, float]:
        """
        Perform a single iteration of the K-SVD algorithm.

        Parameters:
            X (NDArray): Input data. Shape: (num_samples, num_features).
            verbose (int): Verbosity level. Default: 0.

        Returns:
            NDArray: Reconstructed data. Shape: (num_samples, num_features).
        """
        # Sparse Coding Stage
        # Compute the sparse representation of the input data using OMP with
        # the current dictionary.
        X_reconstructed, coefs = self.transform(X, return_coefs=True)

        err = KSVD._error(X, X_reconstructed)
        if err < self.tol:
            return X_reconstructed, err

        # Update the dictionary using the sparse representation computed in the
        # previous step.
        it = trange(self.k, disable=(verbose < 2), leave=False)
        for k in it:
            coefs = self._entry_update(X, coefs, k)

        X_reconstructed = coefs @ self.dictionary
        return X_reconstructed, err

    def _entry_update(self, X: NDArray, coefs: NDArray, k: int) -> NDArray:
        """
        Update the k-th entry of the dictionary.

        X (NDArray): Input data. Shape: (num_samples, num_features).
        coefs (NDArray): Coefficients of the sparse representation.
            Shape: (num_samples, num_coefs).
        k (int): Index of the entry to be updated.

        Returns:
            coefs (NDArray): Updated coefficients of the sparse representation.
        """
        Ek = X - coefs @ self.dictionary + coefs[:, k, None] @ self.dictionary[k, None]
        Ek_R = Ek[coefs[:, k] != 0].T

        if Ek_R.shape[1] == 0:
            return coefs

        u, s, vh = np.linalg.svd(Ek_R, full_matrices=False)
        self.dictionary[k] = u[:, 0]
        try:
            coefs[coefs[:, k] != 0, k] = s[0] * vh[0, :]
        except ValueError:
            raise
        return coefs

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

        coefs = orthogonal_mp(
            self.dictionary.T, X.T, n_nonzero_coefs=self.num_coefs, precompute="auto"
        ).T
        X_reconstructed = coefs @ self.dictionary

        if return_coefs:
            return X_reconstructed, coefs
        return X_reconstructed
