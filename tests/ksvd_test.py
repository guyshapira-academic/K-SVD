import pytest

import numpy as np
from numpy.typing import NDArray

from sklearn.datasets import make_sparse_coded_signal

from ksvd import KSVD
try:
    from ksvd import utils
except ImportError:
    import utils


@pytest.fixture
def random_signal():
    return np.random.randn(512, 64)


@pytest.fixture
def sparse_signal():
    return make_sparse_coded_signal(
        n_samples=1024,
        n_components=441,
        n_features=64,
        random_state=42,
        n_nonzero_coefs=8,
        data_transposed=False,
    )


def test_transform(random_signal: NDArray):
    ksvd = KSVD(max_iter=2)
    ksvd.fit(random_signal, verbose=0)
    X_reconstructed, coefs = ksvd.transform(random_signal, return_coefs=True)
    assert X_reconstructed.shape == random_signal.shape


def test_ksvd(sparse_signal: NDArray):
    X, D, coefs = sparse_signal
    ksvd = KSVD(max_iter=2)
    ksvd.fit(X, verbose=0)
    X_reconstructed, coefs = ksvd.transform(X, return_coefs=True)

    assert X_reconstructed.shape == X.shape
    assert coefs.shape == (X.shape[0], D.shape[0])

    assert np.allclose(coefs @ ksvd.dictionary, X_reconstructed, atol=1e-6)


def test_masked_transform(random_signal: NDArray):
    ksvd = KSVD(max_iter=2, tol=1e-1)
    ksvd.fit(random_signal, verbose=2)
    mask = np.zeros_like(random_signal, dtype=bool)
    X_reconstructed = ksvd.masked_transform(random_signal, mask)

    assert np.allclose(X_reconstructed, random_signal, atol=1e-6)

