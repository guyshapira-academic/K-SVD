import pytest

import numpy as np
from numpy.typing import NDArray

from ksvd import KSVD


@pytest.fixture
def random_signal():
    return np.random.randn(1024, 64)


def test_transform(random_signal: NDArray):
    ksvd = KSVD(max_iter=1)
    ksvd.fit(random_signal)
    X_reconstructed, coefs = ksvd.transform(random_signal, return_coefs=True)

    assert X_reconstructed.shape == random_signal.shape
