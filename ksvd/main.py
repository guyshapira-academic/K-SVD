import numpy as np
from omegaconf import OmegaConf, DictConfig

from ksvd import KSVD
try:
    from ksvd import utils
except ImportError:
    import utils


def main(cfg: DictConfig):
    # Load the training data.
    X = utils.load_faces(patch_size=cfg.data.patch_size)

    # Random sample 1000 patches.
    X = X[np.random.choice(X.shape[0], 11000, replace=False)]

    # Initialize the K-SVD object.
    ksvd = KSVD(**cfg.model)

    # Learn the dictionary.
    ksvd.fit(X, **cfg.fit)


if __name__ == '__main__':
    try:
        cfg = OmegaConf.load('config/config.yaml')
    except FileNotFoundError:
        cfg = OmegaConf.load('../config/config.yaml')
    main(cfg)
