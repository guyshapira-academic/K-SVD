import numpy as np
from omegaconf import OmegaConf, DictConfig

from ksvd import KSVD

try:
    from ksvd import utils
except ImportError:
    import utils


def main(cfg: DictConfig):
    # Load the training data.
    X, sample_image = utils.load_faces(patch_size=cfg.data.patch_size)

    # If restrict_dataset is set to True, only use a subset of the training data.
    if cfg.data.restrict_dataset is not None and X.shape[0] > cfg.data.restrict_dataset:
        X = X[np.random.choice(X.shape[0], cfg.data.restrict_dataset, replace=False)]

    # Initialize the K-SVD object.
    ksvd = KSVD(**cfg.model)

    # Learn the dictionary.
    ksvd.fit(X, **cfg.fit)

    utils.display_patches(ksvd.dictionary)

    sample_image_reconstructed = ksvd.transform_image(sample_image)
    utils.display_images(sample_image, sample_image_reconstructed)

    mse = np.mean((sample_image - sample_image_reconstructed) ** 2)
    print(f"MSE: {mse:.4f}")

    mask_idx = np.random.choice(sample_image.size, sample_image.size // 8, replace=False)
    mask = np.zeros_like(sample_image, dtype=bool)
    mask.ravel()[mask_idx] = True

    sample_image_corrupted = sample_image.copy()
    sample_image_corrupted[mask] = 0

    sample_image_reconstructed = ksvd.masked_transform(sample_image_corrupted, mask)
    utils.display_images(sample_image, sample_image_corrupted, sample_image_reconstructed)


if __name__ == "__main__":
    try:
        cfg = OmegaConf.load("config/config.yaml")
    except FileNotFoundError:
        cfg = OmegaConf.load("../config/config.yaml")
    main(cfg)
