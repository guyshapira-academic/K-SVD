import einops
import numpy as np
from omegaconf import OmegaConf, DictConfig

from ksvd import KSVD

try:
    from ksvd import utils
except ImportError:
    import utils


def main(cfg: DictConfig):
    # Load the training data.
    X, sample_image = utils.load_faces(patch_size=cfg.data.patch_size, resize=cfg.data.resize_images)

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


    for ratio in [0, 0.3, 0.5, 0.7]:
        print(f"\nRatio: {ratio}")
        mask = utils.random_mask(sample_image.shape, ratio)
        sample_image_corrupted = utils.corrupt_image(sample_image, mask)
        sample_image_reconstructed = ksvd.masked_image_transform(
            sample_image_corrupted, mask
        )
        if ratio == 0:
            utils.display_images(
                sample_image, sample_image_corrupted, sample_image_reconstructed
            )
        else:
            utils.display_images(
                sample_image, sample_image_corrupted, sample_image_reconstructed
            )
        rmse = utils.rmse(sample_image, sample_image_reconstructed)
        print(f"RMSE: {rmse:.4f}")
        mae = utils.mae(sample_image, sample_image_reconstructed)
        print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    try:
        cfg = OmegaConf.load("config/config.yaml")
    except FileNotFoundError:
        cfg = OmegaConf.load("../config/config.yaml")
    main(cfg)
