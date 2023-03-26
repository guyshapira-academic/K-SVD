import os
import logging
import pickle
import csv
import sys
import warnings

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import tabulate

try:
    from ksvd import KSVD
    import ksvd
    sys.modules['ksvd.ksvd'] = ksvd
except ImportError:
    from ksvd.ksvd import KSVD

try:
    from ksvd import utils
except ImportError:
    import utils


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    logger.info(f"output_dir: {output_dir}")

    # Load the training data.
    X, sample_image = utils.load_faces(
        patch_size=cfg.data.patch_size, resize=cfg.data.resize_images
    )
    if cfg.use_pretrained is not None:
        with open(cfg.use_pretrained, "rb") as f:
            model = pickle.loads(f.read())
        ksvd = model
    else:
        # If restrict_dataset is set to True, only use a subset of the training data.
        if (
            cfg.data.restrict_dataset is not None
            and X.shape[0] > cfg.data.restrict_dataset
        ):
            X = X[
                np.random.choice(X.shape[0], cfg.data.restrict_dataset, replace=False)
            ]

        # Initialize the K-SVD object.
        ksvd = KSVD(**cfg.model)

        # Learn the dictionary.
        err_log = ksvd.fit(X, **cfg.fit)
        err_log = np.array(err_log)
        np.savetxt(os.path.join(output_dir, "error_log.csv"), err_log, delimiter=",")
        utils.display_error_log(
            err_log, show=False, save=os.path.join(output_dir, "error_log.png")
        )

    # Save the model.
    if cfg.save_model:
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(ksvd, f)

    utils.display_patches(
        ksvd.dictionary, show=False, save=os.path.join(output_dir, "dictionary.png")
    )

    # Evaluate reconstruction task
    reconstruction_metrics = list()
    rmse_list = list()
    reconstruction_metrics.append(["corruption_ratio", "RMSE", "MAE"])
    for ratio in cfg.eval.corruption_ratios:
        mask = utils.random_mask(sample_image.shape, ratio)
        sample_image_corrupted = utils.corrupt_image(sample_image, mask)
        sample_image_reconstructed = ksvd.masked_image_transform(
            sample_image_corrupted, mask
        )
        sample_image_reconstructed = np.where(mask, sample_image_reconstructed, sample_image_corrupted)
        save_path = os.path.join(output_dir, f"reconstruction-{ratio * 100}.png")
        rmse = utils.rmse(sample_image, sample_image_reconstructed)
        mae = utils.mae(sample_image, sample_image_reconstructed)
        if ratio == 0:
            utils.display_images(
                sample_image, sample_image_reconstructed, show=False, save=save_path, metrics={'RMSE': rmse, 'MAE': mae}
            )
        else:
            utils.display_images(
                sample_image_corrupted,
                sample_image_reconstructed,
                show=False,
                save=save_path,
                metrics={'RMSE': rmse, 'MAE': mae}
            )

        log_str = f"Reconstruction - {ratio * 100}%: RMSE: {rmse:.4f}, MAE: {mae:.4f}"
        rmse_list.append(rmse)
        logger.info(log_str)
        reconstruction_metrics.append([ratio, rmse, mae])
    # log reconstruction to csv
    with open(os.path.join(output_dir, "reconstruction.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(reconstruction_metrics)
    reconstruction_table = tabulate.tabulate(
        reconstruction_metrics, headers="firstrow", tablefmt="github"
    )
    logger.info(f"Reconstruction Results:\n{reconstruction_table}")

    # plot reconstruction error vs corruption ratio
    plt.plot(cfg.eval.corruption_ratios, rmse_list)
    plt.xlabel("Corruption Ratio")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(output_dir, "reconstruction_error.png"))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
