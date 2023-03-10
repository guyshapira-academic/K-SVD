from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from PIL import Image
import einops


def load_image(path: str, rgb: bool = False) -> NDArray:
    """Load image from path and convert to numpy array.

    Args:
        path: Path to image.
        rgb: If True, load as RGB or grayscale, otherwise load as grayscale.

    Returns:
        Image as numpy array.
    """
    if rgb:
        image = Image.open(path).convert("RGB")
    else:
        image = Image.open(path).convert("L")

    return np.array(image)


def image_to_patches(image: NDArray, patch_size: int) -> NDArray:
    """Convert image to patches.

    Args:
        image: Image as numpy array.
        patch_size: Size of patches.

    Returns:
        Patches as numpy array.
    """
    return einops.rearrange(
        image, "(h p1) (w p2) -> (h w) (p1 p2)", p1=patch_size, p2=patch_size
    )


def load_dataset_from_dir(
    path: str, patch_size: int, rgb: bool = False
) -> NDArray:
    """Load dataset from directory.

    Args:
        path: Path to directory.
        patch_size: Size of patches.
        rgb: If True, load as RGB or grayscale, otherwise load as grayscale.

    Returns:
        Dataset as numpy array.
    """
    dataset = []
    for image_path in Path(path).glob("*.png"):
        image = load_image(image_path, rgb=rgb)
        patches = image_to_patches(image, patch_size)
        dataset.append(patches)

    return np.concatenate(dataset, axis=0)
