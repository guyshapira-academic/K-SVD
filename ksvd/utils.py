from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from PIL import Image
import einops


def center_crop(image: NDArray, crop_size: int) -> NDArray:
    """Center crop image.

    Args:
        image: Image as numpy array.
        crop_size: Size of crop.

    Returns:
        Cropped image as numpy array.
    """
    h, w = image.shape[:2]
    h_start = (h - crop_size) // 2
    w_start = (w - crop_size) // 2
    return image[h_start : h_start + crop_size, w_start : w_start + crop_size]


def load_image(path: str, size: int = 168, rgb: bool = False) -> NDArray:
    """Load image from path and convert to numpy array.

    Args:
        path: Path to image.
        size: Size of image.
        rgb: If True, load as RGB or grayscale, otherwise load as grayscale.

    Returns:
        Image as numpy array.
    """
    if rgb:
        image = Image.open(path).convert("RGB")
    else:
        image = Image.open(path).convert("L")

    current_size = image.size
    ratio = size / max(current_size)
    new_size = tuple([int(x * ratio) for x in current_size])
    image.resize(new_size)

    image = np.array(image) / 255
    image = center_crop(image, size)

    return image


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
    path: str, patch_size: int, image_size: int = 168, rgb: bool = False
) -> NDArray:
    """Load dataset from directory.

    Args:
        path: Path to directory.
        patch_size: Size of patches.
        image_size: Size of images.
        rgb: If True, load as RGB or grayscale, otherwise load as grayscale.

    Returns:
        Dataset as numpy array.
    """
    dataset = []
    for image_path in Path(path).glob("*.png"):
        image = load_image(image_path, image_size, rgb=rgb)
        patches = image_to_patches(image, patch_size)
        dataset.append(patches)

    return np.concatenate(dataset, axis=0)