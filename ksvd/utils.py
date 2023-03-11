from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from sklearn import datasets
from skimage import transform
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


def load_image(path: str, size: int = 256, rgb: bool = False) -> NDArray:
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
    path: str, patch_size: int, image_size: int = 256, rgb: bool = False
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
    sample_image = None
    for i, image_path in enumerate(Path(path).glob("*_HR.png")):
        image = load_image(image_path, image_size, rgb=rgb)
        if i == 0:
            sample_image = image
        else:
            patches = image_to_patches(image, patch_size)
            dataset.append(patches)

    dataset = np.concatenate(dataset[1:], axis=0)

    return dataset, sample_image


def load_faces(patch_size: int = 8, resize: Optional[int] = None) -> NDArray:
    """Load Olivetti faces dataset.

    Args:
        patch_size: Size of patches.
        resize: Resize images to this size.

    Returns:
        Dataset as numpy array.
    """
    X = datasets.fetch_olivetti_faces().data
    X = einops.rearrange(X, "n (h w) -> n h w", h=64, w=64)
    if resize is not None:
        resized_images = list()
        for i in range(X.shape[0]):
            resized_images.append(
                transform.resize(X[i], (resize, resize))
            )
        X = np.array(resized_images)
    sample_image = X[0]
    X = einops.rearrange(
        X, "n (h p1) (w p2) -> (n h w) (p1 p2)", p1=patch_size, p2=patch_size
    )
    return X, sample_image


def learned_dictionary_patches(dictionary: NDArray) -> NDArray:
    patch_size = int(np.sqrt(dictionary.shape[1]))
    patches = einops.rearrange(
        dictionary, "n (h w) -> n h w", h=patch_size, w=patch_size
    )

    # Sort by variance.
    variances = np.var(patches, axis=(1, 2))
    sorted_indices = np.argsort(variances)[::-1]
    patches = patches[sorted_indices]

    return patches


def display_patches(patches: NDArray) -> None:
    """Display patches.

    Args:
        patches: Patches as numpy array.

    Returns:
        None.
    """
    if len(patches.shape) == 2:
        patches = learned_dictionary_patches(patches)
    n_patches = patches.shape[0]
    n_cols = int(np.sqrt(n_patches))
    n_rows = n_patches // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].imshow(patches[i * n_cols + j], cmap="gray")
            axes[i, j].axis("off")
    fig.tight_layout()
    plt.show()


def display_images(*images: NDArray) -> None:
    """Display images.

    Args:
        images: Images as numpy arrays.

    Returns:
        None.
    """
    n_images = len(images)

    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(n_images, 1))
    for i in range(n_images):
        axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
    fig.tight_layout()
    plt.show()
