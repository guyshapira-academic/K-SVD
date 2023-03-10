from pathlib import Path

import pytest

from PIL import Image

from ksvd import utils


@pytest.fixture
def image_path(tmp_path) -> str:
    """
    Crate random image and return path to it.
    """
    image = Image.new("RGB", (112, 112))
    path = tmp_path / "test.png"
    image.save(path)
    return str(path)


def test_load_patches(image_path: str) -> None:
    """
    Test load_image function.
    """
    image = utils.load_image(image_path, rgb=False)
    assert image.shape == (112, 112)

    patches = utils.image_to_patches(image, 8)
    assert patches.shape == (196, 64)


def test_load_dataset_from_dir(image_path: str) -> None:
    """
    Test load_dataset_from_dir function.
    """
    folder = Path(image_path).parent
    dataset = utils.load_dataset_from_dir(folder, 8)
    assert dataset.shape == (196, 64)
