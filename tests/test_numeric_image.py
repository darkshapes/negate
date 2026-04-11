# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for NumericImage class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestNumericImage:
    """Test suite for NumericImage class."""

    def test_numeric_image_creation(self):
        """Test NumericImage creation from PIL image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)

            assert hasattr(numeric, "gray")
            assert hasattr(numeric, "color")
            assert hasattr(numeric, "hsv")
            assert numeric.gray.shape == (255, 255)
            assert numeric.color.shape == (255, 255, 3)

    def test_numeric_image_gray(self):
        """Test numeric image grayscale array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)

            assert isinstance(numeric.gray, np.ndarray)
            assert numeric.gray.shape == (255, 255)

    def test_numeric_image_color(self):
        """Test numeric image color array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)

            assert isinstance(numeric.color, np.ndarray)
            assert numeric.color.shape == (255, 255, 3)

    def test_numeric_image_hsv(self):
        """Test numeric image HSV array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)

            assert isinstance(numeric.hsv, np.ndarray)
            assert numeric.hsv.shape == (255, 255, 3)
