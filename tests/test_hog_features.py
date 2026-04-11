# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for HOGFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.hog import HOGFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestHOGFeatures:
    """Test suite for HOGFeatures class."""

    def test_hog_features_extraction(self):
        """Test HOGFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = HOGFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "hog_fine_energy" in features
            assert "jpeg_ghost_q50_rmse" in features

    def test_hog_features_extended_hog(self):
        """Test extended HOG features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = HOGFeatures(numeric)

            features = extractor.extended_hog_features(numeric.gray)

            assert "hog_fine_energy" in features
            assert "hog_fine_entropy" in features
            assert "hog_coarse_energy" in features

    def test_hog_features__jpeg_ghost(self):
        """Test JPEG ghost detection features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = HOGFeatures(numeric)

            features = extractor.jpeg_ghost_features(numeric.color)

            assert "jpeg_ghost_q50_rmse" in features
            assert "jpeg_ghost_q70_rmse" in features
            assert "jpeg_ghost_q90_rmse" in features
