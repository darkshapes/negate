# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for GaborFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.gabor import GaborFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestGaborFeatures:
    """Test suite for GaborFeatures class."""

    def test_gabor_features_extraction(self):
        """Test GaborFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = GaborFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "gabor_f0_t0_energy" in features
            assert "wvt_L1_LH_mean" in features

    def test_gabor_features_gabor(self):
        """Test Gabor filter features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = GaborFeatures(numeric)

            features = extractor.gabor_features(numeric.gray)

            assert "gabor_f0_t0_energy" in features
            assert "gabor_mean_energy" in features

    def test_gabor_features_wavelet(self):
        """Test wavelet packet features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = GaborFeatures(numeric)

            features = extractor.wavelet_packet_features(numeric.gray)

            assert "wvt_L1_LH_mean" in features
            assert "wvt_L2_HL_mean" in features
