# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for PatchFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.patch import PatchFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestPatchFeatures:
    """Test suite for PatchFeatures class."""

    def test_patch_features_extraction(self):
        """Test PatchFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = PatchFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "midband_energy_ratio" in features
            assert "patch_mean_cv" in features
            assert "mslbp_s1_mean" in features

    def test_patch_features_midband(self):
        """Test mid-band frequency features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = PatchFeatures(numeric)

            features = extractor.midband_frequency_features(numeric.gray)

            assert "midband_energy_ratio" in features
            assert "midband_deviation" in features

    def test_patch_features_patch_consistency(self):
        """Test patch consistency features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = PatchFeatures(numeric)

            features = extractor.patch_consistency_features(numeric.gray)

            assert "patch_mean_cv" in features
            assert "patch_std_cv" in features

    def test_patch_features_multiscale_lbp(self):
        """Test multi-scale LBP features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = PatchFeatures(numeric)

            features = extractor.multiscale_lbp_features(numeric.gray)

            assert "mslbp_s1_mean" in features
            assert "mslbp_s2_mean" in features
            assert "mslbp_s3_mean" in features
