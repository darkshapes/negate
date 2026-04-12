# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for EnhancedFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.enhanced import EnhancedFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestEnhancedFeatures:
    """Test suite for EnhancedFeatures class."""

    def test_enhanced_features_extraction(self):
        """Test EnhancedFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = EnhancedFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "glcm_multi_contrast_mean" in features

    def test_enhanced_features_enhanced_texture(self):
        """Test enhanced texture features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = EnhancedFeatures(numeric)

            features = extractor.enhanced_texture_features(numeric.gray)

            assert "glcm_multi_contrast_mean" in features
            assert "lbp_coarse_entropy" in features
