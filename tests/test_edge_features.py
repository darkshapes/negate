# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for EdgeFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.edge import EdgeFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestEdgeFeatures:
    """Test suite for EdgeFeatures class."""

    def test_edge_features_extraction(self):
        """Test EdgeFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = EdgeFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "edge_cooc_contrast_mean" in features

    def test_edge_features_edge_cooccurrence(self):
        """Test edge co-occurrence features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = EdgeFeatures(numeric)

            features = extractor.edge_cooccurrence_features(numeric.gray)

            assert "edge_cooc_contrast_mean" in features
            assert "edge_cooc_homogeneity_mean" in features
