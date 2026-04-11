# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for ComplexFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.complex import ComplexFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestComplexFeatures:
    """Test suite for ComplexFeatures class."""

    def test_complex_features_extraction(self):
        """Test ComplexFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = ComplexFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "fractal_dim_gray" in features
            assert "acf_n_secondary_peaks" in features
            assert "stroke_edge_roughness" in features
            assert "color_grad_curvature_mean" in features

    def test_complex_features_fractal(self):
        """Test fractal dimension features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = ComplexFeatures(numeric)

            features = extractor.fractal_dimension_features(numeric.gray)

            assert "fractal_dim_gray" in features
            assert "fractal_dim_edges" in features

    def test_complex_features_noise_residual(self):
        """Test noise residual autocorrelation features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = ComplexFeatures(numeric)

            features = extractor.noise_residual_autocorr_features(numeric.gray)

            assert "acf_n_secondary_peaks" in features
            assert "acf_max_secondary_peak" in features

    def test_complex_features_stroke_edge(self):
        """Test stroke edge roughness features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = ComplexFeatures(numeric)

            features = extractor.stroke_edge_roughness_features(numeric.gray)

            assert "stroke_edge_roughness" in features
            assert "stroke_edge_length_var" in features
