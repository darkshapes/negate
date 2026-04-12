# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for SurfaceFeatures class."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.surface import SurfaceFeatures
from negate.decompose.numeric import NumericImage


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestSurfaceFeatures:
    """Test suite for SurfaceFeatures class."""

    def test_surface_features_creation(self):
        """Test SurfaceFeatures creation from NumericImage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            assert isinstance(extractor.image, NumericImage)

    def test_surface_features_extraction(self):
        """Test SurfaceFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "mean_brightness" in features
            assert "entropy_brightness" in features

    def test_surface_features_brightness(self):
        """Test brightness features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor.brightness_features(numeric.gray)

            assert "mean_brightness" in features
            assert "entropy_brightness" in features

    def test_surface_features_color(self):
        """Test color features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor.color_features(numeric.color)

            assert "red_mean" in features
            assert "green_mean" in features
            assert "blue_mean" in features

    def test_surface_features_texture(self):
        """Test texture features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor.texture_features(numeric.gray)

            assert "contrast" in features
            assert "correlation" in features
            assert "energy" in features
            assert "homogeneity" in features

    def test_surface_features_shape(self):
        """Test shape features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor.shape_features(numeric.gray)

            assert "edgelen" in features
            assert "hog_mean" in features
            assert "hog_variance" in features

    def test_surface_features_noise(self):
        """Test noise features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor.noise_features(numeric.gray)

            assert "noise_entropy" in features
            assert "snr" in features

    def test_surface_features_frequency(self):
        """Test frequency features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = SurfaceFeatures(numeric)

            features = extractor.frequency_features(numeric.gray)

            assert "fft_low_energy_ratio" in features
            assert "fft_mid_energy_ratio" in features
            assert "fft_high_energy_ratio" in features
