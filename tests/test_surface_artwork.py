# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for surface_artwork feature extraction classes."""

from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest
from negate.decompose.surface_artwork import (
    NumericImage,
    SurfaceFeatures,
    EnhancedFeatures,
    PatchFeatures,
    GaborFeatures,
    EdgeFeatures,
    ComplexFeatures,
    HOGFeatures,
    LineworkFeatures,
)


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

    def test_gabor_features_edge_cooccurrence(self):
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


class TestLineworkFeatures:
    """Test suite for LineworkFeatures class."""

    def test_linework_features_extraction(self):
        """Test LineworkFeatures feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = LineworkFeatures(numeric)

            features = extractor()

            assert isinstance(features, dict)
            assert len(features) > 0
            assert "line_thickness_mean" in features
            assert "line_density" in features

    def test_linework_features_linework(self):
        """Test line work features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)
            image = Image.open(img_path)
            numeric = NumericImage(image)
            extractor = LineworkFeatures(numeric)

            features = extractor.linework_features(numeric.gray)

            assert "line_thickness_mean" in features
            assert "line_density" in features
            assert "line_straightness" in features
