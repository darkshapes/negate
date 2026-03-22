# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Artwork feature extraction for AI-generated image detection.

Implements the 39-feature extraction pipeline from:
    Li & Stamp, "Detecting AI-generated Artwork", arXiv:2504.07078, 2025.

Extended with a dedicated frequency analysis branch (FFT/DCT) that captures
spectral fingerprints left by generative models.

Features are grouped into 6 categories:
    - Brightness (2): mean, entropy
    - Color (23): RGB/HSV histogram statistics
    - Texture (6): GLCM + LBP
    - Shape (6): HOG + edge length
    - Noise (2): noise entropy, SNR
    - Frequency (10): FFT/DCT spectral analysis
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import entropy, kurtosis, skew
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


_TARGET_SIZE = (255, 255)


def _to_array(image: Image.Image) -> NDArray:
    """Resize to 255x255 and convert to float64 numpy array."""
    image = image.convert("RGB").resize(_TARGET_SIZE, Image.BICUBIC)
    return np.asarray(image, dtype=np.float64)


def _brightness_features(gray: NDArray) -> dict[str, float]:
    """Mean and entropy of pixel brightness."""
    return {
        "mean_brightness": float(gray.mean()),
        "entropy_brightness": float(entropy(np.histogram(gray, bins=256, range=(0, 1))[0] + 1e-10)),
    }


def _color_features(rgb: NDArray) -> dict[str, float]:
    """RGB and HSV histogram statistics (23 features)."""
    features: dict[str, float] = {}

    # RGB: mean, variance, kurtosis, skewness per channel + entropy
    for i, name in enumerate(("red", "green", "blue")):
        channel = rgb[:, :, i].ravel()
        features[f"{name}_mean"] = float(channel.mean())
        features[f"{name}_variance"] = float(channel.var())
        features[f"{name}_kurtosis"] = float(kurtosis(channel))
        features[f"{name}_skewness"] = float(skew(channel))

    # RGB entropy (joint)
    rgb_flat = rgb.reshape(-1, 3)
    rgb_hist = np.histogramdd(rgb_flat, bins=32)[0]
    features["rgb_entropy"] = float(entropy(rgb_hist.ravel() + 1e-10))

    # HSV: variance, kurtosis, skewness per channel + entropy
    hsv = rgb2hsv(rgb / 255.0 if rgb.max() > 1 else rgb)
    for i, name in enumerate(("hue", "saturation", "value")):
        channel = hsv[:, :, i].ravel()
        features[f"{name}_variance"] = float(channel.var())
        features[f"{name}_kurtosis"] = float(kurtosis(channel))
        features[f"{name}_skewness"] = float(skew(channel))

    hsv_flat = hsv.reshape(-1, 3)
    hsv_hist = np.histogramdd(hsv_flat, bins=32)[0]
    features["hsv_entropy"] = float(entropy(hsv_hist.ravel() + 1e-10))

    return features


def _texture_features(gray: NDArray) -> dict[str, float]:
    """GLCM and LBP texture features (6 features)."""
    # GLCM requires uint8
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)

    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    features: dict[str, float] = {
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
    }

    # LBP
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    features["lbp_entropy"] = float(entropy(np.histogram(lbp, bins=10)[0] + 1e-10))
    features["lbp_variance"] = float(lbp.var())

    return features


def _shape_features(gray: NDArray) -> dict[str, float]:
    """HOG statistics and edge length (6 features)."""
    from skimage.feature import hog, canny

    # HOG
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    features: dict[str, float] = {
        "hog_mean": float(hog_features.mean()),
        "hog_variance": float(hog_features.var()),
        "hog_kurtosis": float(kurtosis(hog_features)),
        "hog_skewness": float(skew(hog_features)),
        "hog_entropy": float(entropy(np.histogram(hog_features, bins=50)[0] + 1e-10)),
    }

    # Edge length via Canny
    edges = canny(gray if gray.max() <= 1 else gray / 255.0)
    features["edgelen"] = float(edges.sum())

    return features


def _noise_features(gray: NDArray) -> dict[str, float]:
    """Noise entropy and signal-to-noise ratio (2 features)."""
    from skimage.restoration import estimate_sigma

    # Estimate noise
    sigma = estimate_sigma(gray)
    noise = gray - np.clip(gray, gray.mean() - 2 * sigma, gray.mean() + 2 * sigma)

    noise_hist = np.histogram(noise.ravel(), bins=256)[0]
    noise_ent = float(entropy(noise_hist + 1e-10))

    # SNR
    signal_power = float(gray.var())
    noise_power = float(sigma ** 2) if sigma > 0 else 1e-10
    snr = float(10 * np.log10(signal_power / noise_power + 1e-10))

    return {
        "noise_entropy": noise_ent,
        "snr": snr,
    }


def _frequency_features(gray: NDArray) -> dict[str, float]:
    """FFT and DCT spectral analysis features (10 features).

    AI generators leave characteristic signatures in the frequency domain
    due to upsampling layers and attention patterns. This branch captures
    those patterns independently of pixel-space features.
    """
    from scipy.fft import dctn
    from numpy.fft import fftfreq

    h, w = gray.shape

    # 2D FFT analysis
    fft_2d = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft_2d)
    magnitude = np.abs(fft_shift)
    log_mag = np.log(magnitude + 1e-10)
    phase = np.angle(fft_shift)

    center_h, center_w = h // 2, w // 2

    # Radial frequency bands (low/mid/high)
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
    max_r = np.sqrt(center_h ** 2 + center_w ** 2)

    low_mask = radius < max_r * 0.2
    mid_mask = (radius >= max_r * 0.2) & (radius < max_r * 0.6)
    high_mask = radius >= max_r * 0.6

    total_energy = float((magnitude ** 2).sum() + 1e-10)
    low_energy = float((magnitude[low_mask] ** 2).sum())
    mid_energy = float((magnitude[mid_mask] ** 2).sum())
    high_energy = float((magnitude[high_mask] ** 2).sum())

    # Spectral centroid (center of mass of frequency distribution)
    row_freqs = fftfreq(h)[:, None] * np.ones((1, w))
    col_freqs = np.ones((h, 1)) * fftfreq(w)[None, :]
    spectral_centroid = float(
        (np.sum(log_mag * np.abs(row_freqs)) + np.sum(log_mag * np.abs(col_freqs)))
        / (log_mag.sum() * 2 + 1e-10)
    )

    # DCT analysis — captures compression and generation artifacts
    dct_coeffs = dctn(gray, type=2, norm="ortho")
    dct_mag = np.abs(dct_coeffs)

    # Ratio of AC to DC energy (how much detail vs flat)
    dc_energy = float(dct_mag[0, 0] ** 2)
    ac_energy = float((dct_mag ** 2).sum() - dc_energy)

    # Phase coherence — AI images often have more regular phase patterns
    phase_std = float(phase.std())

    return {
        "fft_low_energy_ratio": low_energy / total_energy,
        "fft_mid_energy_ratio": mid_energy / total_energy,
        "fft_high_energy_ratio": high_energy / total_energy,
        "fft_spectral_centroid": spectral_centroid,
        "fft_log_mag_mean": float(log_mag.mean()),
        "fft_log_mag_std": float(log_mag.std()),
        "fft_phase_std": phase_std,
        "dct_ac_dc_ratio": ac_energy / (dc_energy + 1e-10),
        "dct_high_freq_energy": float((dct_mag[h // 2:, w // 2:] ** 2).sum() / (dct_mag ** 2).sum()),
        "dct_sparsity": float((dct_mag < 0.01 * dct_mag.max()).mean()),
    }


class ArtworkExtract:
    """Extract artwork features for AI detection.

    Combines the 39 features from Li & Stamp (2025) with a dedicated
    frequency analysis branch (10 features) for 49 total features.

    All features are CPU-only and work on any image type (photos,
    illustrations, artwork). No pretrained models required.

    Usage:
        >>> extractor = ArtworkExtract()
        >>> features = extractor(pil_image)
        >>> len(features)  # 49
    """

    def __call__(self, image: Image.Image) -> dict[str, float]:
        """Extract all features from a single PIL image.

        :param image: PIL Image in any mode (will be converted to RGB).
        :returns: Dictionary of scalar features.
        """
        rgb = _to_array(image)
        gray = rgb2gray(rgb / 255.0 if rgb.max() > 1 else rgb)

        features: dict[str, float] = {}
        features |= _brightness_features(gray)
        features |= _color_features(rgb)
        features |= _texture_features(gray)
        features |= _shape_features(gray)
        features |= _noise_features(gray)
        features |= _frequency_features(gray)

        return features

    def feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        # Generate from a dummy image to get exact keys
        dummy = Image.new("RGB", (255, 255), color="gray")
        return list(self(dummy).keys())
