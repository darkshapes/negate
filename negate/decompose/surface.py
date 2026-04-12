# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract brightness, color, texture, shape, noise, and frequency features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from negate.decompose.numeric import NumericImage


class SurfaceFeatures:
    """Extract artwork features for AI detection."""

    def __init__(self, image: NumericImage) -> None:
        """Initialize SurfaceFeatures with NumericImage.\n
        :param image: NumericImage.
        """
        self.image = image
        self._numeric = image

    def __call__(self) -> dict[str, float]:
        """Extract all features from the image.\n
        :returns: Dictionary of scalar features.
        """
        gray = self._numeric.gray
        rgb = self._numeric.color
        features: dict[str, float] = {}
        features |= self.brightness_features(gray)
        features |= self.color_features(rgb)
        features |= self.texture_features(gray)
        features |= self.shape_features(gray)
        features |= self.noise_features(gray)
        features |= self.frequency_features(gray)
        return features

    def entropy(self, counts: NDArray) -> float:
        """Compute Shannon entropy from histogram counts."""
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def brightness_features(self, gray: NDArray) -> dict[str, float]:
        """Mean and entropy of pixel brightness."""
        gray_clean = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
        gray_clipped = np.clip(gray_clean, 0, 1)
        return {
            "mean_brightness": float(gray.mean()),
            "entropy_brightness": float(self.entropy(np.histogram(gray_clipped, bins=256, range=(0, 1))[0] + 1e-10)),
        }

    def color_features(self, rgb: NDArray) -> dict[str, float]:
        """RGB and HSV histogram statistics."""
        features: dict[str, float] = {}
        rgb_clean = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb_clipped = np.clip(rgb_clean, 0, 1)
        for i, name in enumerate(("red", "green", "blue")):
            channel = rgb_clipped[:, :, i].ravel()
            features[f"{name}_mean"] = float(channel.mean())
            features[f"{name}_variance"] = float(channel.var())
            features[f"{name}_kurtosis"] = float(kurtosis(channel))
            features[f"{name}_skewness"] = float(skew(channel))
        rgb_flat = rgb_clipped.reshape(-1, 3)
        rgb_hist = np.histogramdd(rgb_flat, bins=32, range=[[0, 1], [0, 1], [0, 1]])[0]
        features["rgb_entropy"] = float(self.entropy(rgb_hist.ravel() + 1e-10))
        hsv = np.nan_to_num(self._numeric.hsv, nan=0.0, posinf=1.0, neginf=0.0)
        hsv_clipped = np.clip(hsv, 0, 1)
        for i, name in enumerate(("hue", "saturation", "value")):
            channel = hsv_clipped[:, :, i].ravel()
            features[f"{name}_variance"] = float(channel.var())
            features[f"{name}_kurtosis"] = float(kurtosis(channel))
            features[f"{name}_skewness"] = float(skew(channel))
        hsv_flat = hsv_clipped.reshape(-1, 3)
        hsv_hist = np.histogramdd(hsv_flat, bins=32, range=[[0, 1], [0, 1], [0, 1]])[0]
        features["hsv_entropy"] = float(self.entropy(hsv_hist.ravel() + 1e-10))
        return features

    def shape_features(self, gray: NDArray) -> dict[str, float]:
        """HOG statistics and edge length."""
        from skimage.feature import hog
        import numpy as np

        hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        gray_uint8 = (gray * 255).astype(np.uint8)
        edges_array = np.where(gray_uint8 < 128, 0, 255)
        features: dict[str, float] = {
            "hog_mean": float(hog_features.mean()),
            "hog_variance": float(hog_features.var()),
            "hog_kurtosis": float(kurtosis(hog_features)),
            "hog_skewness": float(skew(hog_features)),
            "hog_entropy": float(self.entropy(np.histogram(hog_features, bins=50)[0] + 1e-10)),
        }
        features["edgelen"] = float(edges_array.sum())
        return features

    def noise_features(self, gray: NDArray) -> dict[str, float]:
        """Noise entropy and signal-to-noise ratio."""
        from skimage.restoration import estimate_sigma

        gray_clean = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
        sigma = estimate_sigma(gray_clean)
        noise = gray_clean - np.clip(gray_clean, gray_clean.mean() - 2 * sigma, gray_clean.mean() + 2 * sigma)
        noise_clean = np.nan_to_num(noise, nan=0.0)
        noise_hist = np.histogram(noise_clean.ravel(), bins=256)[0]
        noise_ent = float(self.entropy(noise_hist + 1e-10))
        signal_power = float(gray_clean.var())
        noise_power = float(sigma**2) if sigma > 0 else 1e-10
        snr = float(10 * np.log10(signal_power / noise_power + 1e-10))
        return {"noise_entropy": noise_ent, "snr": snr}

    def texture_features(self, gray: NDArray) -> dict[str, float]:
        """GLCM and LBP texture features."""
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
        glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        features: dict[str, float] = {
            "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
            "correlation": float(graycoprops(glcm, "correlation")[0, 0]),
            "energy": float(graycoprops(glcm, "energy")[0, 0]),
            "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        }
        lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
        features["lbp_entropy"] = float(self.entropy(np.histogram(lbp, bins=10)[0] + 1e-10))
        features["lbp_variance"] = float(lbp.var())
        return features

    def frequency_features(self, gray: NDArray) -> dict[str, float]:
        """FFT and DCT spectral analysis features."""
        from scipy.fft import dctn
        from numpy.fft import fftfreq

        height, width = gray.shape
        fft_2d = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft_2d)
        magnitude = np.abs(fft_shift)
        log_mag = np.log(magnitude + 1e-10)
        phase = np.angle(fft_shift)
        center_h, center_w = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        max_r = np.sqrt(center_h**2 + center_w**2)
        low_mask = radius < max_r * 0.2
        mid_mask = (radius >= max_r * 0.2) & (radius < max_r * 0.6)
        high_mask = radius >= max_r * 0.6
        total_energy = float((magnitude**2).sum() + 1e-10)
        low_energy = float((magnitude[low_mask] ** 2).sum())
        mid_energy = float((magnitude[mid_mask] ** 2).sum())
        high_energy = float((magnitude[high_mask] ** 2).sum())
        row_freqs = fftfreq(height)[:, None] * np.ones((1, width))
        col_freqs = np.ones((height, 1)) * fftfreq(width)[None, :]
        spectral_centroid = float((np.sum(log_mag * np.abs(row_freqs)) + np.sum(log_mag * np.abs(col_freqs))) / (log_mag.sum() * 2 + 1e-10))
        dct_coeffs: NDArray = dctn(gray.astype(np.float64), type=2, norm="ortho")
        dct_mag = np.abs(dct_coeffs)
        if dct_mag.ndim == 1:
            dct_mag = dct_mag.reshape(height, width)
        flat_dc_energy = float(dct_mag[0, 0] ** 2)
        detail_ac_energy = float((dct_mag**2).sum() - flat_dc_energy)
        phase_coherence = float(phase.std())
        return {
            "fft_low_energy_ratio": low_energy / total_energy,
            "fft_mid_energy_ratio": mid_energy / total_energy,
            "fft_high_energy_ratio": high_energy / total_energy,
            "fft_spectral_centroid": spectral_centroid,
            "fft_log_mag_mean": float(log_mag.mean()),
            "fft_log_mag_std": float(log_mag.std()),
            "fft_phase_std": phase_coherence,
            "dct_ac_dc_ratio": detail_ac_energy / (flat_dc_energy + 1e-10),
            "dct_high_freq_energy": float((dct_mag[height // 2 :, width // 2 :] ** 2).sum() / (dct_mag**2).sum()),
            "dct_sparsity": float((dct_mag < 0.01 * dct_mag.max()).mean()),
        }
