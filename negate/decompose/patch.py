# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract patch and multi-scale LBP features for AI detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_entropy
from skimage.feature import canny, local_binary_pattern

from negate.decompose.numeric import NumericImage


def entropy(counts: NDArray) -> float:
    """Compute Shannon entropy from histogram counts."""
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


class PatchFeatures:
    """Extract patch and multi-scale LBP features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract patch and multi-scale LBP features from the NumericImage."""
        gray = self.image.gray
        features: dict[str, float] = {}
        features |= self.midband_frequency_features(gray)
        features |= self.patch_consistency_features(gray)
        features |= self.multiscale_lbp_features(gray)
        return features

    def midband_frequency_features(self, gray: NDArray) -> dict[str, float]:
        """Mid-band frequency analysis features."""
        h, w = gray.shape
        fft_2d = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft_2d)
        magnitude = np.abs(fft_shift)
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        max_r = np.sqrt(center_h**2 + center_w**2)
        bands = [(0, 0.1), (0.1, 0.25), (0.25, 0.45), (0.45, 0.7), (0.7, 1.0)]
        band_energies = []
        for lo, hi in bands:
            mask = (radius >= max_r * lo) & (radius < max_r * hi)
            band_energies.append(float((magnitude[mask] ** 2).sum()))
        total = sum(band_energies) + 1e-10
        band_ratios = [e / total for e in band_energies]
        expected_ratios = np.array([0.65, 0.20, 0.10, 0.035, 0.015])
        actual_ratios = np.array(band_ratios)
        deviation = actual_ratios - expected_ratios
        return {
            "midband_energy_ratio": float(band_ratios[2]),
            "midband_deviation": float(deviation[2]),
            "spectral_slope_deviation": float(np.std(deviation)),
            "high_to_mid_ratio": float(band_ratios[4] / (band_ratios[2] + 1e-10)),
        }

    def patch_consistency_features(self, gray: NDArray) -> dict[str, float]:
        """Cross-patch consistency features."""
        h, w = gray.shape
        patch_size = 32
        n_patches = 0
        patch_means = []
        patch_stds = []
        patch_edges = []
        patch_freq_centroids = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y : y + patch_size, x : x + patch_size]
                patch_means.append(float(patch.mean()))
                patch_stds.append(float(patch.std()))
                edges = canny(patch)
                patch_edges.append(float(edges.mean()))
                fft_p = np.fft.fft2(patch)
                mag_p = np.abs(fft_p)
                freqs = np.fft.fftfreq(patch_size)
                freq_grid = np.sqrt(freqs[:, None] ** 2 + freqs[None, :] ** 2)
                centroid = float(np.sum(mag_p * freq_grid) / (mag_p.sum() + 1e-10))
                patch_freq_centroids.append(centroid)
                n_patches += 1
        if n_patches < 4:
            return {
                k: 0.0
                for k in [
                    "patch_mean_cv",
                    "patch_std_cv",
                    "patch_edge_cv",
                    "patch_freq_centroid_cv",
                    "patch_freq_centroid_range",
                    "patch_coherence_score",
                ]
            }

        def _cv(arr: list[float]) -> float:
            a = np.array(arr)
            return float(a.std() / (abs(a.mean()) + 1e-10))

        freq_arr = np.array(patch_freq_centroids)
        return {
            "patch_mean_cv": _cv(patch_means),
            "patch_std_cv": _cv(patch_stds),
            "patch_edge_cv": _cv(patch_edges),
            "patch_freq_centroid_cv": _cv(patch_freq_centroids),
            "patch_freq_centroid_range": float(freq_arr.max() - freq_arr.min()),
            "patch_coherence_score": float(np.corrcoef(patch_means, patch_stds)[0, 1]) if len(patch_means) > 2 else 0.0,
        }

    def multiscale_lbp_features(self, gray: NDArray) -> dict[str, float]:
        """Multi-scale LBP features."""
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
        features: dict[str, float] = {}
        scales = [(8, 1, "s1"), (16, 2, "s2"), (24, 3, "s3")]
        for p, r, label in scales:
            lbp = local_binary_pattern(gray_uint8, P=p, R=r, method="uniform")
            n_bins = p + 2
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            features[f"mslbp_{label}_mean"] = float(lbp.mean())
            features[f"mslbp_{label}_var"] = float(lbp.var())
            if r == 3:
                features[f"mslbp_{label}_entropy"] = float(entropy(hist + 1e-10))
                features[f"mslbp_{label}_uniformity"] = float(hist.max())
        return features
