# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract enhanced texture features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from negate.decompose.numeric import NumericImage


def entropy(counts: NDArray) -> float:
    """Compute Shannon entropy from histogram counts."""
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


class EnhancedFeatures:
    """Extract enhanced texture features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract enhanced texture features from the NumericImage."""
        gray = self.image.gray
        features: dict[str, float] = {}
        features |= self.enhanced_texture_features(gray)
        return features

    def enhanced_texture_features(self, gray: NDArray) -> dict[str, float]:
        """Extended GLCM + full LBP histogram + block DCT features."""
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        distances = [1, 3]
        glcm = graycomatrix(gray_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        features: dict[str, float] = {}
        for prop in ("contrast", "correlation", "energy", "homogeneity"):
            vals = graycoprops(glcm, prop)
            features[f"glcm_multi_{prop}_mean"] = float(vals.mean())
            features[f"glcm_multi_{prop}_std"] = float(vals.std())
        lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        features["lbp_hist_kurtosis"] = float(kurtosis(lbp_hist))
        features["lbp_hist_skew"] = float(skew(lbp_hist))
        features["lbp_hist_max"] = float(lbp_hist.max())
        lbp_coarse = local_binary_pattern(gray_uint8, P=16, R=2, method="uniform")
        features["lbp_coarse_entropy"] = float(entropy(np.histogram(lbp_coarse, bins=18)[0] + 1e-10))
        from scipy.fft import dctn

        h, w = gray.shape
        block_size = 8
        block_energies = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y : y + block_size, x : x + block_size]
                dct_block = dctn(block, type=2, norm="ortho")
                ac_energy = float((dct_block**2).sum() - dct_block[0, 0] ** 2)  # type: ignore
                block_energies.append(ac_energy)
        block_energies = np.array(block_energies)
        features["dct_block_energy_mean"] = float(block_energies.mean())
        features["dct_block_energy_std"] = float(block_energies.std())
        return features
