# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract Gabor and wavelet features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from skimage.filters import gabor


class GaborFeatures:
    """Extract Gabor and wavelet features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract Gabor and wavelet features from the NumericImage."""
        gray = self.image.gray
        features: dict[str, float] = {}
        features |= self.gabor_features(gray)
        features |= self.wavelet_packet_features(gray)
        return features

    def gabor_features(self, gray: NDArray) -> dict[str, float]:
        """Gabor filter bank features."""
        features: dict[str, float] = {}
        all_energies = []
        freqs = [0.1, 0.2, 0.3, 0.4]
        thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        for fi, freq in enumerate(freqs):
            for ti, theta in enumerate(thetas):
                filt_real, filt_imag = gabor(gray, frequency=freq, theta=theta)
                energy = float(np.sqrt(filt_real**2 + filt_imag**2).mean())
                features[f"gabor_f{fi}_t{ti}_energy"] = energy
                all_energies.append(energy)
        all_e = np.array(all_energies)
        features["gabor_mean_energy"] = float(all_e.mean())
        features["gabor_std_energy"] = float(all_e.std())
        return features

    def wavelet_packet_features(self, gray: NDArray) -> dict[str, float]:
        """Wavelet packet statistics features."""
        import pywt

        coeffs = pywt.wavedec2(gray, "haar", level=2)
        features: dict[str, float] = {}
        subband_names = ["LH", "HL", "HH"]
        for level_idx, level in enumerate([1, 2]):
            detail_tuple = coeffs[len(coeffs) - level]
            for sb_idx, sb_name in enumerate(subband_names):
                c = detail_tuple[sb_idx]
                prefix = f"wvt_L{level}_{sb_name}"
                features[f"{prefix}_mean"] = float(np.abs(c).mean())
                features[f"{prefix}_std"] = float(c.std())
        return features
