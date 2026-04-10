# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract HOG and JPEG features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from io import BytesIO
from skimage.feature import hog
from scipy.stats import entropy
from skimage.color import rgb2gray


class HOGFeatures:
    """Extract HOG and JPEG features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract HOG and JPEG features from the NumericImage."""
        gray = self.image.gray
        rgb = self.image.color
        features: dict[str, float] = {}
        features |= self.extended_hog_features(gray)
        features |= self.jpeg_ghost_features(rgb)
        return features

    def extended_hog_features(self, gray: NDArray) -> dict[str, float]:
        """Extended HOG features."""
        features: dict[str, float] = {}
        hog_fine = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        fine_energy = float((hog_fine**2).sum())
        fine_hist = np.histogram(hog_fine, bins=50)[0]
        features["hog_fine_energy"] = fine_energy
        features["hog_fine_entropy"] = float(entropy(fine_hist + 1e-10))
        hog_coarse = hog(gray, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
        coarse_energy = float((hog_coarse**2).sum())
        coarse_hist = np.histogram(hog_coarse, bins=50)[0]
        features["hog_coarse_energy"] = coarse_energy
        features["hog_coarse_entropy"] = float(entropy(coarse_hist + 1e-10))
        features["hog_fine_coarse_ratio"] = fine_energy / (coarse_energy + 1e-10)
        features["hog_energy_ratio_to_mean"] = fine_energy / (float(hog_fine.mean()) + 1e-10)
        return features

    def jpeg_ghost_features(self, rgb: NDArray) -> dict[str, float]:
        """JPEG ghost detection features."""
        arr = rgb.astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
        features: dict[str, float] = {}
        rmses = []
        for q in [50, 70, 90]:
            try:
                buf = BytesIO()
                PILImage.fromarray(arr).save(buf, format="JPEG", quality=q)
                buf.seek(0)
                resaved = np.array(PILImage.open(buf).convert("RGB"), dtype=np.float64)
                arr_f = arr.astype(np.float64)
                rmse = float(np.sqrt(((arr_f - resaved) ** 2).mean()))
            except Exception:
                rmse = 0.0
            features[f"jpeg_ghost_q{q}_rmse"] = rmse
            rmses.append(rmse)
        if len(rmses) >= 2 and rmses[0] > 0:
            features["jpeg_ghost_rmse_slope"] = float(rmses[0] - rmses[-1])
        else:
            features["jpeg_ghost_rmse_slope"] = 0.0
        return features
