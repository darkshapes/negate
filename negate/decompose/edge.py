# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract edge co-occurrence features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from skimage.feature import canny
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import sobel


class EdgeFeatures:
    """Extract edge co-occurrence features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract edge co-occurrence features from the NumericImage."""
        gray = self.image.gray
        features: dict[str, float] = {}
        features |= self.edge_cooccurrence_features(gray)
        return features

    def edge_cooccurrence_features(self, gray: NDArray) -> dict[str, float]:
        """Edge co-occurrence features."""
        gray_f = gray if gray.max() <= 1 else gray / 255.0
        edges = canny(gray_f)
        gx = sobel(gray_f, axis=1)
        gy = sobel(gray_f, axis=0)
        angles = np.arctan2(gy, gx)
        n_dirs = 8
        dir_map = np.zeros_like(gray_f, dtype=np.uint8)
        dir_map[:] = ((angles + np.pi) / (2 * np.pi) * n_dirs).astype(np.uint8) % n_dirs
        dir_map[~edges] = 0
        edge_glcm = graycomatrix(dir_map, distances=[1], angles=[0, np.pi / 2], levels=n_dirs, symmetric=True, normed=True)
        features: dict[str, float] = {}
        for prop in ("contrast", "homogeneity", "energy", "correlation"):
            vals = graycoprops(edge_glcm, prop)
            features[f"edge_cooc_{prop}_mean"] = float(vals.mean())
            features[f"edge_cooc_{prop}_std"] = float(vals.std())
        return features
