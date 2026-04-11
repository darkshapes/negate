# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract line work analysis features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from skimage.feature import canny
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from scipy.ndimage import distance_transform_edt, label, sobel, binary_dilation

from negate.decompose.numeric import NumericImage


class LineworkFeatures:
    """Extract line work analysis features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract line work analysis features from the NumericImage."""
        gray = self.image.gray
        features: dict[str, float] = {}
        features |= self.linework_features(gray)
        return features

    def linework_features(self, gray: NDArray) -> dict[str, float]:
        """Anime/illustration line work analysis features."""
        gray_f = gray if gray.max() <= 1 else gray / 255.0
        edges_tight = canny(gray_f, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
        edges_loose = canny(gray_f, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
        if edges_tight.sum() < 10:
            return {
                k: 0.0
                for k in [
                    "line_thickness_mean",
                    "line_thickness_std",
                    "line_thickness_cv",
                    "line_density",
                    "line_straightness",
                    "edge_sharpness_mean",
                    "edge_sharpness_std",
                    "medium_consistency",
                ]
            }
        dist_map = distance_transform_edt(~edges_tight)
        stroke_regions = edges_loose
        if stroke_regions.sum() > 0:
            thicknesses = dist_map[stroke_regions]  # type: ignore[misc]
            thickness_mean = float(thicknesses.mean())
            thickness_std = float(thicknesses.std())
            thickness_cv = thickness_std / (thickness_mean + 1e-10)
        else:
            thickness_mean, thickness_std, thickness_cv = 0.0, 0.0, 0.0
        line_density = float(edges_tight.sum() / edges_tight.size)
        labeled_edges, n_components = label(edges_tight)
        straightness_values = []
        for i in range(1, min(n_components + 1, 30)):
            component: NDArray = labeled_edges == i
            n_pixels = component.sum()
            if n_pixels < 5:
                continue
            ys, xs = np.where(component)
            extent = max(ys.max() - ys.min(), xs.max() - xs.min(), 1)
            straightness_values.append(n_pixels / extent)
        line_straightness = float(np.mean(straightness_values)) if straightness_values else 0.0
        gx = sobel(gray_f, axis=1)
        gy = sobel(gray_f, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_gradients = grad_mag[edges_tight]
        edge_sharpness_mean = float(edge_gradients.mean())
        edge_sharpness_std = float(edge_gradients.std())
        non_edge = ~edges_loose
        if non_edge.sum() > 100:
            h, w = gray_f.shape
            patch_vars = []
            for y in range(0, h - 16, 16):
                for x in range(0, w - 16, 16):
                    patch = gray_f[y : y + 16, x : x + 16]
                    patch_edge = edges_tight[y : y + 16, x : x + 16]
                    if patch_edge.mean() < 0.1:
                        patch_vars.append(float(patch.var()))
            medium_consistency = float(np.std(patch_vars)) if len(patch_vars) > 5 else 0.0
        else:
            medium_consistency = 0.0
        return {
            "line_thickness_mean": thickness_mean,
            "line_thickness_std": thickness_std,
            "line_thickness_cv": thickness_cv,
            "line_density": line_density,
            "line_straightness": line_straightness,
            "edge_sharpness_mean": edge_sharpness_mean,
            "edge_sharpness_std": edge_sharpness_std,
            "medium_consistency": medium_consistency,
        }

# type: ignore[reportGeneralTypeIssues]
