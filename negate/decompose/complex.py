# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract complex artwork features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from skimage.feature import canny
from skimage.color import rgb2lab
from scipy.ndimage import gaussian_filter, label, sobel, binary_dilation


class ComplexFeatures:
    """Extract complex artwork features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract all complex features from the NumericImage."""
        gray, rgb = self.image.gray, self.image.color
        features: dict[str, float] = {}
        features |= self.fractal_dimension_features(gray)
        features |= self.noise_residual_autocorr_features(gray)
        features |= self.stroke_edge_roughness_features(gray)
        features |= self.color_gradient_curvature_features(rgb)
        return features

    def fractal_dimension_features(self, gray: NDArray) -> dict[str, float]:
        """Fractal dimension via box-counting features."""

        def _box_counting_dim(binary: NDArray, box_sizes: list[int] | None = None) -> float:
            if box_sizes is None:
                box_sizes = [2, 4, 8, 16, 32, 64]
            sizes, counts = [], []
            for box_size in box_sizes:
                h, w = binary.shape
                nh, nw = h // box_size, w // box_size
                if nh < 1 or nw < 1:
                    continue
                cropped = binary[: nh * box_size, : nw * box_size]
                reshaped = cropped.reshape(nh, box_size, nw, box_size)
                count = int((reshaped.any(axis=(1, 3))).sum())
                if count > 0:
                    sizes.append(box_size)
                    counts.append(count)
            if len(sizes) < 2:
                return 1.0
            coeffs = np.polyfit(np.log(1.0 / np.array(sizes, dtype=np.float64)), np.log(np.array(counts, dtype=np.float64)), 1)
            return float(coeffs[0])

        gray_f = gray if gray.max() <= 1 else gray / 255.0
        binary_gray = gray_f > np.median(gray_f)
        fd_gray = _box_counting_dim(binary_gray)
        fd_edges = _box_counting_dim(canny(gray_f))
        return {"fractal_dim_gray": fd_gray, "fractal_dim_edges": fd_edges}

    def noise_residual_autocorr_features(self, gray: NDArray) -> dict[str, float]:
        """Autocorrelation of noise residuals features."""
        gray_f = gray if gray.max() <= 1 else gray / 255.0
        smoothed = gaussian_filter(gray_f, sigma=1.5)
        residual = gray_f - smoothed
        h, w = residual.shape
        max_lag = min(64, w // 4)
        res_rows = residual[:, : w - w % 1]
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                shifted, original = residual[:, lag:], residual[:, : w - lag]
                if original.size > 0:
                    acf[lag] = float(np.corrcoef(original.ravel(), shifted.ravel())[0, 1])
        acf_tail = acf[3:]
        if len(acf_tail) > 2:
            peaks = [
                (i + 3, acf_tail[i]) for i in range(1, len(acf_tail) - 1) if acf_tail[i] > acf_tail[i - 1] and acf_tail[i] > acf_tail[i + 1]
            ]
            n_peaks = len(peaks)
            max_peak = max(p[1] for p in peaks) if peaks else 0.0
            decay_rate = float(acf[1] - acf[min(10, max_lag - 1)]) if max_lag > 10 else 0.0
        else:
            n_peaks, max_peak, decay_rate = 0, 0.0, 0.0
        return {
            "acf_n_secondary_peaks": float(n_peaks),
            "acf_max_secondary_peak": float(max_peak),
            "acf_decay_rate": decay_rate,
            "acf_lag2": float(acf[2]) if max_lag > 2 else 0.0,
            "acf_lag8": float(acf[8]) if max_lag > 8 else 0.0,
        }

    def stroke_edge_roughness_features(self, gray: NDArray) -> dict[str, float]:
        """Stroke edge roughness features."""
        gray_f = gray if gray.max() <= 1 else gray / 255.0
        edges = canny(gray_f, sigma=1.5)
        if edges.sum() < 20:
            return {
                "stroke_edge_roughness": 0.0,
                "stroke_edge_length_var": 0.0,
                "stroke_edge_curvature_mean": 0.0,
                "stroke_edge_curvature_std": 0.0,
            }
        gx, gy = sobel(gray_f, axis=1), sobel(gray_f, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        stroke_mask = mag > np.percentile(mag, 80)
        stroke_dilated = binary_dilation(stroke_mask, iterations=2)
        stroke_edges = edges & stroke_dilated
        if stroke_edges.sum() > 5:
            labeled, n_components = label(binary_dilation(stroke_edges, iterations=1))
            lengths = [n_pixels for i in range(1, min(n_components + 1, 50)) if (labeled == i).sum() > 3]
            roughness = float(stroke_edges.sum()) / (stroke_dilated.sum() + 1e-10)
            length_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0
            edge_y, edge_x = np.where(stroke_edges)
            if len(edge_y) > 10:
                dirs = np.abs(np.diff(np.arctan2(np.diff(edge_y.astype(float)), np.diff(edge_x.astype(float)))))
                curvatures = np.minimum(dirs, 2 * np.pi - dirs)
                curv_mean, curv_std = float(curvatures.mean()), float(curvatures.std())
            else:
                curv_mean, curv_std = 0.0, 0.0
        else:
            roughness, length_var, curv_mean, curv_std = 0.0, 0.0, 0.0, 0.0
        return {
            "stroke_edge_roughness": roughness,
            "stroke_edge_length_var": length_var,
            "stroke_edge_curvature_mean": curv_mean,
            "stroke_edge_curvature_std": curv_std,
        }

    def color_gradient_curvature_features(self, rgb: NDArray) -> dict[str, float]:
        """Color gradient curvature in blended regions features."""
        rgb_f = rgb / 255.0 if rgb.max() > 1 else rgb.copy()
        try:
            lab = rgb2lab(rgb_f)
        except (MemoryError, Exception):
            return {
                "color_grad_curvature_mean": 0.0,
                "color_grad_curvature_std": 0.0,
                "blend_saturation_dip": 0.0,
                "blend_lightness_dip": 0.0,
            }
        grad_l = np.sqrt(sobel(lab[:, :, 0], axis=0) ** 2 + sobel(lab[:, :, 0], axis=1) ** 2)
        grad_a = np.sqrt(sobel(lab[:, :, 1], axis=0) ** 2 + sobel(lab[:, :, 1], axis=1) ** 2)
        grad_b = np.sqrt(sobel(lab[:, :, 2], axis=0) ** 2 + sobel(lab[:, :, 2], axis=1) ** 2)
        color_grad = grad_a + grad_b
        p30, p70 = np.percentile(color_grad, 30), np.percentile(color_grad, 70)
        blend_mask = (color_grad > p30) & (color_grad < p70)
        if blend_mask.sum() < 100:
            return {
                "color_grad_curvature_mean": 0.0,
                "color_grad_curvature_std": 0.0,
                "blend_saturation_dip": 0.0,
                "blend_lightness_dip": 0.0,
            }
        h, w = rgb_f.shape[:2]
        curvatures, sat_dips, light_dips = [], [], []
        for row in range(0, h, 8):
            cols = np.where(blend_mask[row])[0]
            if len(cols) < 10:
                continue
            path_lab = lab[row, cols]
            if len(path_lab) < 3:
                continue
            start, end = path_lab[0], path_lab[-1]
            n = len(path_lab)
            t = np.linspace(0, 1, n)
            straight = start[None, :] + t[:, None] * (end - start)[None, :]
            curvatures.append(float(np.linalg.norm(path_lab - straight, axis=1).mean()))
            chroma = np.sqrt(path_lab[:, 1] ** 2 + path_lab[:, 2] ** 2)
            endpoint_chroma = (chroma[0] + chroma[-1]) / 2
            if endpoint_chroma > 1:
                sat_dips.append(float(chroma.min() / endpoint_chroma))
            endpoint_L = (path_lab[0, 0] + path_lab[-1, 0]) / 2
            if endpoint_L > 1:
                light_dips.append(float(path_lab[:, 0].min() / endpoint_L))
        return {
            "color_grad_curvature_mean": float(np.mean(curvatures)) if curvatures else 0.0,
            "color_grad_curvature_std": float(np.std(curvatures)) if curvatures else 0.0,
            "blend_saturation_dip": float(np.mean(sat_dips)) if sat_dips else 0.0,
            "blend_lightness_dip": float(np.mean(light_dips)) if light_dips else 0.0,
        }
