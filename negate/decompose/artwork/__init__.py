# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract artwork features for AI detection."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image as PILImage
from scipy.stats import entropy, kurtosis, skew
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import canny
from skimage.restoration import estimate_sigma


class ArtworkExtract:
    """Extract artwork features for AI detection."""

    def __init__(self, image: NumericImage):
        self.image = image

    def __call__(self) -> dict[str, float]:
        """Extract all features from the NumericImage."""
        gray, rgb, hsv = self.image.gray, self.image.color, self.image.hsv
        features: dict[str, float] = {}
        features |= self.brightness_features(gray)
        features |= self.color_features(rgb, hsv)
        features |= self.texture_features(gray)
        features |= self.shape_features(gray)
        features |= self.noise_features(gray)
        features |= self.frequency_features(gray)
        features |= self.enhanced_texture_features(gray)
        features |= self.midband_frequency_features(gray)
        features |= self.patch_consistency_features(gray)
        features |= self.multiscale_lbp_features(gray)
        features |= self.gabor_features(gray)
        features |= self.wavelet_packet_features(gray)
        features |= self.edge_cooccurrence_features(gray)
        features |= self.fractal_dimension_features(gray)
        features |= self.noise_residual_autocorr_features(gray)
        features |= self.stroke_edge_roughness_features(gray)
        features |= self.color_gradient_curvature_features(rgb)
        features |= self.extended_hog_features(gray)
        features |= self.jpeg_ghost_features(rgb)
        features |= self.linework_features(gray)
        return features

    def brightness_features(self, gray: NDArray) -> dict[str, float]:
        """Mean and entropy of pixel brightness."""
        return {
            "mean_brightness": float(gray.mean()),
            "entropy_brightness": float(self.entropy(np.histogram(gray, bins=256, range=(0, 1))[0] + 1e-10)),
        }

    def color_features(self, rgb: NDArray, hsv: NDArray) -> dict[str, float]:
        """RGB and HSV histogram statistics."""
        features: dict[str, float] = {}
        for i, name in enumerate(("red", "green", "blue")):
            channel = rgb[:, :, i].ravel()
            features[f"{name}_mean"] = float(channel.mean())
            features[f"{name}_variance"] = float(channel.var())
            features[f"{name}_kurtosis"] = float(kurtosis(channel))
            features[f"{name}_skewness"] = float(skew(channel))
        rgb_flat = rgb.reshape(-1, 3)
        features["rgb_entropy"] = float(self.entropy(np.histogramdd(rgb_flat, bins=32)[0] + 1e-10))
        for i, name in enumerate(("hue", "saturation", "value")):
            channel = hsv[:, :, i].ravel()
            features[f"{name}_variance"] = float(channel.var())
            features[f"{name}_kurtosis"] = float(kurtosis(channel))
            features[f"{name}_skewness"] = float(skew(channel))
        hsv_flat = hsv.reshape(-1, 3)
        features["hsv_entropy"] = float(self.entropy(np.histogramdd(hsv_flat, bins=32)[0] + 1e-10))
        return features

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

    def shape_features(self, gray: NDArray) -> dict[str, float]:
        """HOG statistics and edge length."""
        from PIL import Image as PilImage

        hog_features = canny(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        gray_uint8 = (gray * 255).astype(np.uint8)
        edges_array = np.asarray(PilImage.fromarray(gray_uint8).convert("L").point(lambda x: 0 if x < 128 else 255, "1"))
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
        sigma = estimate_sigma(gray)
        noise = gray - np.clip(gray, gray.mean() - 2 * sigma, gray.mean() + 2 * sigma)
        noise_hist = np.histogram(noise.ravel(), bins=256)[0]
        noise_ent = float(self.entropy(noise_hist + 1e-10))
        signal_power = float(gray.var())
        noise_power = float(sigma**2) if sigma > 0 else 1e-10
        snr = float(10 * np.log10(signal_power / noise_power + 1e-10))
        return {"noise_entropy": noise_ent, "snr": snr}

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
        dct_coeffs = dctn(gray, type=2, norm="ortho")
        dct_mag = np.abs(dct_coeffs)
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
                ac_energy = float((dct_block**2).sum() - dct_block[0, 0] ** 2)
                block_energies.append(ac_energy)
        block_energies = np.array(block_energies)
        features["dct_block_energy_mean"] = float(block_energies.mean())
        features["dct_block_energy_std"] = float(block_energies.std())
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

    def gabor_features(self, gray: NDArray) -> dict[str, float]:
        """Gabor filter bank features."""
        from skimage.filters import gabor

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

    def edge_cooccurrence_features(self, gray: NDArray) -> dict[str, float]:
        """Edge co-occurrence features."""
        gray_f = gray if gray.max() <= 1 else gray / 255.0
        edges = canny(gray_f)
        from scipy.ndimage import sobel

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

    def fractal_dimension_features(self, gray: NDArray) -> dict[str, float]:
        """Fractal dimension via box-counting features."""
        from skimage.feature import canny

        def _box_counting_dim(binary: NDArray, box_sizes: list[int] | None = None) -> float:
            if box_sizes is None:
                box_sizes = [2, 4, 8, 16, 32, 64]
            sizes = []
            counts = []
            for box_size in box_sizes:
                h, w = binary.shape
                nh = h // box_size
                nw = w // box_size
                if nh < 1 or nw < 1:
                    continue
                cropped = binary[: nh * box_size, : nw * box_size]
                reshaped = cropped.reshape(nh, box_size, nw, box_size)
                box_has_pixel = reshaped.any(axis=(1, 3))
                count = int(box_has_pixel.sum())
                if count > 0:
                    sizes.append(box_size)
                    counts.append(count)
            if len(sizes) < 2:
                return 1.0
            log_sizes = np.log(1.0 / np.array(sizes, dtype=np.float64))
            log_counts = np.log(np.array(counts, dtype=np.float64))
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            return float(coeffs[0])

        gray_f = gray if gray.max() <= 1 else gray / 255.0
        binary_gray = gray_f > np.median(gray_f)
        fd_gray = _box_counting_dim(binary_gray)
        edges = canny(gray_f)
        fd_edges = _box_counting_dim(edges)
        return {"fractal_dim_gray": fd_gray, "fractal_dim_edges": fd_edges}

    def noise_residual_autocorr_features(self, gray: NDArray) -> dict[str, float]:
        """Autocorrelation of noise residuals features."""
        from scipy.ndimage import gaussian_filter

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
                shifted = residual[:, lag:]
                original = residual[:, : w - lag]
                if original.size > 0:
                    acf[lag] = float(np.corrcoef(original.ravel(), shifted.ravel())[0, 1])
        acf_tail = acf[3:]
        if len(acf_tail) > 2:
            peaks = []
            for i in range(1, len(acf_tail) - 1):
                if acf_tail[i] > acf_tail[i - 1] and acf_tail[i] > acf_tail[i + 1]:
                    peaks.append((i + 3, acf_tail[i]))
            n_peaks = len(peaks)
            max_peak = max(p[1] for p in peaks) if peaks else 0.0
            decay_rate = float(acf[1] - acf[min(10, max_lag - 1)]) if max_lag > 10 else 0.0
        else:
            n_peaks = 0
            max_peak = 0.0
            decay_rate = 0.0
        return {
            "acf_n_secondary_peaks": float(n_peaks),
            "acf_max_secondary_peak": float(max_peak),
            "acf_decay_rate": decay_rate,
            "acf_lag2": float(acf[2]) if max_lag > 2 else 0.0,
            "acf_lag8": float(acf[8]) if max_lag > 8 else 0.0,
        }

    def stroke_edge_roughness_features(self, gray: NDArray) -> dict[str, float]:
        """Stroke edge roughness features."""
        from scipy.ndimage import sobel, binary_dilation

        gray_f = gray if gray.max() <= 1 else gray / 255.0
        edges = canny(gray_f, sigma=1.5)
        if edges.sum() < 20:
            return {
                "stroke_edge_roughness": 0.0,
                "stroke_edge_length_var": 0.0,
                "stroke_edge_curvature_mean": 0.0,
                "stroke_edge_curvature_std": 0.0,
            }
        gx = sobel(gray_f, axis=1)
        gy = sobel(gray_f, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        stroke_mask = mag > np.percentile(mag, 80)
        stroke_dilated = binary_dilation(stroke_mask, iterations=2)
        stroke_edges = edges & stroke_dilated
        if stroke_edges.sum() > 5:
            from scipy.ndimage import label

            labeled, n_components = label(binary_dilation(stroke_edges, iterations=1))
            lengths = []
            for i in range(1, min(n_components + 1, 50)):
                component = labeled == i
                n_pixels = component.sum()
                if n_pixels > 3:
                    lengths.append(n_pixels)
            roughness = float(stroke_edges.sum()) / (stroke_dilated.sum() + 1e-10)
            length_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0
            edge_y, edge_x = np.where(stroke_edges)
            if len(edge_y) > 10:
                dirs = np.arctan2(np.diff(edge_y.astype(float)), np.diff(edge_x.astype(float)))
                curvatures = np.abs(np.diff(dirs))
                curvatures = np.minimum(curvatures, 2 * np.pi - curvatures)
                curv_mean = float(curvatures.mean())
                curv_std = float(curvatures.std())
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
        from skimage.color import rgb2lab

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
        p30 = np.percentile(color_grad, 30)
        p70 = np.percentile(color_grad, 70)
        blend_mask = (color_grad > p30) & (color_grad < p70)
        if blend_mask.sum() < 100:
            return {
                "color_grad_curvature_mean": 0.0,
                "color_grad_curvature_std": 0.0,
                "blend_saturation_dip": 0.0,
                "blend_lightness_dip": 0.0,
            }
        h, w = rgb_f.shape[:2]
        curvatures = []
        sat_dips = []
        light_dips = []
        for row in range(0, h, 8):
            cols = np.where(blend_mask[row])[0]
            if len(cols) < 10:
                continue
            path_lab = lab[row, cols]
            if len(path_lab) < 3:
                continue
            start = path_lab[0]
            end = path_lab[-1]
            n = len(path_lab)
            t = np.linspace(0, 1, n)
            straight = start[None, :] + t[:, None] * (end - start)[None, :]
            deviations = np.linalg.norm(path_lab - straight, axis=1)
            curvatures.append(float(deviations.mean()))
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

    def extended_hog_features(self, gray: NDArray) -> dict[str, float]:
        """Extended HOG features."""
        from skimage.feature import hog

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
        from io import BytesIO

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

    def linework_features(self, gray: NDArray) -> dict[str, float]:
        """Anime/illustration line work analysis features."""
        from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern
        from skimage.feature import sobel
        from scipy.ndimage import distance_transform_edt, label, binary_dilation

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
            thicknesses = dist_map[stroke_regions]
            thickness_mean = float(thicknesses.mean())
            thickness_std = float(thicknesses.std())
            thickness_cv = thickness_std / (thickness_mean + 1e-10)
        else:
            thickness_mean, thickness_std, thickness_cv = 0.0, 0.0, 0.0
        line_density = float(edges_tight.sum() / edges_tight.size)
        labeled_edges, n_components = label(edges_tight)
        straightness_values = []
        for i in range(1, min(n_components + 1, 30)):
            component = labeled_edges == i
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
