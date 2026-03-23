# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Style-specific feature extraction for AI-generated artwork detection.

Captures properties of human artistic craft that AI generators struggle to
replicate authentically:

Features (15 total):
    - Stroke analysis (4): direction variance, length distribution, pressure simulation
    - Color palette (4): palette size, harmony, temperature variance, saturation coherence
    - Composition (4): rule-of-thirds energy, symmetry score, focal point strength, edge density distribution
    - Micro-texture (3): grain regularity, patch-level entropy variance, brushwork periodicity
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import entropy, kurtosis
from scipy.ndimage import sobel, gaussian_filter, uniform_filter

_TARGET_SIZE = (255, 255)


def _to_gray(image: Image.Image) -> NDArray:
    """Resize and convert to float64 grayscale."""
    img = image.convert("L").resize(_TARGET_SIZE, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0


def _to_rgb(image: Image.Image) -> NDArray:
    """Resize and convert to float64 RGB [0,1]."""
    img = image.convert("RGB").resize(_TARGET_SIZE, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64) / 255.0


def _stroke_features(gray: NDArray) -> dict[str, float]:
    """Analyze brush stroke properties via gradient analysis.

    Human artists have variable stroke direction and pressure.
    AI tends to produce more uniform gradient patterns.
    """
    # Gradient direction via Sobel
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    # Only analyze pixels with significant gradient (edges/strokes)
    threshold = np.percentile(magnitude, 75)
    stroke_mask = magnitude > threshold
    stroke_directions = direction[stroke_mask]
    stroke_magnitudes = magnitude[stroke_mask]

    # Direction variance — humans have more varied stroke directions
    dir_hist = np.histogram(stroke_directions, bins=36, range=(-np.pi, np.pi))[0]
    stroke_dir_entropy = float(entropy(dir_hist + 1e-10))

    # Direction variance in local patches (16x16)
    h, w = gray.shape
    patch_size = 16
    local_dir_vars = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_dirs = direction[y:y+patch_size, x:x+patch_size]
            patch_mags = magnitude[y:y+patch_size, x:x+patch_size]
            # Weight by magnitude
            if patch_mags.sum() > 1e-10:
                weighted_var = float(np.average(
                    (patch_dirs - np.average(patch_dirs, weights=patch_mags + 1e-10))**2,
                    weights=patch_mags + 1e-10
                ))
                local_dir_vars.append(weighted_var)

    # Stroke pressure simulation — variation in gradient magnitude along strokes
    # Humans have pressure variation; AI is more uniform
    pressure_kurtosis = float(kurtosis(stroke_magnitudes)) if len(stroke_magnitudes) > 4 else 0.0

    # Stroke length distribution — via connected component-like analysis
    # Use thresholded magnitude as binary stroke map
    stroke_binary = (magnitude > threshold).astype(np.float64)
    # Row-wise and col-wise run lengths
    runs = []
    for row in stroke_binary:
        current_run = 0
        for val in row:
            if val > 0:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
    stroke_length_var = float(np.var(runs)) if len(runs) > 1 else 0.0

    return {
        "stroke_dir_entropy": stroke_dir_entropy,
        "stroke_local_dir_var": float(np.mean(local_dir_vars)) if local_dir_vars else 0.0,
        "stroke_pressure_kurtosis": pressure_kurtosis,
        "stroke_length_var": stroke_length_var,
    }


def _palette_features(rgb: NDArray) -> dict[str, float]:
    """Analyze color palette properties.

    Human artists work with deliberate, often limited palettes.
    AI generators tend to use broader, less coherent color distributions.
    """
    # Flatten to pixel colors
    pixels = rgb.reshape(-1, 3)

    # Effective palette size — number of distinct color clusters
    # Quantize to 8-level per channel and count unique
    quantized = (pixels * 7).astype(int)
    unique_colors = len(set(map(tuple, quantized)))
    max_possible = 8**3  # 512
    palette_richness = float(unique_colors / max_possible)

    # Color harmony — measure how well colors cluster in HSV hue space
    from skimage.color import rgb2hsv
    hsv = rgb2hsv(rgb)
    hue = hsv[:, :, 0].ravel()
    sat = hsv[:, :, 1].ravel()

    # Only consider saturated pixels (ignore grays)
    saturated = sat > 0.15
    if saturated.sum() > 10:
        hue_saturated = hue[saturated]
        hue_hist = np.histogram(hue_saturated, bins=36, range=(0, 1))[0]
        # Harmony = how peaked the hue distribution is (fewer peaks = more harmonious)
        hue_entropy = float(entropy(hue_hist + 1e-10))
        # Peak count — number of significant hue modes
        hue_smooth = gaussian_filter(hue_hist.astype(float), sigma=2)
        peaks = np.sum((hue_smooth[1:-1] > hue_smooth[:-2]) & (hue_smooth[1:-1] > hue_smooth[2:]))
        palette_harmony = float(peaks)
    else:
        hue_entropy = 0.0
        palette_harmony = 0.0

    # Temperature variance — warm vs cool across image regions
    # Warm = red/yellow hue, cool = blue/green
    patch_size = 32
    h, w = rgb.shape[:2]
    temps = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = rgb[y:y+patch_size, x:x+patch_size]
            # Simple temperature: red-channel dominance vs blue
            temp = float(patch[:, :, 0].mean() - patch[:, :, 2].mean())
            temps.append(temp)
    temp_variance = float(np.var(temps)) if temps else 0.0

    # Saturation coherence — how consistent saturation is across patches
    sat_patches = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_sat = hsv[y:y+patch_size, x:x+patch_size, 1]
            sat_patches.append(float(patch_sat.mean()))
    sat_coherence = float(np.std(sat_patches)) if sat_patches else 0.0

    return {
        "palette_richness": palette_richness,
        "palette_hue_entropy": hue_entropy,
        "palette_harmony_peaks": palette_harmony,
        "palette_temp_variance": temp_variance,
    }


def _composition_features(gray: NDArray) -> dict[str, float]:
    """Analyze compositional properties.

    Human artists follow compositional rules (rule of thirds, focal points).
    AI images may have different compositional statistics.
    """
    h, w = gray.shape

    # Rule of thirds — energy at third lines vs elsewhere
    third_h = [h // 3, 2 * h // 3]
    third_w = [w // 3, 2 * w // 3]
    margin = max(h, w) // 20

    # Energy at third intersections
    thirds_energy = 0.0
    for th in third_h:
        for tw in third_w:
            y_lo = max(0, th - margin)
            y_hi = min(h, th + margin)
            x_lo = max(0, tw - margin)
            x_hi = min(w, tw + margin)
            thirds_energy += float(gray[y_lo:y_hi, x_lo:x_hi].var())
    thirds_energy /= 4.0

    total_energy = float(gray.var())
    thirds_ratio = thirds_energy / (total_energy + 1e-10)

    # Symmetry — correlation between left and right halves
    left = gray[:, :w//2]
    right = gray[:, w//2:w//2 + left.shape[1]][:, ::-1]  # mirror
    if left.shape == right.shape:
        symmetry = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    else:
        symmetry = 0.0

    # Focal point strength — how concentrated the high-detail areas are
    detail = np.abs(sobel(gray, axis=0)) + np.abs(sobel(gray, axis=1))
    detail_flat = detail.ravel()
    total_detail = detail_flat.sum() + 1e-10

    # Find center of mass of detail
    yy, xx = np.mgrid[:h, :w]
    cy = float(np.sum(yy * detail) / total_detail)
    cx = float(np.sum(xx * detail) / total_detail)

    # Concentration around center of mass (lower = more focused focal point)
    dist_from_focal = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    focal_spread = float(np.sum(dist_from_focal * detail) / total_detail)
    focal_strength = 1.0 / (focal_spread + 1.0)  # inverse = stronger focal point

    # Edge density distribution — where edges are in the image (center vs periphery)
    edges = detail > np.percentile(detail, 80)
    center_mask = np.zeros_like(edges)
    ch, cw = h // 4, w // 4
    center_mask[ch:3*ch, cw:3*cw] = True
    center_edge_ratio = float(edges[center_mask].sum()) / (float(edges.sum()) + 1e-10)

    return {
        "comp_thirds_ratio": thirds_ratio,
        "comp_symmetry": symmetry,
        "comp_focal_strength": focal_strength,
        "comp_center_edge_ratio": center_edge_ratio,
    }


def _microtexture_features(gray: NDArray) -> dict[str, float]:
    """Analyze micro-texture properties.

    Human art has irregular grain from physical media (canvas, paper, pigment).
    AI images have subtly different micro-texture statistics.
    """
    h, w = gray.shape
    patch_size = 16

    # Patch-level entropy variance
    patch_entropies = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            hist = np.histogram(patch, bins=32, range=(0, 1))[0]
            patch_entropies.append(float(entropy(hist + 1e-10)))

    entropy_variance = float(np.var(patch_entropies)) if patch_entropies else 0.0

    # Grain regularity — autocorrelation of high-frequency residual
    # High-pass via difference from blurred version
    blurred = gaussian_filter(gray, sigma=1.0)
    residual = gray - blurred

    # Autocorrelation at small lags (grain regularity)
    res_flat = residual.ravel()
    if len(res_flat) > 100:
        acf_1 = float(np.corrcoef(res_flat[:-1], res_flat[1:])[0, 1])
        acf_2 = float(np.corrcoef(res_flat[:-2], res_flat[2:])[0, 1])
    else:
        acf_1, acf_2 = 0.0, 0.0

    grain_regularity = (acf_1 + acf_2) / 2.0  # higher = more regular/periodic grain

    # Brushwork periodicity — FFT of the residual, look for peaks
    fft_res = np.fft.fft2(residual)
    fft_mag = np.abs(fft_res)
    # Ratio of peak to mean (higher = more periodic = more AI-like)
    fft_peak_ratio = float(fft_mag.max() / (fft_mag.mean() + 1e-10))

    return {
        "micro_entropy_variance": entropy_variance,
        "micro_grain_regularity": grain_regularity,
        "micro_brushwork_periodicity": fft_peak_ratio,
    }


class StyleExtract:
    """Extract 15 style-specific features for artwork AI detection.

    These features target properties of human artistic craft:
    stroke patterns, color palettes, composition, and micro-texture.

    Usage:
        >>> extractor = StyleExtract()
        >>> features = extractor(pil_image)
        >>> len(features)  # 15
    """

    def __call__(self, image: Image.Image) -> dict[str, float]:
        gray = _to_gray(image)
        rgb = _to_rgb(image)

        features: dict[str, float] = {}
        features |= _stroke_features(gray)
        features |= _palette_features(rgb)
        features |= _composition_features(gray)
        features |= _microtexture_features(gray)

        return features

    def feature_names(self) -> list[str]:
        dummy = Image.new("RGB", (255, 255), color="gray")
        return list(self(dummy).keys())
