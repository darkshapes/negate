# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Artwork feature extraction for AI-generated image detection.

Implements the 39-feature extraction pipeline from:
    Li & Stamp, "Detecting AI-generated Artwork", arXiv:2504.07078, 2025.

Extended with:
    - Dedicated frequency analysis branch (FFT/DCT) for spectral fingerprints
    - Enhanced GLCM (multi-angle/distance) per Nirob et al. (2026)
    - Full LBP histogram features per Nirob et al. (2026)
    - Mid-band frequency analysis per FIRE (CVPR 2025)
    - Patch-level consistency features per CINEMAE (2025)
    - Multi-scale LBP (8): R=3/P=24 coarse texture + per-scale stats
    - Gabor filter bank (18): 4 freq x 4 orient energy + summary stats
    - Wavelet packet statistics (12): 2-level Haar detail coefficients
    - Color coherence vectors (6): coherent/incoherent pixel ratios per channel
    - Edge co-occurrence (8): edge-direction GLCM properties
    - Fractal dimension (2): box-counting on grayscale + edge map
    - Extended HOG (6): multi-scale HOG + cross-scale ratios
    - JPEG ghost detection (4): recompression RMSE at multiple quality levels

Features are grouped into 16 categories:
    - Brightness (2): mean, entropy
    - Color (23): RGB/HSV histogram statistics
    - Texture (6): GLCM + LBP
    - Shape (6): HOG + edge length
    - Noise (2): noise entropy, SNR
    - Frequency (10): FFT/DCT spectral analysis
    - Enhanced texture (14): multi-angle GLCM, full LBP histogram, DCT block stats
    - Patch consistency (6): cross-patch feature variance (CINEMAE-inspired)
    - Mid-band frequency (4): fine-grained radial band analysis
    - Multi-scale LBP (8): coarse texture descriptors
    - Gabor filter bank (18): oriented frequency responses
    - Wavelet packets (12): Haar detail coefficient statistics
    - Color coherence (6): spatial color consistency
    - Edge co-occurrence (8): edge direction relationships
    - Fractal dimension (2): complexity measures
    - Extended HOG (6): multi-scale gradient histograms
    - JPEG ghosts (4): recompression artifacts
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import entropy, kurtosis, skew
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


_TARGET_SIZE = (255, 255)


def _to_array(image: Image.Image) -> NDArray:
    """Resize to 255x255 and convert to float64 numpy array."""
    image = image.convert("RGB").resize(_TARGET_SIZE, Image.BICUBIC)
    return np.asarray(image, dtype=np.float64)


def _brightness_features(gray: NDArray) -> dict[str, float]:
    """Mean and entropy of pixel brightness."""
    return {
        "mean_brightness": float(gray.mean()),
        "entropy_brightness": float(entropy(np.histogram(gray, bins=256, range=(0, 1))[0] + 1e-10)),
    }


def _color_features(rgb: NDArray) -> dict[str, float]:
    """RGB and HSV histogram statistics (23 features)."""
    features: dict[str, float] = {}

    # RGB: mean, variance, kurtosis, skewness per channel + entropy
    for i, name in enumerate(("red", "green", "blue")):
        channel = rgb[:, :, i].ravel()
        features[f"{name}_mean"] = float(channel.mean())
        features[f"{name}_variance"] = float(channel.var())
        features[f"{name}_kurtosis"] = float(kurtosis(channel))
        features[f"{name}_skewness"] = float(skew(channel))

    # RGB entropy (joint)
    rgb_flat = rgb.reshape(-1, 3)
    rgb_hist = np.histogramdd(rgb_flat, bins=32)[0]
    features["rgb_entropy"] = float(entropy(rgb_hist.ravel() + 1e-10))

    # HSV: variance, kurtosis, skewness per channel + entropy
    hsv = rgb2hsv(rgb / 255.0 if rgb.max() > 1 else rgb)
    for i, name in enumerate(("hue", "saturation", "value")):
        channel = hsv[:, :, i].ravel()
        features[f"{name}_variance"] = float(channel.var())
        features[f"{name}_kurtosis"] = float(kurtosis(channel))
        features[f"{name}_skewness"] = float(skew(channel))

    hsv_flat = hsv.reshape(-1, 3)
    hsv_hist = np.histogramdd(hsv_flat, bins=32)[0]
    features["hsv_entropy"] = float(entropy(hsv_hist.ravel() + 1e-10))

    return features


def _texture_features(gray: NDArray) -> dict[str, float]:
    """GLCM and LBP texture features (6 features)."""
    # GLCM requires uint8
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)

    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    features: dict[str, float] = {
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
    }

    # LBP
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    features["lbp_entropy"] = float(entropy(np.histogram(lbp, bins=10)[0] + 1e-10))
    features["lbp_variance"] = float(lbp.var())

    return features


def _shape_features(gray: NDArray) -> dict[str, float]:
    """HOG statistics and edge length (6 features)."""
    from skimage.feature import hog, canny

    # HOG
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    features: dict[str, float] = {
        "hog_mean": float(hog_features.mean()),
        "hog_variance": float(hog_features.var()),
        "hog_kurtosis": float(kurtosis(hog_features)),
        "hog_skewness": float(skew(hog_features)),
        "hog_entropy": float(entropy(np.histogram(hog_features, bins=50)[0] + 1e-10)),
    }

    # Edge length via Canny
    edges = canny(gray if gray.max() <= 1 else gray / 255.0)
    features["edgelen"] = float(edges.sum())

    return features


def _noise_features(gray: NDArray) -> dict[str, float]:
    """Noise entropy and signal-to-noise ratio (2 features)."""
    from skimage.restoration import estimate_sigma

    # Estimate noise
    sigma = estimate_sigma(gray)
    noise = gray - np.clip(gray, gray.mean() - 2 * sigma, gray.mean() + 2 * sigma)

    noise_hist = np.histogram(noise.ravel(), bins=256)[0]
    noise_ent = float(entropy(noise_hist + 1e-10))

    # SNR
    signal_power = float(gray.var())
    noise_power = float(sigma ** 2) if sigma > 0 else 1e-10
    snr = float(10 * np.log10(signal_power / noise_power + 1e-10))

    return {
        "noise_entropy": noise_ent,
        "snr": snr,
    }


def _frequency_features(gray: NDArray) -> dict[str, float]:
    """FFT and DCT spectral analysis features (10 features).

    AI generators leave characteristic signatures in the frequency domain
    due to upsampling layers and attention patterns. This branch captures
    those patterns independently of pixel-space features.
    """
    from scipy.fft import dctn
    from numpy.fft import fftfreq

    h, w = gray.shape

    # 2D FFT analysis
    fft_2d = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft_2d)
    magnitude = np.abs(fft_shift)
    log_mag = np.log(magnitude + 1e-10)
    phase = np.angle(fft_shift)

    center_h, center_w = h // 2, w // 2

    # Radial frequency bands (low/mid/high)
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
    max_r = np.sqrt(center_h ** 2 + center_w ** 2)

    low_mask = radius < max_r * 0.2
    mid_mask = (radius >= max_r * 0.2) & (radius < max_r * 0.6)
    high_mask = radius >= max_r * 0.6

    total_energy = float((magnitude ** 2).sum() + 1e-10)
    low_energy = float((magnitude[low_mask] ** 2).sum())
    mid_energy = float((magnitude[mid_mask] ** 2).sum())
    high_energy = float((magnitude[high_mask] ** 2).sum())

    # Spectral centroid (center of mass of frequency distribution)
    row_freqs = fftfreq(h)[:, None] * np.ones((1, w))
    col_freqs = np.ones((h, 1)) * fftfreq(w)[None, :]
    spectral_centroid = float(
        (np.sum(log_mag * np.abs(row_freqs)) + np.sum(log_mag * np.abs(col_freqs)))
        / (log_mag.sum() * 2 + 1e-10)
    )

    # DCT analysis — captures compression and generation artifacts
    dct_coeffs = dctn(gray, type=2, norm="ortho")
    dct_mag = np.abs(dct_coeffs)

    # Ratio of AC to DC energy (how much detail vs flat)
    dc_energy = float(dct_mag[0, 0] ** 2)
    ac_energy = float((dct_mag ** 2).sum() - dc_energy)

    # Phase coherence — AI images often have more regular phase patterns
    phase_std = float(phase.std())

    return {
        "fft_low_energy_ratio": low_energy / total_energy,
        "fft_mid_energy_ratio": mid_energy / total_energy,
        "fft_high_energy_ratio": high_energy / total_energy,
        "fft_spectral_centroid": spectral_centroid,
        "fft_log_mag_mean": float(log_mag.mean()),
        "fft_log_mag_std": float(log_mag.std()),
        "fft_phase_std": phase_std,
        "dct_ac_dc_ratio": ac_energy / (dc_energy + 1e-10),
        "dct_high_freq_energy": float((dct_mag[h // 2:, w // 2:] ** 2).sum() / (dct_mag ** 2).sum()),
        "dct_sparsity": float((dct_mag < 0.01 * dct_mag.max()).mean()),
    }


def _enhanced_texture_features(gray: NDArray) -> dict[str, float]:
    """Extended GLCM + full LBP histogram + block DCT (14 features).

    Per Nirob et al. (2026): fusing multiple GLCM angles/distances and
    full LBP histogram distributions significantly improves detection.
    """
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)

    # Multi-angle GLCM: 4 angles × 2 distances, averaged per property
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    distances = [1, 3]
    glcm = graycomatrix(gray_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    features: dict[str, float] = {}
    for prop in ("contrast", "correlation", "energy", "homogeneity"):
        vals = graycoprops(glcm, prop)
        features[f"glcm_multi_{prop}_mean"] = float(vals.mean())
        features[f"glcm_multi_{prop}_std"] = float(vals.std())

    # Full LBP histogram (10-bin uniform + variance of spatial LBP)
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    features["lbp_hist_kurtosis"] = float(kurtosis(lbp_hist))
    features["lbp_hist_skew"] = float(skew(lbp_hist))
    features["lbp_hist_max"] = float(lbp_hist.max())

    # Multi-scale LBP: R=2, P=16 captures coarser texture
    lbp_coarse = local_binary_pattern(gray_uint8, P=16, R=2, method="uniform")
    features["lbp_coarse_entropy"] = float(entropy(np.histogram(lbp_coarse, bins=18)[0] + 1e-10))

    # Block-level DCT statistics (8x8 blocks, like JPEG)
    from scipy.fft import dctn
    h, w = gray.shape
    block_size = 8
    block_energies = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            dct_block = dctn(block, type=2, norm="ortho")
            # Energy in AC coefficients (exclude DC at [0,0])
            ac_energy = float((dct_block ** 2).sum() - dct_block[0, 0] ** 2)
            block_energies.append(ac_energy)

    block_energies = np.array(block_energies)
    features["dct_block_energy_mean"] = float(block_energies.mean())
    features["dct_block_energy_std"] = float(block_energies.std())

    return features


def _midband_frequency_features(gray: NDArray) -> dict[str, float]:
    """Mid-band frequency analysis (4 features).

    Per FIRE (CVPR 2025): diffusion models specifically fail to accurately
    reconstruct mid-band frequency information. This measures the mid-band
    energy distribution relative to natural image expectations.
    """
    h, w = gray.shape
    fft_2d = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft_2d)
    magnitude = np.abs(fft_shift)

    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
    max_r = np.sqrt(center_h ** 2 + center_w ** 2)

    # Fine-grained radial bands (5 bands instead of 3)
    bands = [(0, 0.1), (0.1, 0.25), (0.25, 0.45), (0.45, 0.7), (0.7, 1.0)]
    band_energies = []
    for lo, hi in bands:
        mask = (radius >= max_r * lo) & (radius < max_r * hi)
        band_energies.append(float((magnitude[mask] ** 2).sum()))

    total = sum(band_energies) + 1e-10
    band_ratios = [e / total for e in band_energies]

    # Natural images follow approximate 1/f power law
    # Deviation from 1/f in mid-bands is a strong AI signal
    expected_ratios = np.array([0.65, 0.20, 0.10, 0.035, 0.015])  # approximate 1/f
    actual_ratios = np.array(band_ratios)
    deviation = actual_ratios - expected_ratios

    return {
        "midband_energy_ratio": float(band_ratios[2]),  # 0.25-0.45 band specifically
        "midband_deviation": float(deviation[2]),  # deviation from expected in midband
        "spectral_slope_deviation": float(np.std(deviation)),  # overall 1/f deviation
        "high_to_mid_ratio": float(band_ratios[4] / (band_ratios[2] + 1e-10)),  # high/mid balance
    }


def _patch_consistency_features(gray: NDArray) -> dict[str, float]:
    """Cross-patch consistency features (6 features).

    Per CINEMAE (2025): real images have consistent patch-to-context
    relationships that AI images subtly violate. We measure variance
    of per-patch statistics across the image.
    """
    h, w = gray.shape
    patch_size = 32
    n_patches = 0

    patch_means = []
    patch_stds = []
    patch_edges = []
    patch_freq_centroids = []

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            patch_means.append(float(patch.mean()))
            patch_stds.append(float(patch.std()))

            # Edge density per patch
            from skimage.feature import canny
            edges = canny(patch)
            patch_edges.append(float(edges.mean()))

            # Frequency centroid per patch
            fft_p = np.fft.fft2(patch)
            mag_p = np.abs(fft_p)
            freqs = np.fft.fftfreq(patch_size)
            freq_grid = np.sqrt(freqs[:, None] ** 2 + freqs[None, :] ** 2)
            centroid = float(np.sum(mag_p * freq_grid) / (mag_p.sum() + 1e-10))
            patch_freq_centroids.append(centroid)
            n_patches += 1

    if n_patches < 4:
        return {k: 0.0 for k in [
            "patch_mean_cv", "patch_std_cv", "patch_edge_cv",
            "patch_freq_centroid_cv", "patch_freq_centroid_range",
            "patch_coherence_score",
        ]}

    # Coefficient of variation (std/mean) for each patch-level statistic
    # Higher CV = more inconsistency across patches
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
        "patch_coherence_score": float(np.corrcoef(patch_means, patch_stds)[0, 1])
            if len(patch_means) > 2 else 0.0,
    }


def _multiscale_lbp_features(gray: NDArray) -> dict[str, float]:
    """Multi-scale LBP features (8 features).

    Extends existing LBP (R=1,P=8 and R=2,P=16) with R=3,P=24 for coarser
    texture, and computes per-scale summary statistics.
    """
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
    features: dict[str, float] = {}

    scales = [
        (8, 1, "s1"),
        (16, 2, "s2"),
        (24, 3, "s3"),
    ]

    for p, r, label in scales:
        lbp = local_binary_pattern(gray_uint8, P=p, R=r, method="uniform")
        n_bins = p + 2  # uniform LBP has P+2 bins
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

        features[f"mslbp_{label}_mean"] = float(lbp.mean())
        features[f"mslbp_{label}_var"] = float(lbp.var())

        # Only add entropy and uniformity for the new R=3 scale to avoid
        # duplicating stats already captured by _texture_features and _enhanced_texture_features
        if r == 3:
            features[f"mslbp_{label}_entropy"] = float(entropy(hist + 1e-10))
            features[f"mslbp_{label}_uniformity"] = float(hist.max())

    return features


def _gabor_features(gray: NDArray) -> dict[str, float]:
    """Gabor filter bank features (18 features).

    4 frequencies x 4 orientations = 16 mean energy values,
    plus overall mean and std across all filter responses.
    """
    from skimage.filters import gabor

    features: dict[str, float] = {}
    all_energies = []

    freqs = [0.1, 0.2, 0.3, 0.4]
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for fi, freq in enumerate(freqs):
        for ti, theta in enumerate(thetas):
            filt_real, filt_imag = gabor(gray, frequency=freq, theta=theta)
            energy = float(np.sqrt(filt_real ** 2 + filt_imag ** 2).mean())
            features[f"gabor_f{fi}_t{ti}_energy"] = energy
            all_energies.append(energy)

    all_e = np.array(all_energies)
    features["gabor_mean_energy"] = float(all_e.mean())
    features["gabor_std_energy"] = float(all_e.std())

    return features


def _wavelet_packet_features(gray: NDArray) -> dict[str, float]:
    """Wavelet packet statistics (12 features).

    2-level Haar wavelet decomposition. For each detail subband
    (LH, HL, HH at levels 1 and 2): mean and std of coefficients.
    """
    import pywt

    coeffs = pywt.wavedec2(gray, "haar", level=2)
    # coeffs: [cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]
    features: dict[str, float] = {}

    subband_names = ["LH", "HL", "HH"]
    for level_idx, level in enumerate([1, 2]):
        # coeffs index: level 2 details are at index 1, level 1 at index 2
        detail_tuple = coeffs[len(coeffs) - level]
        for sb_idx, sb_name in enumerate(subband_names):
            c = detail_tuple[sb_idx]
            prefix = f"wvt_L{level}_{sb_name}"
            features[f"{prefix}_mean"] = float(np.abs(c).mean())
            features[f"{prefix}_std"] = float(c.std())

    return features


def _color_coherence_features(rgb: NDArray) -> dict[str, float]:
    """Color coherence vector features (6 features).

    For each RGB channel: ratio of coherent pixels (in large connected
    regions) to incoherent (small isolated regions). Threshold tau=25.
    """
    from scipy.ndimage import label as ndlabel

    features: dict[str, float] = {}
    tau = 25

    rgb_uint8 = rgb.astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)

    for i, name in enumerate(("red", "green", "blue")):
        channel = rgb_uint8[:, :, i]
        # Quantize to reduce noise: 64 bins
        quantized = (channel // 4).astype(np.uint8)

        # For a representative threshold, use median intensity
        median_val = np.median(quantized)
        binary = quantized >= median_val

        labeled, n_components = ndlabel(binary)
        if n_components == 0:
            features[f"ccv_{name}_coherent_ratio"] = 0.0
            features[f"ccv_{name}_incoherent_ratio"] = 1.0
            continue

        total_pixels = float(binary.sum())
        if total_pixels < 1:
            features[f"ccv_{name}_coherent_ratio"] = 0.0
            features[f"ccv_{name}_incoherent_ratio"] = 1.0
            continue

        coherent = 0.0
        for comp_id in range(1, n_components + 1):
            comp_size = float((labeled == comp_id).sum())
            if comp_size >= tau:
                coherent += comp_size

        incoherent = total_pixels - coherent
        features[f"ccv_{name}_coherent_ratio"] = coherent / (total_pixels + 1e-10)
        features[f"ccv_{name}_incoherent_ratio"] = incoherent / (total_pixels + 1e-10)

    return features


def _edge_cooccurrence_features(gray: NDArray) -> dict[str, float]:
    """Edge co-occurrence features (8 features).

    Compute Canny edges, quantize gradient directions into bins,
    build a GLCM of edge directions, and extract standard properties.
    """
    from skimage.feature import canny

    gray_f = gray if gray.max() <= 1 else gray / 255.0
    edges = canny(gray_f)

    # Compute gradient directions using Sobel
    from scipy.ndimage import sobel
    gx = sobel(gray_f, axis=1)
    gy = sobel(gray_f, axis=0)
    angles = np.arctan2(gy, gx)  # -pi to pi

    # Quantize angles to 8 direction bins (only at edge pixels)
    n_dirs = 8
    # Map -pi..pi to 0..n_dirs
    dir_map = np.zeros_like(gray_f, dtype=np.uint8)
    dir_map[:] = ((angles + np.pi) / (2 * np.pi) * n_dirs).astype(np.uint8) % n_dirs

    # Mask to edge pixels only
    dir_map[~edges] = 0

    # Build edge direction co-occurrence (GLCM on direction map at edge pixels)
    # Use graycomatrix on the direction map
    edge_glcm = graycomatrix(
        dir_map, distances=[1], angles=[0, np.pi / 2],
        levels=n_dirs, symmetric=True, normed=True,
    )

    features: dict[str, float] = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(edge_glcm, prop)
        features[f"edge_cooc_{prop}_mean"] = float(vals.mean())
        features[f"edge_cooc_{prop}_std"] = float(vals.std())

    return features


def _fractal_dimension_features(gray: NDArray) -> dict[str, float]:
    """Fractal dimension via box-counting (2 features).

    Estimates fractal dimension of the grayscale image (thresholded)
    and the edge map. Real artwork often has different fractal
    characteristics than AI-generated images.
    """
    from skimage.feature import canny

    def _box_counting_dim(binary: NDArray, box_sizes: list[int] | None = None) -> float:
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16, 32, 64]

        sizes = []
        counts = []
        for box_size in box_sizes:
            h, w = binary.shape
            # Count boxes needed to cover all True pixels
            # Reshape into grid of boxes
            nh = h // box_size
            nw = w // box_size
            if nh < 1 or nw < 1:
                continue
            cropped = binary[:nh * box_size, :nw * box_size]
            # Reshape and check if any pixel in each box is True
            reshaped = cropped.reshape(nh, box_size, nw, box_size)
            box_has_pixel = reshaped.any(axis=(1, 3))
            count = int(box_has_pixel.sum())
            if count > 0:
                sizes.append(box_size)
                counts.append(count)

        if len(sizes) < 2:
            return 1.0  # degenerate case

        log_sizes = np.log(1.0 / np.array(sizes, dtype=np.float64))
        log_counts = np.log(np.array(counts, dtype=np.float64))

        # Linear regression: slope = fractal dimension
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return float(coeffs[0])

    gray_f = gray if gray.max() <= 1 else gray / 255.0

    # Threshold grayscale at median
    binary_gray = gray_f > np.median(gray_f)
    fd_gray = _box_counting_dim(binary_gray)

    # Edge map fractal dimension
    edges = canny(gray_f)
    fd_edges = _box_counting_dim(edges)

    return {
        "fractal_dim_gray": fd_gray,
        "fractal_dim_edges": fd_edges,
    }


def _extended_hog_features(gray: NDArray) -> dict[str, float]:
    """Extended HOG features (6 features).

    HOG at two cell sizes (8x8 fine, 32x32 coarse), plus cross-scale
    energy ratio and angular histogram entropy at each scale.
    """
    from skimage.feature import hog

    features: dict[str, float] = {}

    # Fine scale: 8x8 cells
    hog_fine = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    fine_energy = float((hog_fine ** 2).sum())
    fine_hist = np.histogram(hog_fine, bins=50)[0]
    features["hog_fine_energy"] = fine_energy
    features["hog_fine_entropy"] = float(entropy(fine_hist + 1e-10))

    # Coarse scale: 32x32 cells
    hog_coarse = hog(gray, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    coarse_energy = float((hog_coarse ** 2).sum())
    coarse_hist = np.histogram(hog_coarse, bins=50)[0]
    features["hog_coarse_energy"] = coarse_energy
    features["hog_coarse_entropy"] = float(entropy(coarse_hist + 1e-10))

    # Cross-scale ratio
    features["hog_fine_coarse_ratio"] = fine_energy / (coarse_energy + 1e-10)

    # Overall angular dispersion
    features["hog_energy_ratio_to_mean"] = fine_energy / (float(hog_fine.mean()) + 1e-10)

    return features


def _jpeg_ghost_features(rgb: NDArray) -> dict[str, float]:
    """JPEG ghost detection features (4 features).

    Resave image at different quality levels and measure RMSE between
    original and resaved. AI and real images respond differently to
    recompression artifacts.
    """
    from io import BytesIO

    arr = rgb.astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
    features: dict[str, float] = {}
    rmses = []

    for q in [50, 70, 90]:
        try:
            buf = BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            resaved = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
            arr_f = arr.astype(np.float64)
            rmse = float(np.sqrt(((arr_f - resaved) ** 2).mean()))
        except Exception:
            rmse = 0.0
        features[f"jpeg_ghost_q{q}_rmse"] = rmse
        rmses.append(rmse)

    # Slope of RMSE across quality levels (how much quality matters)
    if len(rmses) >= 2 and rmses[0] > 0:
        features["jpeg_ghost_rmse_slope"] = float(rmses[0] - rmses[-1])
    else:
        features["jpeg_ghost_rmse_slope"] = 0.0

    return features


def _noise_residual_autocorr_features(gray: NDArray) -> dict[str, float]:
    """Autocorrelation of noise residuals (5 features).

    Canvas texture produces periodic peaks in the autocorrelation at thread
    spacing intervals. Generator artifacts produce peaks at architecture-specific
    frequencies. Real digital art has smooth monotonic decay.
    """
    from scipy.ndimage import gaussian_filter

    gray_f = gray if gray.max() <= 1 else gray / 255.0
    # Extract noise residual
    smoothed = gaussian_filter(gray_f, sigma=1.5)
    residual = gray_f - smoothed

    h, w = residual.shape
    # Compute 1D autocorrelation along rows (averaged)
    max_lag = min(64, w // 4)
    res_rows = residual[:, :w - w % 1]  # trim for alignment
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf[lag] = 1.0
        else:
            shifted = residual[:, lag:]
            original = residual[:, :w - lag]
            if original.size > 0:
                acf[lag] = float(np.corrcoef(original.ravel(), shifted.ravel())[0, 1])

    # Look for secondary peaks (evidence of periodic structure)
    # Skip lag 0 and first few lags (always high)
    acf_tail = acf[3:]
    if len(acf_tail) > 2:
        # Find peaks
        peaks = []
        for i in range(1, len(acf_tail) - 1):
            if acf_tail[i] > acf_tail[i - 1] and acf_tail[i] > acf_tail[i + 1]:
                peaks.append((i + 3, acf_tail[i]))

        n_peaks = len(peaks)
        max_peak = max(p[1] for p in peaks) if peaks else 0.0
        # Decay rate: how fast ACF drops
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


def _stroke_edge_roughness_features(gray: NDArray) -> dict[str, float]:
    """Stroke edge roughness (4 features).

    Physical brush strokes have characteristic edge roughness from bristles.
    AI strokes tend to have smoother, more regular edges.
    Uses fractal dimension of edge contours within high-gradient regions.
    """
    from scipy.ndimage import sobel, binary_dilation
    from skimage.feature import canny

    gray_f = gray if gray.max() <= 1 else gray / 255.0

    # Detect edges
    edges = canny(gray_f, sigma=1.5)
    if edges.sum() < 20:
        return {
            "stroke_edge_roughness": 0.0,
            "stroke_edge_length_var": 0.0,
            "stroke_edge_curvature_mean": 0.0,
            "stroke_edge_curvature_std": 0.0,
        }

    # Find strong gradient regions (likely strokes)
    gx = sobel(gray_f, axis=1)
    gy = sobel(gray_f, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    stroke_mask = mag > np.percentile(mag, 80)

    # Dilate stroke mask and intersect with edges = stroke edges
    stroke_dilated = binary_dilation(stroke_mask, iterations=2)
    stroke_edges = edges & stroke_dilated

    # Edge roughness: ratio of edge pixels to the convex area they span
    # More rough = more edge pixels per unit area
    if stroke_edges.sum() > 5:
        from scipy.ndimage import label
        labeled, n_components = label(binary_dilation(stroke_edges, iterations=1))
        lengths = []
        for i in range(1, min(n_components + 1, 50)):  # cap at 50 components
            component = (labeled == i)
            n_pixels = component.sum()
            if n_pixels > 3:
                lengths.append(n_pixels)

        roughness = float(stroke_edges.sum()) / (stroke_dilated.sum() + 1e-10)
        length_var = float(np.var(lengths)) if len(lengths) > 1 else 0.0

        # Local curvature via direction changes along edges
        edge_y, edge_x = np.where(stroke_edges)
        if len(edge_y) > 10:
            # Sample direction changes
            dirs = np.arctan2(np.diff(edge_y.astype(float)), np.diff(edge_x.astype(float)))
            curvatures = np.abs(np.diff(dirs))
            curvatures = np.minimum(curvatures, 2 * np.pi - curvatures)  # wrap
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


def _color_gradient_curvature_features(rgb: NDArray) -> dict[str, float]:
    """Color gradient curvature in blended regions (4 features).

    Physical paint mixing (subtractive) curves through lower saturation/luminance.
    Digital blending produces straighter paths in color space.
    """
    from skimage.color import rgb2lab
    from scipy.ndimage import sobel

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

    # Find blended regions: moderate gradient magnitude
    grad_l = np.sqrt(sobel(lab[:, :, 0], axis=0) ** 2 + sobel(lab[:, :, 0], axis=1) ** 2)
    grad_a = np.sqrt(sobel(lab[:, :, 1], axis=0) ** 2 + sobel(lab[:, :, 1], axis=1) ** 2)
    grad_b = np.sqrt(sobel(lab[:, :, 2], axis=0) ** 2 + sobel(lab[:, :, 2], axis=1) ** 2)
    color_grad = grad_a + grad_b

    # Moderate gradient = blending (not edges, not flat)
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

    # Sample horizontal lines through blend regions, measure color path curvature
    h, w = rgb_f.shape[:2]
    curvatures = []
    sat_dips = []
    light_dips = []

    for row in range(0, h, 8):
        cols = np.where(blend_mask[row])[0]
        if len(cols) < 10:
            continue
        # Take the Lab values along this row at blend pixels
        path_lab = lab[row, cols]
        if len(path_lab) < 3:
            continue
        # Compute curvature: deviation from straight line in Lab space
        start = path_lab[0]
        end = path_lab[-1]
        n = len(path_lab)
        t = np.linspace(0, 1, n)
        straight = start[None, :] + t[:, None] * (end - start)[None, :]
        deviations = np.linalg.norm(path_lab - straight, axis=1)
        curvatures.append(float(deviations.mean()))

        # Saturation dip: min chroma along path vs endpoints
        chroma = np.sqrt(path_lab[:, 1] ** 2 + path_lab[:, 2] ** 2)
        endpoint_chroma = (chroma[0] + chroma[-1]) / 2
        if endpoint_chroma > 1:
            sat_dips.append(float(chroma.min() / endpoint_chroma))

        # Lightness dip
        endpoint_L = (path_lab[0, 0] + path_lab[-1, 0]) / 2
        if endpoint_L > 1:
            light_dips.append(float(path_lab[:, 0].min() / endpoint_L))

    return {
        "color_grad_curvature_mean": float(np.mean(curvatures)) if curvatures else 0.0,
        "color_grad_curvature_std": float(np.std(curvatures)) if curvatures else 0.0,
        "blend_saturation_dip": float(np.mean(sat_dips)) if sat_dips else 0.0,
        "blend_lightness_dip": float(np.mean(light_dips)) if light_dips else 0.0,
    }


def _patch_selfsimilarity_features(gray: NDArray) -> dict[str, float]:
    """Patch self-similarity statistics (4 features).

    AI generators sometimes produce suspiciously similar patches in textured
    regions due to attention mechanisms and tiling. Human art has more
    natural variation.
    """
    gray_f = gray if gray.max() <= 1 else gray / 255.0
    h, w = gray_f.shape
    patch_size = 16
    stride = 16

    # Extract non-overlapping patches
    patches = []
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = gray_f[y:y+patch_size, x:x+patch_size].ravel()
            patches.append(patch)

    if len(patches) < 10:
        return {
            "selfsim_min_dist": 0.0,
            "selfsim_mean_min_dist": 0.0,
            "selfsim_near_duplicate_ratio": 0.0,
            "selfsim_dist_std": 0.0,
        }

    patches = np.array(patches)
    n = len(patches)

    # Normalize patches
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches_norm = patches / (norms + 1e-10)

    # Compute cosine similarity matrix (sample if too many patches)
    if n > 200:
        idx = np.random.default_rng(42).choice(n, 200, replace=False)
        patches_norm = patches_norm[idx]
        n = 200

    sim_matrix = patches_norm @ patches_norm.T
    # Zero out diagonal
    np.fill_diagonal(sim_matrix, -1)

    # Best match for each patch (excluding self)
    max_sims = sim_matrix.max(axis=1)

    # Near-duplicate ratio: patches with similarity > 0.95
    near_dup_ratio = float((max_sims > 0.95).mean())

    return {
        "selfsim_min_dist": float(1 - max_sims.max()),  # smallest distance between any two patches
        "selfsim_mean_min_dist": float(1 - max_sims.mean()),
        "selfsim_near_duplicate_ratio": near_dup_ratio,
        "selfsim_dist_std": float(max_sims.std()),
    }


def _cross_subband_correlation_features(gray: NDArray) -> dict[str, float]:
    """Cross-subband wavelet correlation (4 features).

    Natural images have specific cross-band correlation structures.
    AI-generated images often have anomalous relationships between
    frequency subbands.
    """
    import pywt

    gray_f = gray if gray.max() <= 1 else gray / 255.0

    # 2-level wavelet decomposition
    coeffs = pywt.wavedec2(gray_f, "haar", level=2)

    # Level 1 details: (LH1, HL1, HH1)
    lh1, hl1, hh1 = coeffs[2]
    # Level 2 details: (LH2, HL2, HH2)
    lh2, hl2, hh2 = coeffs[1]

    # Resize level 2 to match level 1 size for correlation
    from skimage.transform import resize
    lh2_up = resize(lh2, lh1.shape, order=1, anti_aliasing=False)
    hl2_up = resize(hl2, hl1.shape, order=1, anti_aliasing=False)

    # Cross-band correlations
    def _safe_corr(a: NDArray, b: NDArray) -> float:
        a_flat, b_flat = a.ravel(), b.ravel()
        if a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    # Within-level: LH vs HL correlation (directional consistency)
    lh_hl_corr_l1 = _safe_corr(lh1, hl1)

    # Cross-level: LH1 vs LH2 (scale consistency)
    lh_cross_corr = _safe_corr(lh1, lh2_up)

    # Cross-level: HL1 vs HL2
    hl_cross_corr = _safe_corr(hl1, hl2_up)

    # HH ratio between levels (detail energy ratio)
    hh1_energy = float((hh1 ** 2).mean())
    hh2_energy = float((hh2 ** 2).mean())
    hh_energy_ratio = hh1_energy / (hh2_energy + 1e-10)

    return {
        "wavelet_lh_hl_corr_l1": lh_cross_corr,
        "wavelet_lh_cross_level_corr": lh_cross_corr,
        "wavelet_hl_cross_level_corr": hl_cross_corr,
        "wavelet_hh_energy_ratio": hh_energy_ratio,
    }


def _linework_features(gray: NDArray) -> dict[str, float]:
    """Anime/illustration line work analysis (8 features).

    AI generators struggle with consistent stroke thickness and medium
    coherence in line art. Per AnimeDL-2M (2025), anime images have
    distinctive sharp, well-defined lines that AI mimics imperfectly.
    """
    from skimage.feature import canny
    from scipy.ndimage import distance_transform_edt, label

    gray_f = gray if gray.max() <= 1 else gray / 255.0

    # Detect edges at two sensitivity levels
    edges_tight = canny(gray_f, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
    edges_loose = canny(gray_f, sigma=1.5, low_threshold=0.05, high_threshold=0.15)

    if edges_tight.sum() < 10:
        return {k: 0.0 for k in [
            "line_thickness_mean", "line_thickness_std", "line_thickness_cv",
            "line_density", "line_straightness",
            "edge_sharpness_mean", "edge_sharpness_std", "medium_consistency",
        ]}

    # Line thickness via distance transform
    # Invert edges to get distance to nearest edge, then sample at edge pixels
    dist_map = distance_transform_edt(~edges_tight)
    # Thickness = local width of strokes. Use loose edges as stroke regions.
    stroke_regions = edges_loose
    if stroke_regions.sum() > 0:
        thicknesses = dist_map[stroke_regions]
        thickness_mean = float(thicknesses.mean())
        thickness_std = float(thicknesses.std())
        thickness_cv = thickness_std / (thickness_mean + 1e-10)
    else:
        thickness_mean, thickness_std, thickness_cv = 0.0, 0.0, 0.0

    # Line density: fraction of image that is edges
    line_density = float(edges_tight.sum() / edges_tight.size)

    # Line straightness: ratio of connected component extent to perimeter
    labeled_edges, n_components = label(edges_tight)
    straightness_values = []
    for i in range(1, min(n_components + 1, 30)):
        component = (labeled_edges == i)
        n_pixels = component.sum()
        if n_pixels < 5:
            continue
        ys, xs = np.where(component)
        extent = max(ys.max() - ys.min(), xs.max() - xs.min(), 1)
        straightness_values.append(n_pixels / extent)
    line_straightness = float(np.mean(straightness_values)) if straightness_values else 0.0

    # Edge sharpness: gradient magnitude at edge pixels
    from scipy.ndimage import sobel as ndimage_sobel
    gx = ndimage_sobel(gray_f, axis=1)
    gy = ndimage_sobel(gray_f, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_gradients = grad_mag[edges_tight]
    edge_sharpness_mean = float(edge_gradients.mean())
    edge_sharpness_std = float(edge_gradients.std())

    # Medium consistency: how uniform is the texture in non-edge regions
    # Human artists use consistent medium; AI mixes characteristics
    non_edge = ~edges_loose
    if non_edge.sum() > 100:
        # Variance of local texture in non-edge regions (patch-based)
        h, w = gray_f.shape
        patch_vars = []
        for y in range(0, h - 16, 16):
            for x in range(0, w - 16, 16):
                patch = gray_f[y:y + 16, x:x + 16]
                patch_edge = edges_tight[y:y + 16, x:x + 16]
                if patch_edge.mean() < 0.1:  # non-edge patch
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


class ArtworkExtract:
    """Extract artwork features for AI detection.

    Combines features from multiple sources:
        - 39 features from Li & Stamp (2025)
        - 10 FFT/DCT spectral features
        - 14 enhanced texture features (Nirob et al. 2026)
        - 4 mid-band frequency features (FIRE, CVPR 2025)
        - 6 patch consistency features (CINEMAE 2025)
        - 8 multi-scale LBP features
        - 18 Gabor filter bank features
        - 12 wavelet packet statistics
        - 6 color coherence vector features
        - 8 edge co-occurrence features
        - 2 fractal dimension features
        - 6 extended HOG features
        - 4 JPEG ghost detection features
        - 5 noise residual autocorrelation features
        - 4 stroke edge roughness features
        - 4 color gradient curvature features
        - 4 patch self-similarity features
        - 4 cross-subband wavelet correlation features
    Total: 158 features, all CPU-only.

    Usage:
        >>> extractor = ArtworkExtract()
        >>> features = extractor(pil_image)
        >>> len(features)  # 158
    """

    def __call__(self, image: Image.Image) -> dict[str, float]:
        """Extract all features from a single PIL image.

        :param image: PIL Image in any mode (will be converted to RGB).
        :returns: Dictionary of scalar features.
        """
        rgb = _to_array(image)
        gray = rgb2gray(rgb / 255.0 if rgb.max() > 1 else rgb)

        features: dict[str, float] = {}
        features |= _brightness_features(gray)
        features |= _color_features(rgb)
        features |= _texture_features(gray)
        features |= _shape_features(gray)
        features |= _noise_features(gray)
        features |= _frequency_features(gray)
        features |= _enhanced_texture_features(gray)
        features |= _midband_frequency_features(gray)
        features |= _patch_consistency_features(gray)
        features |= _multiscale_lbp_features(gray)
        features |= _gabor_features(gray)
        features |= _wavelet_packet_features(gray)
        # color_coherence and cross_subband removed — ablation showed they hurt accuracy
        features |= _edge_cooccurrence_features(gray)
        features |= _fractal_dimension_features(gray)
        features |= _noise_residual_autocorr_features(gray)
        features |= _stroke_edge_roughness_features(gray)
        features |= _color_gradient_curvature_features(rgb)
        features |= _patch_selfsimilarity_features(gray)
        features |= _extended_hog_features(gray)
        features |= _jpeg_ghost_features(rgb)
        features |= _linework_features(gray)

        return features

    def feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        # Generate from a dummy image to get exact keys
        dummy = Image.new("RGB", (255, 255), color="gray")
        return list(self(dummy).keys())
