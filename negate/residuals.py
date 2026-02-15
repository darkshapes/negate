# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Residual image processing with GPU acceleration"""

import numpy as np
from numpy.fft import fftfreq
import torch
from torch import Tensor
from PIL import Image
from scipy.fft import fftn, fftshift
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.feature import local_binary_pattern
from skimage.filters import difference_of_gaussians, laplace, sobel_h, sobel_v, window

from negate.config import Spec


class Residual:
    """Image residual feature computations using various processing and filtering techniques."""

    def __init__(self, spec: Spec) -> None:
        """Initialize residual class for residual image processing.\n
        :param spec: Configuration specification object.
        return: None."""

        self.dtype = spec.dtype
        self.np_dtype = spec.np_dtype
        self.top_k = spec.hyper_param.top_k
        self.device = spec.device
        self.dim_patch = spec.opt.dim_patch
        self.patch_resolution = self.dim_patch * self.dim_patch

    @adapt_rgb(each_channel)
    def __call__(self, image: Image.Image | Tensor) -> dict[str, float]:
        """Compute residual features from a single image.\n
        :param image: Input PIL image.
        :returns: Dictionary with flattened residual metrics.
        """

        assert isinstance(image, (Tensor, Image.Image)), TypeError(f"Unable to detect type : {type(image)} to convert to array")

        if isinstance(image, Tensor):
            img_array = image.cpu().numpy()
        else:
            img_array = np.array(image)

        float_image = img_array.astype(self.np_dtype)

        diff_residual = difference_of_gaussians(img_array, low_sigma=1.5)
        lap_residual = laplace(img_array, ksize=3)
        sob_residual = sobel_h(img_array)
        spec_residual = np.abs(np.fft.fftn(img_array))
        fft_magnitude = np.log(np.abs(spec_residual) + 1)

        img_mean = float(float_image.mean())
        img_std = float(float_image.std())

        diff_mean = float(diff_residual.mean())
        lap_mean = float(lap_residual.mean())
        sob_mean = float(sob_residual.mean())
        fft_mean = float(fft_magnitude.mean())

        return {
            "img_mean": img_mean,
            "img_std": img_std,
            "diff_mean": diff_mean,
            "lap_mean": lap_mean,
            "sob_mean": sob_mean,
            "fft_mean": fft_mean,
        }

    def batch_fourier_discrepancy(self, images: np.ndarray) -> dict[str, np.ndarray]:
        """Compute discrepancy metrics across image batch.\n
        :param images: Batch of images (batch x H x W).
        :returns: Dictionary with per-image metric arrays."""

        results = {key: [] for key in ["spectral_centroid", "high_freq_ratio", "max_magnitude"]}

        for img in images:
            disc = self.fourier_discrepancy(img)
            for k in results:
                results[k].append(disc[k])

        return {k: np.array(v) for k, v in results.items()}

    def fourier_discrepancy(self, image: np.ndarray | Tensor) -> dict[str, float | list[float]]:
        """Compute Fourier-based discrepancy metrics for discriminating image differences.\n
        :param image: Input numpy array.
        :returns: Dictionary with magnitude-based discrimination metrics."""

        if isinstance(image, Tensor):
            numeric_image = image.cpu().numpy()
        else:
            numeric_image = np.array(image)

        # Handle multi-dimensional inputs by averaging non-spatial dimensions
        while numeric_image.ndim > 2:
            numeric_image = numeric_image.mean(axis=tuple(range(numeric_image.ndim - 2)))

        spec_residual = self.spectral_residual(numeric_image)  # Spectral residual preserves spatial frequency information

        normalized_spec = (spec_residual - spec_residual.min()) / (spec_residual.max() - spec_residual.min() + 1e-10)

        fft_2d = np.fft.fftn(numeric_image.astype(np.float16))
        magnitude_spectrum = np.abs(fft_2d)
        log_mag = np.log(magnitude_spectrum + 1e-10)

        h, w = numeric_image.shape
        center_h, center_w = h // 2, w // 2
        spectral_centroid = float(np.sum(log_mag * fftfreq(h)[:, None] + log_mag.T * fftfreq(w)[None, :]) / (log_mag.sum() * 2 + 1e-10))
        return {
            "spectral_centroid": spectral_centroid,
            "high_freq_ratio": float((magnitude_spectrum[center_h:, center_w:] ** 2).sum() / (magnitude_spectrum**2).sum()),
            "low_freq_energy": float((magnitude_spectrum[:center_h, :center_w] ** 2).sum()),
            "spectral_entropy": -(normalized_spec * np.log(normalized_spec + 1e-10)).sum(),
            "max_magnitude": float(magnitude_spectrum.max()),
            "mean_log_magnitude": float(log_mag.mean()),
        }

    def _find_local_pattern(self, images: dict[str, np.ndarray], radius: int = 3) -> dict[str, tuple[float, float]]:
        """Compute (mean, sum) for each LBP result.\n
        :param images: Dictionary of image arrays.
        :param radius: Radius parameter for Local Binary Pattern.
        return: Dictionary with mean and sum results."""

        point = 8 * radius
        results = {}
        for k, img in images.items():
            lbp = local_binary_pattern(self._normalize_to_uint8(img), P=point, R=radius)
            results[k] = (np.mean(lbp, dtype=np.float32), np.sum(lbp, dtype=np.float32))
        return results

    def _normalize_to_uint8(self, numeric_image: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 255] and convert to uint8.\n
        :param numeric_image: Input numpy array.
        return: Normalized uint8 array."""

        mn, mx = numeric_image.min(), numeric_image.max()
        if mx - mn > 0:
            normalized = (numeric_image - mn) / (mx - mn)
        else:
            normalized = np.zeros_like(numeric_image)
        return (normalized * 255).astype(np.uint8)

    def make_numeric(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to a numeric numpy array.\n
        :param image: Input PIL Image.
        return: Numeric representation of the image."""

        original_grey = image.convert("L")
        numeric_image = np.array(original_grey, dtype=self.np_dtype)

        match numeric_image.ndim:
            case 4:
                return numeric_image[0]
            case 3 if numeric_image.shape[0] in (1, 3):
                return numeric_image[0]
            case 3:
                return numeric_image.mean(axis=0)
            case _:
                return numeric_image

    @adapt_rgb(each_channel)
    def _sobel_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Sobel edge detection residual from grayscale image.\n
        :param numeric_image: Input numpy array.
        return: Sobel residual."""

        grad_x = sobel_h(numeric_image)
        grad_y = sobel_v(numeric_image)
        return np.sqrt(grad_x**2 + grad_y**2).astype(self.np_dtype)

    def spectral_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Spectral residual using FFT magnitude spectrum.\n
        :param numeric_image: Input numpy array.
        return: Spectral residual image."""

        if self.patch_resolution >= 65536:
            import torch

            img_gpu = torch.from_numpy(numeric_image).to(self.device, dtype=torch.float32)
            fft_gpu = torch.fft.fftn(img_gpu)
            result = (20 * torch.log(fft_gpu.abs() + 1)).cpu().numpy()
            return np.fft.fftshift(result)

        fourier_transform = np.fft.fftn(numeric_image)
        fourier_2d_shift = np.fft.fftshift(fourier_transform)

        return (20 * np.log(np.abs(fourier_2d_shift) + 1)).astype(self.np_dtype)

    def texture_complexity(self, residual: np.ndarray, patch_size: int = 16) -> float:
        """Texture complexity via nested loop over patches.\n
        :param residual: Input numpy array for analysis.
        :param patch_size: Size of the analysis patch.
        return: Calculated texture complexity."""

        if self.patch_resolution >= 65536:
            return self.texture_complexity_gpu(residual)

        rows, cols = residual.shape
        tc_values = []

        for i in range(0, rows - patch_size + 1, patch_size):
            for j in range(0, cols - patch_size + 1, patch_size):
                patch = residual[i : i + patch_size, j : j + patch_size]
                r_sq_mean = np.mean(patch**2)
                t_val = 1.0 - (r_sq_mean / 255.0**2)

                if 0 < t_val < 1:
                    tc_values.append(np.log(t_val / (1 - t_val)) + 4)

        return float(np.mean(tc_values)) if tc_values else 0.0

    def texture_complexity_gpu(self, residual: np.ndarray, patch_size: int = 16) -> float:
        """Vectorized texture complexity using PyTorch GPU if available.\n
        :param residual: Input numpy array for analysis.
        :param patch_size: Size of the analysis patch.
        return: Calculated texture complexity."""

        res_gpu = torch.from_numpy(residual).to(self.device, dtype=self.dtype)

        h, w = res_gpu.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            res_np = res_gpu.cpu().numpy()  # PyTorch doesn't support reflect padding for 2D tensors with size=4
            padded_np = np.pad(res_np, [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)], mode="reflect")
            res_gpu = torch.from_numpy(padded_np).to(self.device)
            h_new, w_new = padded_np.shape
        else:
            h_new, w_new = h, w

        n_patches_h = h_new // patch_size
        n_patches_w = w_new // patch_size

        patches = res_gpu.reshape(n_patches_h, patch_size, n_patches_w, patch_size).permute(2, 0, 3, 1)
        patches = patches.reshape(-1, patch_size, patch_size)

        r_sq_mean = (patches**2).mean(dim=(1, 2))
        t_val = 1.0 - (r_sq_mean / 255.0**2)

        valid = (t_val > 0) & (t_val < 1)
        if not valid.any():
            return 0.0

        tc_values = torch.log(t_val[valid] / (1 - t_val[valid])) + 4
        return float(tc_values.mean().item())
