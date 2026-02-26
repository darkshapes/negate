# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Residual image processing with GPU acceleration"""

import numpy as np
from numpy.fft import fftfreq
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import difference_of_gaussians, laplace, sobel_h, sobel_v
from torch import Tensor

from negate.io.config import Spec


class Residual:
    """Image residual feature computations using various processing and filtering techniques."""

    def __init__(self, spec: Spec) -> None:
        """Initialize residual class for residual image processing.\n
        :param spec: Configuration specification object.
        return: None."""

        self.residual_dtype = getattr(np, spec.opt.residual_dtype, np.float64)
        self.top_k = spec.opt.top_k
        self.device = spec.device
        self.dim_patch = spec.opt.dim_patch
        self.patch_resolution = self.dim_patch * self.dim_patch

    def __call__(self, image: Image.Image | Tensor) -> dict[str, float | tuple[int, int]]:
        """Compute residual features from a single image.\n
        :param image: Input PIL image.
        :returns: Dictionary with flattened residual metrics.
        """
        numeric_image = self.make_numeric(image)

        diff_residual = difference_of_gaussians(numeric_image, low_sigma=1.5)
        laplace_residual = laplace(numeric_image, ksize=3)
        sobel_residual = self.sobel_hv_residual(numeric_image)

        res_map = {
            "image": numeric_image,
            "diff": diff_residual,
            "laplace": laplace_residual,
            "sobel": sobel_residual,
        }

        lbp_mean, tc_mean = (
            self.find_local_pattern(res_map),
            self.texture_complexity(
                {f"{label}": num for label, num in {**res_map, "spectral": numeric_image}.items()},
            ),
        )

        return {
            "image_mean_ff": float(numeric_image.mean()),
            "image_std": float(numeric_image.std()),
            **{f"{label}_mean": (int(np.mean(num)), int(np.sum(num))) for label, num in lbp_mean.items()},
            **{f"{label}_tc": (int(np.mean(num)), int(np.sum(num))) for label, num in tc_mean.items()},
        }

    def forward(self, image: Tensor) -> dict[str, float | tuple[int, ...]]:
        numeric_image = self.make_numeric(image)
        forward_result: dict[str, float | tuple[int, ...]] = {"image_mean_ff": float(numeric_image.mean())}
        sobel_residual = self.sobel_hv_residual(numeric_image)
        local_pattern = self.find_local_pattern({"sobel": sobel_residual})
        for key, value in local_pattern.items():
            forward_result.setdefault(key, (int(np.mean(value)), int(np.sum(value))))
        return forward_result

    def fourier_discrepancy(self, image: np.ndarray | Tensor) -> dict[str, float]:
        """Compute Fourier-based discrepancy metrics for discriminating image differences.\n
        :param image: Input numpy array.
        :returns: Dictionary with magnitude-based discrimination metrics."""

        numeric_image = self.make_numeric(image)

        spec_residual = self.spectral_residual(numeric_image)  # Spectral residual preserves spatial frequency information
        normalized_spec = (spec_residual - spec_residual.min()) / (spec_residual.max() - spec_residual.min() + 1e-10)
        fft_2d = np.fft.fftn(numeric_image.astype(self.residual_dtype))
        magnitude_spectrum: np.ndarray = np.abs(fft_2d)
        log_mag: np.ndarray = np.log(magnitude_spectrum + 1e-10)

        h, w = numeric_image.shape
        center_h, center_w = h // 2, w // 2
        spectral_centroid = float(np.sum(log_mag * fftfreq(h)[:, None] + log_mag.T * fftfreq(w)[None, :]) / (log_mag.sum() * 2 + 1e-10))
        return {
            "spectral_centroid": float(spectral_centroid),
            "high_freq_ratio": float((magnitude_spectrum[center_h:, center_w:] ** 2).sum() / (magnitude_spectrum**2).sum()),
            "low_freq_energy": float((magnitude_spectrum[:center_h, :center_w] ** 2).sum()),
            "spectral_entropy": -(normalized_spec * np.log(normalized_spec + 1e-10)).sum(),
            "max_magnitude": float(magnitude_spectrum.max()),
            "mean_log_magnitude": float(log_mag.mean()),
        }

    def make_numeric(self, image: Image.Image | Tensor | np.ndarray) -> np.ndarray:
        """Convert a PIL Image or tensor to a 2-D grayscale numpy array.\n
        :param image: Input image (PIL or torch.Tensor).
        :return: Grayscale numeric representation (HxW).
        """
        if isinstance(image, Tensor):
            numeric_image = image.cpu().numpy()
        elif isinstance(image, Image.Image):
            numeric_image = np.asarray(image)
        else:
            numeric_image = np.array(image)

        while numeric_image.ndim > 3:
            numeric_image = numeric_image.squeeze(0)  # remove leading batch dim

        if numeric_image.ndim == 3 and numeric_image.shape[0] <= 4:  # bring channel axis to last position
            numeric_image = np.moveaxis(numeric_image, 0, -1)

        gray = rgb2gray(numeric_image)
        return gray.astype(self.residual_dtype)

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

    def batch_fourier_discrepancy(self, images: np.ndarray) -> dict[str, np.ndarray]:
        """Compute discrepancy metrics across image batch.\n
        :param images: Batch of images (batch x H x W).
        :returns: Dictionary with per-image metric arrays."""

        results = {label: [] for label in ["spectral_centroid", "high_freq_ratio", "max_magnitude"]}

        for img in images:
            disc = self.fourier_discrepancy(img)
            for label in results:
                results[label].append(disc[label])  # type: ignore

        return {label: np.array(result) for label, result in results.items()}

    def find_local_pattern(self, images: dict[str, np.ndarray], radius: int = 3) -> dict[str, tuple[float, float]]:
        """Compute (mean, sum) for each LBP result.\n
        :param images: Dictionary of image arrays.
        :param radius: Radius parameter for Local Binary Pattern.
        return: Dictionary with mean and sum results."""

        point = 8 * radius
        results = {}
        for label, img in images.items():
            lbp = local_binary_pattern(self._normalize_to_uint8(img), P=point, R=radius)
            results[label] = (np.mean(lbp), np.sum(lbp))
        return results

    def sobel_hv_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Sobel edge detection residual from grayscale image.\n
        :param numeric_image: Input numpy array.
        return: Sobel residual."""

        grad_x = sobel_h(numeric_image)
        grad_y = sobel_v(numeric_image)
        return np.sqrt(grad_x**2 + grad_y**2).astype(self.residual_dtype)

    def spectral_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Spectral residual using FFT magnitude spectrum.\n
        :param numeric_image: Input numpy array.
        return: Spectral residual image."""

        if self.patch_resolution >= 65536:
            import torch

            img_gpu = torch.from_numpy(numeric_image).to(self.device, dtype=torch.float32)  # unsupported half precision
            fft_gpu = torch.fft.fftn(img_gpu)
            result = (20 * torch.log(fft_gpu.abs() + 1)).cpu().numpy()
            return np.fft.fftshift(result)

        fourier_transform = np.fft.fftn(numeric_image)
        fourier_2d_shift = np.fft.fftshift(fourier_transform)

        return (20 * np.log(np.abs(fourier_2d_shift) + 1)).astype(self.residual_dtype)

    def texture_complexity(self, residuals: dict[str, np.ndarray], patch_size: int = 16) -> dict[str, float]:
        """Texture complexity via nested loop over patches.\n
        :param residual: Input numpy array for analysis.
        :param patch_size: Size of the analysis patch.
        return: Calculated texture complexity."""

        results = {}
        for label, residual in residuals.items():
            rows, cols = residual.shape
            tc_values = []

            for num in range(0, rows - patch_size + 1, patch_size):
                for j in range(0, cols - patch_size + 1, patch_size):
                    patch = residual[num : num + patch_size, j : j + patch_size]
                    r_sq_mean = np.mean(patch**2)
                    t_val = 1.0 - (r_sq_mean / 255.0**2)

                    if 0 < t_val < 1:
                        tc_values.append(np.log(t_val / (1 - t_val)) + 4)

            results[label] = float(np.mean(tc_values)) if tc_values else 0.0
        return results
