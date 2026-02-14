# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import numpy as np
import torch
from PIL import Image
from scipy.fft import fftn, fftshift
from skimage.feature import local_binary_pattern
from skimage.filters import difference_of_gaussians, laplace, sobel_h, sobel_v, window

from negate.config import Spec


class Residual:
    """Residual image processing with GPU acceleration"""

    def __init__(self, spec: Spec) -> None:
        """Initialize residual class for residual image processing.

        :param spec: Configuration specification object.
        return: None."""

        self.dtype = spec.dtype
        self.np_dtype = spec.np_dtype
        self.top_k = spec.hyper_param.top_k
        self.device = spec.device

    def __call__(self, image: Image.Image) -> dict[str, str | dict[str, float | int]]:
        """Process image through all residual methods and save visualizations.\n
        :param name: Output filename for saved figure.
        :param image: Input PIL Image to process.
        :param radius: Parameter for LBP (unused).
        return: Dictionary containing processing results."""

        numeric_image = self.make_numeric(image)

        diff_res = difference_of_gaussians(numeric_image, 1, 12)
        laplace_res = laplace(numeric_image, ksize=3).astype(self.np_dtype)
        sobel_res = np.array(self._sobel_residual(numeric_image))
        spectral_res = np.array(self.spectral_residual(numeric_image))

        img_mag, img_mean = self.window_shift(numeric_image)  # baseline: raw image spectrum
        ff_dg_mag, tc_dg_mean = self.window_shift(diff_res)  # DoG filter effect in freq
        ff_lapl_mag, tc_lapl_mean = self.window_shift(laplace_res)
        ff_sobl_mag, tc_sobl_mean = self.window_shift(sobel_res)
        ff_spc_mag, tc_spc_mean = self.window_shift(sobel_res)

        texture_mean = {
            "original": img_mean,
            "diff": tc_dg_mean,
            "laplace": tc_lapl_mean,
            "sobel": tc_sobl_mean,
            "spectral": tc_spc_mean,
        }

        lbp_num = {}
        for k, v in self._find_local_pattern(
            {
                "original": numeric_image,
                "diff": diff_res,
                "laplace": laplace_res,
                "sobel": sobel_res,
                "spectral": spectral_res,
            }
        ).items():
            lbp_num[k] = (int(np.mean(v, dtype=np.float32)), int(np.sum(v, dtype=np.float32)))

        magnitudes = {
            "original": img_mag,
            "diff": ff_dg_mag,
            "laplace": ff_lapl_mag,
            "sobel": ff_sobl_mag,
            "spectral": ff_spc_mag,
        }

        return {
            "texture_complexity_mean": {k: float(v) for k, v in texture_mean.items()},
            "local binary pattern": lbp_num,
            "magnitudes": {k: int(v) for k, v in magnitudes.items()},
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

    def window_shift(self, numeric_image: np.ndarray) -> tuple[np.float32, float]:
        """Apply FFT and compute magnitude sum + texture complexity.\n
        :param numeric_image: Input numpy array.
        return: Tuple of magnitude sum and texture complexity."""

        if numeric_image.size == 0:
            return np.float32(0), 0.0

        if numeric_image.size >= 256 * 256:
            import torch

            img_gpu = torch.from_numpy(numeric_image).to(device=self.device, dtype=self.dtype)
            window_hann = torch.from_numpy(window("hann", numeric_image.shape)).to(self.device, dtype=self.dtype)

            fft_gpu = torch.fft.fftn(img_gpu * window_hann)
            mag = fft_gpu.abs().cpu().numpy().astype(np.float32)

            return np.sum(mag), self.texture_complexity_gpu(mag, 16)

        if np.iscomplexobj(numeric_image) or numeric_image.ndim < 2:
            mag = np.abs(numeric_image).astype(np.float32)
        else:
            windowed = fftn(numeric_image * window("hann", numeric_image.shape))
            magnitude = fftshift(np.abs(windowed))  # type: ignore
            mag = magnitude.astype(np.float32)

        return np.sum(mag), self.texture_complexity_gpu(mag, 16)

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
        """Convert a PIL Image to a numeric numpy array.

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

    def _sobel_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Sobel edge detection residual from grayscale image.

        :param numeric_image: Input numpy array.
        return: Sobel residual."""

        grad_x = sobel_h(numeric_image)
        grad_y = sobel_v(numeric_image)
        return np.sqrt(grad_x**2 + grad_y**2).astype(self.np_dtype)

    def spectral_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Spectral residual using FFT magnitude spectrum.

        :param numeric_image: Input numpy array.
        return: Spectral residual image."""

        if numeric_image.size >= 256 * 256:
            import torch

            img_gpu = torch.from_numpy(numeric_image).to(self.device, dtype=torch.float32)
            fft_gpu = torch.fft.fftn(img_gpu)
            result = (20 * torch.log(fft_gpu.abs() + 1)).cpu().numpy()
            return np.fft.fftshift(result)

        fourier_transform = np.fft.fftn(numeric_image)
        fourier_2d_shift = np.fft.fftshift(fourier_transform)
        return (20 * np.log(np.abs(fourier_2d_shift) + 1)).astype(self.np_dtype)

    def texture_complexity(self, residual: np.ndarray, patch_size: int = 16) -> float:
        """Texture complexity via nested loop over patches.

        :param residual: Input numpy array for analysis.
        :param patch_size: Size of the analysis patch.
        return: Calculated texture complexity."""

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

        if residual.size < 256 * 256:
            return self.texture_complexity(residual, patch_size)

        res_gpu = torch.from_numpy(residual).to(self.device, dtype=self.dtype)

        h, w = res_gpu.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            # Use numpy's reflect padding since PyTorch doesn't support it for 2D tensors with size=4
            res_np = res_gpu.cpu().numpy()
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
