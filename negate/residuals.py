# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import laplace, sobel_h, sobel_v
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from torch import Tensor

from negate.config import Spec
from negate.scaling import split_array


class Residual:
    def __init__(self, spec: Spec) -> None:
        """Initialize residual class for residual image processing.\n
        :param dtype: dtype for internal numpy conversion.
        return: None."""

        self.np_dtype = spec.np_dtype
        self.top_k = spec.hyper_param.top_k

    def __call__(self, image: Image.Image, radius: int = 3) -> dict[str, list[np.ndarray] | np.ndarray]:
        greyscale = image.convert("L")
        numeric_image = np.array(greyscale, dtype=self.np_dtype)

        if numeric_image.ndim == 4:
            numeric_image = numeric_image[0]  # Remove batch dim
        if numeric_image.ndim == 3:
            numeric_image = numeric_image[0] if numeric_image.shape[0] in (1, 3) else numeric_image.mean(axis=0)

        point = 8 * radius
        lapl_tc = local_binary_pattern(self.laplace_residual(numeric_image), P=point, R=radius)
        sobl_tc = local_binary_pattern(self.sobel_residual(numeric_image), P=point, R=radius)
        spcl_tc = local_binary_pattern(np.array(self.spectral_residual(numeric_image)), P=point, R=radius)

        lapl_chunks = split_array(lapl_tc.flatten())
        sobl_chunks = split_array(sobl_tc.flatten())
        spcl_chunks = split_array(spcl_tc.flatten())

        return {
            "lapl_tc": lapl_chunks,
            "sobl_tc": sobl_chunks,
            "spcl_tc": spcl_chunks,
        }

    def laplace_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Create a 3-channel residual from a grayscale image.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale mode."""

        residual = laplace(numeric_image, ksize=3)
        return np.asarray(residual).astype(np.int8)

    def sobel_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Create a 3-channel residual using Sobel edge detection.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale mode."""

        grad_x = sobel_h(numeric_image)
        grad_y = sobel_v(numeric_image)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        return np.asarray(gradient).astype(np.int8)

    def spectral_residual(self, numeric_image: np.ndarray) -> np.ndarray:
        """Create a 3-channel residual using FFT magnitude spectrum of the frequency domain.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale mode."""

        fourier_transform = np.fft.fft2(numeric_image)
        fourier_2d_shift = np.fft.fftshift(fourier_transform)

        magnitude_spectra = 20 * np.log(np.abs(fourier_2d_shift) + 1)
        return np.asarray(magnitude_spectra).astype(np.int8)

    def texture_complexity(self, residual: np.ndarray, patch_size: int = 16, d: float = 255.0) -> np.ndarray:
        """Compute texture complexity for each NxN patch in the residual.

        Args:
            residual: Laplacian residual noise array.
            patch_size: Size of square patches (default 8).
            d: Normalization denominator (default 255.0 for 8-bit images).

        Returns:
            Array of TC values, one per patch.
        """
        rows, cols = residual.shape
        tc_map = np.zeros((rows // patch_size, cols // patch_size), dtype=self.np_dtype)

        for i in range(0, rows - patch_size + 1, patch_size):
            for j in range(0, cols - patch_size + 1, patch_size):
                patch = residual[i : i + patch_size, j : j + patch_size]
                r_sq_mean = np.mean(patch**2)
                t_val = 1.0 - (r_sq_mean / d**2)

                if t_val <= 0 or t_val >= 1:
                    tc_map[i // patch_size, j // patch_size] = 0.0
                else:
                    tc_map[i // patch_size, j // patch_size] = np.log(t_val / (1 - t_val)) + 4

        return tc_map
