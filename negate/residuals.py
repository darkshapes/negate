# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image


class Residual:
    def __init__(self, dtype: np.typing.DTypeLike = np.float32):
        """Initialize residual class for residual image processing.\n
        :param dtype: dtype for internal numpy conversion.
        return: None."""

        self.dtype = dtype

    def __call__(self, image: Image) -> Image:
        """Create a 3-channel residual from a grayscale image.\n
        :param image: PIL image to process.
        :return: Residual image in RGB mode."""

        from PIL.Image import fromarray
        from skimage.filters import laplace

        greyscale = image.convert("L")
        numeric_image = np.array(greyscale, dtype=self.dtype)
        residual = laplace(numeric_image, ksize=3).astype(self.dtype)
        residual_image: Image = fromarray(np.uint8(residual), mode="L").convert("RGB")
        return residual_image

    def sobel(self, image: Image) -> Image:
        """Create a 3-channel residual using Sobel edge detection.\n
        :param image: PIL image to process.
        :return: Residual image in RGB mode."""

        import cv2
        from PIL.Image import fromarray

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.dtype)
        grad_x = cv2.Sobel(numeric_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(numeric_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        residual_image: Image = fromarray(np.uint8(gradient), mode="L").convert("RGB")
        return residual_image

    def frequency(self, image: Image) -> Image:
        """Create a 3-channel high-frequency residual using FFT magnitude spectrum.\n
        :param image: PIL image to process.
        :return: Residual image in RGB mode."""

        from PIL.Image import fromarray

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.dtype)

        fourier_transform = np.fft.fft2(numeric_image)
        fourier_2d_shift = np.fft.fftshift(fourier_transform)

        magnitude_spectrum = 20 * np.log(np.abs(fourier_2d_shift) + 1)

        residual_image: Image = fromarray(np.uint8(magnitude_spectrum), mode="L").convert("RGB")
        return residual_image

    def spectral(self, image: Image) -> Image:
        """Create a 3-channel spectral residual using magnitude of frequency domain.\n
        :param image: PIL image to process.
        :return: Residual image in RGB mode."""

        from PIL.Image import fromarray

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.dtype)

        f_transform = np.fft.fft2(numeric_image)
        f_shift = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        residual_image: Image = fromarray(np.uint8(magnitude_spectrum), mode="L").convert("RGB")
        return residual_image

    def masked_spectral(self, numeric_image: NDArray, mask_radius: int = 50) -> tuple[NDArray, NDArray]:
        """Apply Masked Spectral Learning logic to an image.\n
        :param image: PIL image to process.
        :param mask_radius: Radius r for the circular mask.
        :param mask_type: 'high' to zero out center (low-freq), 'low' to zero out edges (high-freq).
        :return: Masked spectral image in RGB mode."""

        fourier_transform = np.fft.fft2(numeric_image)  # Compute Discrete Fourier Transform: chi = F(x)
        fourier_shift = np.fft.fftshift(fourier_transform)

        rows, cols = fourier_shift.shape
        center = (rows // 2, cols // 2)

        y, x = np.ogrid[:rows, :cols]
        euclid_dist_from_center: NDArray = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        mask: NDArray = euclid_dist_from_center < mask_radius

        return mask, fourier_shift  # type: ignore

    def image_from_fourier(self, masked_spectrum: NDArray, fourier_shift: NDArray, mask_type: Literal["high", "low"] = "high") -> Image:
        """Return image from Fourier domain.\n
        :return: PIL Image.
        :param masked_spectrum: masked spectrum array.
        :param fourier_shf: original shift array.
        :param mask_type: "high" or "low" band fouriers.
        :raises ValueError: mask_type must be 'high' or 'low'."""

        match mask_type:
            case "high":
                masked_spectrum = fourier_shift * masked_spectrum
            case "low":
                masked_spectrum = fourier_shift * (1 - masked_spectrum)
            case _:
                raise ValueError("mask_type must be 'high' or 'low'")

        from PIL.Image import fromarray

        fourier_inverse_shift = np.fft.ifftshift(masked_spectrum)
        reconstructed = np.fft.ifft2(fourier_inverse_shift)

        reconstructed_real = np.real(reconstructed)
        reconstructed_uint8 = ((reconstructed_real - reconstructed_real.min()) / (reconstructed_real.max() - reconstructed_real.min()) * 255).astype(np.uint8)

        return fromarray(reconstructed_uint8, mode="L").convert("RGB")

    def mask_patches(self, numeric_image: NDArray, size: int):
        """Crop patches and compute freq divergence.\n
        :return: List of (divergence, patch Image).
        :param numeric_image: Image converted into an array.
        :param size: Patch dimensions in pixels."""

        from PIL.Image import fromarray

        metrics: list[tuple[float, Image]] = []

        h, w = numeric_image.shape
        nx = (w + size - 1) // size
        ny = (h + size - 1) // size
        for iy in range(ny):
            for ix in range(nx):
                x0 = ix * size
                y0 = iy * size
                patch_arr = numeric_image[y0 : y0 + size, x0 : x0 + size]
                if patch_arr.shape != (size, size):
                    pad = np.zeros((size, size), dtype=self.dtype)
                    pad[: patch_arr.shape[0], : patch_arr.shape[1]] = patch_arr
                    patch_arr = pad

                high_mask, fourier_shift = self.masked_spectral(patch_arr)
                low_mask = ~high_mask

                low_magnitude = np.abs(fourier_shift[low_mask])
                high_magnitude = np.abs(fourier_shift[high_mask])

                div = float(abs(np.mean(high_magnitude) - np.mean(low_magnitude)))

                patch_img = fromarray(np.uint8(patch_arr), mode="L").convert("RGB")
                metrics.append((div, patch_img))
        return metrics

    def crop_select(
        self,
        image: Image,
        size: int,
        top_k: int = 5,
    ) -> list[Image]:
        """Crop image into patches, compute freq-divergence, return most extreme patches.\n
        :param image: PIL image to process.
        :param size: Patch dimension.
        :param top_k: Number of extreme patches to return.
        :param mask_radius: Radius used in masked_spectral logic.
        :return: List of selected patch images."""

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.dtype)

        metrics: list[tuple[float, Image]] = self.mask_patches(numeric_image, size=size)

        metrics.sort(key=lambda x: x[0], reverse=True)

        chosen: list[Image] = []
        chosen.extend([p for _, p in metrics[:top_k]])  # high diverges
        chosen.extend([p for _, p in metrics[-top_k:]])  # low diverges

        return chosen
