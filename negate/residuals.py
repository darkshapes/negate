# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from PIL.Image import Image, fromarray
from skimage.filters import laplace
import numpy as np


class Residual:
    def __init__(self, dtype: np.typing.DTypeLike = np.float32):
        """Initialize Residual.\n
        :param dtype: dtype for internal numpy conversion.
        return: None."""
        self.dtype = dtype

    def __call__(self, image: Image) -> Image:
        """Create a 3-channel residual from a grayscale image.\n
        :param image: PIL image to process.
        :return: Residual image in RGB mode."""

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
        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.dtype)

        f_transform = np.fft.fft2(numeric_image)
        f_shift = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        residual_image: Image = fromarray(np.uint8(magnitude_spectrum), mode="L").convert("RGB")
        return residual_image

    @staticmethod
    def masked_spectral(image: Image, mask_radius: int = 50, mask_type: str = "high") -> Image:
        """Apply Masked Spectral Learning logic to an image.\n
        :param image: PIL image to process.
        :param mask_radius: Radius r for the circular mask.
        :param mask_type: 'high' to zero out center (low-freq), 'low' to zero out edges (high-freq).
        :return: Masked spectral image in RGB mode."""

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=np.float32)

        # 1. Compute Discrete Fourier Transform: chi = F(x)
        fourier_transform = np.fft.fft2(numeric_image)
        fourier_shift = np.fft.fftshift(fourier_transform)

        rows, cols = fourier_shift.shape
        center = (rows // 2, cols // 2)

        y, x = np.ogrid[:rows, :cols]
        euclid_dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        mask = euclid_dist_from_center < mask_radius

        match mask_type:
            case "high":
                masked_spectrum = fourier_shift * mask
            case "low":
                masked_spectrum = fourier_shift * (1 - mask)
            case _:
                raise ValueError("mask_type must be 'high' or 'low'")

        fourier_inverse_shift = np.fft.ifftshift(masked_spectrum)
        reconstructed = np.fft.ifft2(fourier_inverse_shift)

        reconstructed_real = np.real(reconstructed)

        reconstructed_uint8 = ((reconstructed_real - reconstructed_real.min()) / (reconstructed_real.max() - reconstructed_real.min()) * 255).astype(np.uint8)

        return fromarray(reconstructed_uint8, mode="L").convert("RGB")
