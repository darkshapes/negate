# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import cv2
import numpy as np
from PIL.Image import Image, fromarray
from torch import Tensor

from negate.config import Spec


class Residual:
    def __init__(self, spec: Spec) -> None:
        """Initialize residual class for residual image processing.\n
        :param dtype: dtype for internal numpy conversion.
        return: None."""

        self.np_dtype = spec.np_dtype
        self.top_k = spec.hyper_param.top_k

    def laplace(self, image: Image) -> Image:
        """Create a 3-channel residual from a grayscale image.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale mode."""

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.np_dtype)
        residual = cv2.Laplacian(numeric_image, ksize=3).astype(self.np_dtype)
        residual_image: Image = fromarray(np.uint8(residual), mode="L")
        return residual_image

    def sobel(self, image: Image) -> Image:
        """Create a 3-channel residual using Sobel edge detection.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale mode."""

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.np_dtype)
        grad_x = cv2.Sobel(numeric_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(numeric_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        residual_image: Image = fromarray(np.uint8(gradient), mode="L")
        return residual_image

    def frequency(self, image: Image) -> Image:
        """Create a 3-channel high-frequency residual using FFT magnitude spectrum.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale mode."""

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.np_dtype)

        fourier_transform = np.fft.fft2(numeric_image)
        fourier_2d_shift = np.fft.fftshift(fourier_transform)

        magnitude_spectrum = 20 * np.log(np.abs(fourier_2d_shift) + 1)

        residual_image: Image = fromarray(np.uint8(magnitude_spectrum), mode="L")
        return residual_image

    def spectral(self, image: Image) -> Image:
        """Create a 3-channel spectral residual using magnitude of frequency domain.\n
        :param image: PIL image to process.
        :return: Residual image in greyscale."""

        gray = image.convert("L")
        numeric_image = np.array(gray, dtype=self.np_dtype)

        f_transform = np.fft.fft2(numeric_image)
        f_shift = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        residual_image: Image = fromarray(np.uint8(magnitude_spectrum), mode="L")
        return residual_image
