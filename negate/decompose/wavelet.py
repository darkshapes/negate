# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Wavelet-based feature extraction for AI detection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from datasets import Dataset

from negate.io.spec import Spec


class WaveletContext:
    """Context for wavelet-based feature extraction."""

    def __init__(self, spec: Spec, verbose: bool = True) -> None:
        """Initialize wavelet context with configuration.

        :param spec: Specification container with model config and hardware settings.
        :param verbose: Whether to print progress messages.
        """
        self.spec = spec
        self.verbose = verbose
        self._image: Image.Image | None = None
        self._wavelet_coeffs: NDArray | None = None

    def set_image(self, image: Image.Image) -> None:
        """Set the image to analyze.

        :param image: PIL image to process.
        """
        self._image = image

    def get_wavelet(self) -> NDArray:
        """Get wavelet coefficients.

        :returns: 2D wavelet coefficient array.
        """
        if self._image is None:
            raise ValueError("No image set")
        if self._wavelet_coeffs is None:
            gray = np.array(self._image.convert("L"))
            self._wavelet_coeffs = self._compute_wavelet(gray)
        return self._wavelet_coeffs

    def _compute_wavelet(self, gray: NDArray) -> NDArray:
        """Compute 2D wavelet transform.

        :param gray: Grayscale image array.
        :returns: 2D wavelet coefficient array.
        """
        import pywt

        wavelet = pywt.Wavelet("haar")  # type: ignore[has-type]
        coeffs = pywt.dwt2(gray, wavelet, mode="reflect")
        return np.array([coeffs[0], coeffs[1]])

    def analyze(self, dataset: Dataset) -> dict[str, Any]:
        """Analyze wavelet features across dataset.

        :param dataset: HuggingFace Dataset with 'image' column.
        :returns: Dictionary with analysis results.
        """
        results = []
        for image in dataset["image"]:
            try:
                gray = np.array(image.convert("L"))
                coeffs = self._compute_wavelet(gray)
                results.append({"coeffs": coeffs.tolist()})
            except Exception:
                results.append({"coeffs": []})
        return {"results": results}


class WaveletAnalyze:
    """Analyze wavelet features for AI detection."""

    def __init__(self, context: WaveletContext) -> None:
        """Initialize wavelet analyzer.

        :param context: Wavelet context instance.
        """
        self.context = context

    def __call__(self, dataset: Dataset) -> dict[str, Any]:
        """Analyze wavelet features.

        :param dataset: HuggingFace Dataset with 'image' column.
        :returns: Dictionary with analysis results.
        """
        return self.context.analyze(dataset)
