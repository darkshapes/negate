# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Convert PIL image to grayscale, RGB, and HSV arrays."""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image as PILImage
from PIL.Image import BICUBIC


_TARGET_SIZE = (255, 255)


class NumericImage:
    """Convert PIL image to grayscale, RGB, and HSV arrays."""

    image: PILImage
    TARGET_SIZE = _TARGET_SIZE

    def __init__(self, image: PILImage) -> None:
        self._image = image
        self.to_gray()
        self.to_rgb()
        self.rgb2hsv()

    @property
    def gray(self) -> NDArray:
        return self.shade

    @property
    def color(self):
        return self.rgb

    @property
    def hsv(self):
        return self._hsv

    def to_gray(self) -> NDArray:
        """Resize and convert to float64 grayscale."""
        img = self._image.convert("L").resize(self.TARGET_SIZE, BICUBIC)
        self.shade = np.asarray(img, dtype=np.float64) / 255.0

    def to_rgb(self) -> NDArray:
        """Resize and convert to float64 RGB [0,1]."""
        img = self._image.convert("RGB").resize(self.TARGET_SIZE, BICUBIC)
        self.rgb = np.asarray(img, dtype=np.float64) / 255.0

    def rgb2hsv(self) -> NDArray:
        """Convert RGB [0,1] array to HSV [0,1]."""
        from colorsys import hsv_to_rgb as rgb_to_hsv

        rgb = self.rgb.copy()
        rgb = rgb / 255.0 if rgb.max() > 1 else rgb
        h, w, c = rgb.shape
        flat = rgb.reshape(-1, 3)
        result = np.array([rgb_to_hsv(r, g, b) for r, g, b in flat])
        self._hsv = result.T.reshape(h, w, 3)
