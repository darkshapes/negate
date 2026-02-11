# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clase-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Hardware configuration for device and dtype management."""

import numpy as np
import torch


class Chip:
    """Manages device and dtype settings for the package."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._device: torch.device | None = None
        self._dtype: torch.dtype = torch.float16
        self._np_dtype: np.typing.DTypeLike = np.float16
        self.dtype_name: str = "float16"

    @property
    def device(self) -> torch.device:
        """Get the compute device."""

        if self._device is None:
            if torch.cuda.is_available():
                self._device_name = "cuda"
            elif torch.mps.is_available() if hasattr(torch, "mps") else False:
                self._device_name = "mps"
            elif torch.xpu.is_available() if hasattr(torch, "xpu") else False:
                self._device_name = "xpu"
            else:
                self._device_name = "cpu"
                self._dtype = torch.float32
                self._np_dtype = np.float32

        self._device = torch.device(self._device_name)
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get PyTorch dtype."""
        return self._dtype

    @property
    def np_dtype(self) -> np.typing.DTypeLike:
        """Get NumPy dtype."""
        return self._np_dtype

    @device.setter
    def device(self, value: str) -> None:
        """Set device by name with validation."""
        self._device = torch.device(value)
        self._device_name = value

    @dtype.setter
    def dtype(self, value: str) -> torch.dtype:
        """Set all dtype by name with validation."""
        if value == "bfloat16":
            raise NotImplementedError("bfloat16 is not supported")
        self.dtype_name: str = value
        self._dtype = getattr(torch, value, torch.float32)
        self._np_dtype = getattr(np, value, np.float32)
        return self._dtype

    @np_dtype.setter
    def np_dtype(self, value: str) -> np.typing.DTypeLike:
        self.dtype = value
        return self._np_dtype


chip = Chip()
