# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a pes */ -->

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import torch


class Spec:
    """Manages device, dtype, and other universal settings for the package."""

    def __init__(self) -> None:
        self.data_paths: NegateDataPaths = data_paths
        self.device: torch.device = chip.device
        self.dtype: torch.dtype = chip.dtype
        self.apply: dict[str, torch.device | torch.dtype] = {"device": self.device, "dtype": self.dtype}
        self.np_dtype: np.typing.DTypeLike = chip.np_dtype
        self.hyper_param: NegateHyperParam = hyperparam_config
        self.models: list[str] = [repo for repo in model_config.list_models]
        self.model = model_config.auto_model[0]
        self.opt: NegateConfig = negate_options
        self.data = data_paths
        self.model_config = model_config


class NegateHyperParam(NamedTuple):
    """Training hyperparameter values."""

    n_components: float
    num_boost_round: int
    top_k: int
    early_stopping_rounds: int
    colsample_bytree: float
    eval_metric: list
    learning_rate: float
    max_depth: int
    objective: str
    subsample: float
    scale_pos_weight: float | None
    seed: int
    export_model_path: str | None


class NegateConfig(NamedTuple):
    """Config values."""

    alpha: float
    batch_size: int
    cache_features: bool
    dim_factor: int
    dim_patch: int
    dtype: str
    euclidean: bool
    feat_ext_path: str
    load_onnx: bool
    magnitude_sampling: bool


class NegateDataPaths(NamedTuple):
    """Dataset config values."""

    eval_data: list
    genuine_data: list
    genuine_local: list
    synthetic_data: list
    synthetic_local: list


class NegateModelConfig:
    """Model configuration with library auto-selection."""

    def __init__(self, data: dict, default: str = "timm"):
        self._model_library_map = data
        self._default = default

    @property
    def libraries(self) -> list[str]:
        """All available library names."""
        return list(self._model_library_map.keys())

    @property
    def list_models(self) -> Iterator[str]:
        """All available model names."""
        return (y for _x, y in self._model_library_map["library"].items())

    def models_for_library(self, lib: str | None = None) -> list[str]:
        """Get models for a specific library, or auto-select if None."""
        lib_name = lib or self.auto_library
        return self._model_library_map.get(lib_name, [])

    def library_for_model(self, model_name: str) -> str | None:
        """Find the library that contains the given model.\n
        :param model_name: The model identifier to look up.
        :return: Library name or None if not found."""
        for lib, models in self._model_library_map["library"].items():
            if isinstance(models, list):
                if model_name in models:
                    return lib
            elif models == model_name:
                return lib
        return None

    @property
    def auto_library(self) -> str:
        """Auto-selected library from first configured entry."""
        for lib in ("timm", "openclip", "transformers"):
            if lib in self._model_library_map and self._model_library_map[lib]:
                return lib
        return self._default

    @property
    def auto_model(self) -> str:
        """First model from auto-selected library."""
        models = self.list_models
        return next(iter(models))  # this will fail if model list is missing


class Chip:
    """Hardware configuration for device and dtype management."""

    _instance = None

    def __new__(cls) -> Chip:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
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


def load_config_options() -> tuple[NegateConfig, NegateHyperParam, NegateDataPaths, NegateModelConfig, Chip]:
    """Load configuration options.\n
    :return: Tuple of (NegateConfig, NegateHyperParam, NegateDataPaths)."""

    config_path = Path(__file__).parent.parent / "config" / "config.toml"
    with open(config_path, "rb") as config_file:
        data = tomllib.load(config_file)

    models = data.pop("model")
    train_cfg = data.pop("train", {})
    dataset_cfg = data.pop("datasets", {})
    library_cfg = data.pop("library", {})

    return (
        NegateConfig(**data),
        NegateHyperParam(**train_cfg),
        NegateDataPaths(**dataset_cfg),
        NegateModelConfig(data=models | library_cfg),
        Chip(),
    )


negate_options, hyperparam_config, data_paths, model_config, chip = load_config_options()
