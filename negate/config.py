# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a pes */ -->

"""
Configuration management for the Negate package.

This module provides centralized configuration handling including hardware settings
(device selection, data types), model configurations, training hyperparameters,
and dataset paths. It implements a singleton pattern for hardware detection and
lazy-loading of configuration values from TOML files.

Classes:
    Spec: Main specification container aggregating all configuration objects.
    Chip: Hardware abstraction layer for device and dtype management.
    NegateConfig: Core runtime settings (batch size, VAE options).
    NegateHyperParam: Training hyperparameters for XGBoost model.
    NegateDataPaths: Dataset directory paths configuration.
    NegateModelConfig: Vision transformer and VAE model registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import tomllib
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import torch


class NegateTrainRounds(NamedTuple):
    """Training hyperparameter values."""

    early_stopping_rounds: int
    export_model_path: str | None
    max_rnd: int
    n_components: float
    num_boost_round: int
    test_size: float
    verbose_eval: int


@dataclass
class NegateHyperParam:
    """Container holding XGBoost model hyperparameters.

    Attributes:
        seed: Random seed for reproducibility in splitting and PCA.
        colsample_bytree: Fraction of features to sample per tree.
        eval_metric: List of evaluation metrics ['auc', 'error'].
        learning_rate: Step size shrinkage to prevent overfitting.
        max_depth: Maximum depth of decision trees.
        objective: Learning objective (binary:logistic).
        subsample: Fraction of training data to sample per tree.

    Example:
        >>> params = TrainingParameters(learning_rate=0.05, max_depth=6)
        >>> print(params.seed)
        12345
    """

    seed: int
    colsample_bytree: float
    eval_metric: list
    learning_rate: float
    max_depth: int
    objective: str
    subsample: float


class NegateConfig(NamedTuple):
    """Config values."""

    alpha: float
    batch_size: int
    dim_factor: list[int]
    dim_patch: int
    dtype: str
    feat_ext_path: str
    load_onnx: bool
    magnitude_sampling: bool
    residual_dtype: str
    top_k: int


class NegateDataPaths(NamedTuple):
    """Dataset config values."""

    eval_data: list
    genuine_data: list
    genuine_local: list
    synthetic_data: list
    synthetic_local: list


class NegateModelConfig:
    """Model configuration with library auto-selection."""

    def __init__(self, data: dict, vae: dict, default: str = "timm"):
        self._model_library_map = data
        self._default = default
        self._vae = vae

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

    @property
    def list_vae(self) -> Iterator[list[str]]:
        """All available model names."""
        return (y for _x, y in self._vae["library"].items())

    @property
    def auto_vae(self) -> list[str]:
        """First vae from auto-selected library.
        :returns: List of vae name and import library"""
        vae = self.list_vae
        return next(iter(vae))


class Chip:
    """Hardware abstraction layer for device and dtype management.

    Implements a singleton pattern to ensure consistent hardware configuration
    across the application. Automatically detects available compute devices
    (CUDA, MPS, XPU) and selects appropriate data types.

    Attributes:
        device: The current compute device (cuda, mps, xpu, cpu).
        dtype: PyTorch tensor data type.
        np_dtype: NumPy array data type.
        dtype_name: String name of the current dtype.

    Example:
        >>> chip = Chip()
        >>> print(chip.device)
        cuda
        >>> chip.dtype = "float32"
    """

    _instance = None

    def __new__(cls) -> Chip:
        """Return singleton instance or create new one."""

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._dtype: torch.dtype = torch.float16
        self._np_dtype: np.typing.DTypeLike = np.float64
        self.dtype_name: str = "float16"

    @property
    def device(self) -> torch.device:
        """Get the compute device, auto-detecting available hardware.

        Checks for CUDA, MPS (Apple Silicon), XPU (Intel XPU), and falls back to CPU.
        Automatically adjusts dtype to float32 on CPU for compatibility.

        :return: PyTorch device object.

        :raises RuntimeError: If no supported device is available.
        """

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
                self._np_dtype = np.float64

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
        self._np_dtype = getattr(np, value, np.float64)
        return self._dtype

    @np_dtype.setter
    def np_dtype(self, value: str) -> np.typing.DTypeLike:
        self.dtype = value
        return self._np_dtype


def load_config_options(file_path_named: str = f"config{os.sep}config.toml") -> tuple[NegateConfig, NegateHyperParam, NegateDataPaths, NegateModelConfig, Chip, NegateTrainRounds]:
    """Load configuration options.\n
    :return: Tuple of (NegateConfig, NegateHyperParam, NegateDataPaths)."""

    config_path = Path(__file__).parent.parent / file_path_named
    with open(config_path, "rb") as config_file:
        data = tomllib.load(config_file)

    models = data.pop("model")
    vae = data.pop("vae")
    param_cfg = data.pop("param", {})
    dataset_cfg = data.pop("datasets", {})
    library_cfg = data.pop("library", {})
    rounds_cfg = data.pop("rounds", {})

    return (
        NegateConfig(**data),
        NegateHyperParam(**param_cfg),
        NegateDataPaths(**dataset_cfg),
        NegateModelConfig(data=models | library_cfg, vae=vae),
        Chip(),
        NegateTrainRounds(**rounds_cfg),
    )


negate_options, hyperparam_config, data_paths, model_config, chip, train_rounds = load_config_options()


class Spec:
    """Main specification container aggregating all configuration objects.

    This class serves as the central access point for all package configuration,
    combining settings from NegateConfig, NegateHyperParam, NegateDataPaths,
    and hardware detection via Chip. It is typically instantiated once at
    application startup.

    Attributes:
        opt: Core runtime configuration (batch size, VAE options).
        hyper_param: Training hyperparameters.
        data_paths: Dataset directory paths.
        model_config: Model registry for vision transformers and VAEs.
        chip: Hardware abstraction layer.

    Example:
        >>> spec = Spec()
        >>> print(f"Using device: {spec.chip.device}")
        Using device: cuda
    """

    def __init__(
        self,
        negate_options=negate_options,
        hyperparam_config=hyperparam_config,
        data_paths=data_paths,
        model_config=model_config,
        chip=chip,
        train_rounds=train_rounds,
    ) -> None:
        """Initialize specification container with loaded configuration."""

        self.data_paths: NegateDataPaths = data_paths
        self.device: torch.device = chip.device
        self.dtype: torch.dtype = chip.dtype
        self.apply: dict[str, torch.device | torch.dtype] = {"device": self.device, "dtype": self.dtype}
        self.np_dtype: np.typing.DTypeLike = chip.np_dtype
        self.hyper_param: NegateHyperParam = hyperparam_config
        self.train_rounds: NegateTrainRounds = train_rounds
        self.models: list[str] = [repo for repo in model_config.list_models]
        self.model = model_config.auto_model[0]
        self.vae = model_config.auto_vae
        self.opt: NegateConfig = negate_options
        self.data = data_paths
        self.model_config = model_config
