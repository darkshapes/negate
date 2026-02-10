# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a pes */ -->

import tomllib
from pathlib import Path
from typing import Iterator, NamedTuple


class NegateHyperParam(NamedTuple):
    """Training hyperparameter values.\n
    :param n_components: Number of components for dimensionality reduction.
    :param num_boost_round: Number of boosting rounds.
    :param top_k: Number of patches.
    :param early_stopping_rounds: Early stopping rounds.
    :param colsample_bytree: Column sample by tree.
    :param eval_metric: Evaluation metrics.
    :param learning_rate: Learning rate.
    :param max_depth: Maximum depth.
    :param objective: Objective function.
    :param subsample: Subsample ratio.
    :param scale_pos_weight: Scale positive weight or None.
    :param seed: Random seed.
    :param export_model_path: Export model path or None."""

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
    """Config values.\n
    :param alpha: Regularization parameter.
    :param batch_size: Batch size for processing.
    :param dim_rescale: Dimension for rescaling.
    :param dtype: Data type for model and NumPy operations.
    :param feat_ext_path: Folder location of the feature extractor
    :param load_onnx: Use ONNX for inference.
    :param magnitude_sampling: Enable magnitude sampling.
    :param patch_dim: Patch dimension for residuals."""

    alpha: float
    batch_size: int
    dim_patch: int
    dim_rescale: int
    dtype: str
    euclidean: bool
    feat_ext_path: str
    load_onnx: bool
    magnitude_sampling: bool


class NegateDataPaths(NamedTuple):
    """Dataset config values.\n
    :param evaluation_data: List of evaluation data paths or None.
    :param genuine_data: List of genuine data paths or None.
    :param genuine_local: List of local genuine data paths or None.
    :param synthetic_data: List of synthetic data paths or None.
    :param synthetic_local: List of local synthetic data paths or None."""

    eval_data: list
    genuine_data: list
    genuine_local: list
    synthetic_data: list
    synthetic_local: list


class NegateModelConfig:
    """Model configuration with library auto-selection.\n
    :param _data: Raw config data.
    :param _default_lib: Default library name."""

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
    def auto_model(self) -> str | None:
        """First model from auto-selected library."""
        models = self.list_models
        return next(iter(models), None)


def load_config_options() -> tuple[NegateConfig, NegateHyperParam, NegateDataPaths, NegateModelConfig]:
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
    )


negate_options, hyperparam_config, data_paths, model_config = load_config_options()
