# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
from typing import NamedTuple

import yaml


class NegateConfig(NamedTuple):
    """YAML config values.\n
    :param model_path: Path to the model file.\n
    :param batch_size: Batch size for processing.\n
    :param cache_features: Cache features during inference.\n
    :param default_vae: Default VAE model name.\n
    :param dtype: Data type for VAE feature extraction.\n
    :param n_components: Number of components for dimensionality reduction.\n
    :param num_boost_round: Number of boosting rounds.\n
    :param patch_size: Patch width for residuals.\n
    :param top_k: Number of patches.\n
    :param load_onnx: Use ONNX for inference.\n
    :param vae_slicing: Enable slicing.\n
    :param vae_tiling: Enable tiling.\n
    :param early_stopping_rounds: Early stopping rounds.\n
    :param colsample_bytree: Column sample by tree.\n
    :param eval_metric: Evaluation metrics.\n
    :param learning_rate: Learning rate.\n
    :param max_depth: Maximum depth.\n
    :param objective: Objective function.\n
    :param subsample: Subsample ratio.\n
    :param scale_pos_weight: Scale positive weight.\n
    :param seed: Random seed.\n
    :param genuine_data: List of genuine data paths.\n
    :param synthetic_data: List of synthetic data paths.\n
    :param evaluation_data: List of evaluation data paths.\n
    :return: Config instance."""  # noqa: D401

    alpha: float
    batch_size: int
    cell_dim: int
    library: str
    model_dtype: str
    numpy_dtype: str
    model: str
    magnitude_sampling: bool
    alpha: float
    resize_pct: float
    dim_rescale: int
    evaluation_data: list[str] | None
    genuine_data: list[str] | None
    genuine_local: list[str] | None
    model_path: str | None
    synthetic_data: list[str] | None
    synthetic_local: list[str] | None


def load_config_options() -> NegateConfig:
    """Load YAML configuration options.\n
    :return: Config dict."""

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as config_file:
        data = yaml.safe_load(config_file)
    train_cfg = data.pop("train", {})
    data.update(train_cfg)
    return NegateConfig(**data)


negate_options = load_config_options()
