# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import tomllib
from pathlib import Path
from typing import NamedTuple


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
    :param feat_ext_path: Folder location of the feature extractor
    :param library: Library to use for feature extraction.
    :param load_onnx: Use ONNX for inference.
    :param magnitude_sampling: Enable magnitude sampling.
    :param model_dtype: Data type for model computation.
    :param model: Model name for feature extraction.
    :param numpy_dtype: Data type for NumPy operations.
    :param patch_dim: Patch dimension for residuals."""

    alpha: float
    batch_size: int
    dim_rescale: int
    feat_ext_path: str
    library: str
    load_onnx: bool
    magnitude_sampling: bool
    model_dtype: str
    model: str
    numpy_dtype: str
    patch_dim: int


class NegateDataPaths(NamedTuple):
    """Dataset config values.\n
    :param evaluation_data: List of evaluation data paths or None.
    :param genuine_data: List of genuine data paths or None.
    :param genuine_local: List of local genuine data paths or None.
    :param synthetic_data: List of synthetic data paths or None.
    :param synthetic_local: List of local synthetic data paths or None."""

    evaluation_data: list[str] | None
    genuine_data: list[str] | None
    genuine_local: list[str] | None
    synthetic_data: list[str] | None
    synthetic_local: list[str] | None


def load_config_options() -> tuple[NegateConfig, NegateHyperParam, NegateDataPaths]:
    """Load configuration options.\n
    :return: Tuple of (NegateConfig, NegateHyperParam, NegateDataPaths)."""

    config_path = Path(__file__).parent.parent / "config" / "config.toml"
    with open(config_path, "rb") as config_file:
        data = tomllib.load(config_file)
    train_cfg = data.pop("train", {})
    dataset_cfg = data.pop("datasets", {})

    return (NegateConfig(**data), NegateHyperParam(**train_cfg), NegateDataPaths(**dataset_cfg))


negate_options, negate_hyperparam, negate_data = load_config_options()
