# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a pes */ -->

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from xgboost import Booster

from negate.io.config import (
    NegateConfig,
    NegateDataPaths,
    NegateHyperParam,
    NegateModelConfig,
    NegateTrainRounds,
    chip,
    data_paths,
    hyperparam_config,
    load_config_options,
    model_config,
    negate_options,
    root_folder,
    train_rounds,
)


@dataclass
class TrainResult:
    """Container holding all artifacts produced by :func:`grade`.

    This dataclass aggregates the trained model, preprocessing transformers,
    training data matrices, and metadata needed for inference or further analysis.

    Attributes:
        d_matrix_test: XGBoost DMatrix with test set features.
        feature_matrix: Full dataset feature array before PCA.
        labels: Complete label vector from original dataset.
        model: Trained XGBoost booster object.
        num_features: Number of features after PCA transformation.
        pca: Fitted sklearn PCA transformer.
        scale_pos_weight: Computed class weight ratio.
        seed: Random seed used for reproducibility.
        X_train_pca: Training set after PCA transform.
        X_train: Original training feature matrix.
        y_test: Test set labels.

    Example:
        >>> result = grade(features_dataset)
        >>> predictions = result.model.predict(result.d_matrix_test)
    """

    d_matrix_test: NDArray
    feature_matrix: NDArray
    labels: Any
    model: Booster
    num_features: int
    pca: PCA
    scale_pos_weight: float | None
    seed: int
    X_train_pca: NDArray
    X_train: NDArray
    y_test: NDArray


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
        self.model_config: NegateModelConfig = model_config


def load_spec(model_version: Path = Path("config")) -> Spec:
    """Load model specification and training metadata.\n
    :param ver_model: Version folder path containing config and results.
    :returns: Updated specification and additional metadata
    """

    if str(model_version) != "config":
        path_result = Path("results") / model_version.stem
    else:
        path_result = model_version
    path_config = str(path_result / "config.toml")
    config_options = load_config_options(path_config)  # load a different config
    spec = Spec(*config_options)
    return spec


def load_metadata(model_version: Path) -> dict[str, Any]:
    """\nLoad serialized training metadata from JSON result file.\n
    :param model_version: Stem of the model version folder.
    :returns: Dictionary containing saved metrics and parameters."""

    results_path = root_folder / "results" / model_version.stem / f"results_{model_version.stem}.json"
    with open(results_path, "rb") as handle:
        metadata = json.load(handle)
    return metadata


def fetch_spec_data(model_version: Path = Path("config")) -> dict[str, Any]:  # unpack metadata, change individual options
    """Load configuration from TOML file in results or config folder.\n
    :param model_version: Subfolder name under results, defaults to 'config'.
    :returns: Dictionary of loaded configuration values."""

    path_conf = root_folder
    if str(model_version) != "config":
        path_result = str(path_conf / "results" / model_version.stem / "config.toml")
    else:
        path_result = str(path_conf / model_version / "config.toml")
    with open(path_result, "rb") as handle:
        metadata = tomllib.load(handle)
    return metadata


def adjust_spec(metadata: dict[str, Any], hyper_param: str | None = None, param_value: int | float | None = None) -> Spec:
    """Reconstruct spec with optional hyperparameter override.
    :param metadata: Base configuration dictionary.
    :param hyper_param: Key name of parameter to modify.
    :param param_value: New value for the hyperparameter.
    :returns: Reconstructed specification object.
    """
    for label in ["model", "vae", "param", "datasets", "library", "rounds", hyper_param]:
        metadata.pop(label)
    config_replacement = NegateConfig(**{str(hyper_param): param_value}, **metadata)
    config_options = load_config_options()
    spec = Spec(config_replacement, *config_options[1:])

    return spec
