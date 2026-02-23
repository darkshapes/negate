# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Training utilities for XGBoost-based classification models.

This module provides the core training pipeline using XGBoost gradient boosted
decision trees. It handles feature extraction from wavelet decomposition data,
PCA dimensionality reduction, train-test splitting with stratification, and model
training with early stopping.

The module exports configuration constants for output directories and timestamp generation.

Functions:
    grade: Train XGBoost classifier on wavelet feature dataset.
    generate_datestamp_path: Generate timestamped file paths for model artifacts.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from datasets import Dataset
from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import Booster

from negate.config import Spec

get_time = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
datestamped_folder = Path("models", get_time())
model_path = Path(__file__).parent.parent / "models"

timestamp = get_time()
result_path = Path(__file__).parent.parent / "results" / timestamp


def generate_datestamp_path(file_name) -> str:
    """Generate a filesystem-safe path with embedded timestamp.\n
    :param file_name: Name of the file (not including path components).
    :return: Absolute string path to the timestamped file location.
    :raises OSError: If directory creation fails due to permissions.

    Example:
        >>> path = generate_datestamp_path("model.json")
        >>> print(path)
        models/20241025_143218/model.json
    """

    datestamped_folder.mkdir(parents=True, exist_ok=True)
    generated_path = str(datestamped_folder / file_name)
    return generated_path


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


def prepare_dataset(features_dataset: Dataset, spec: Spec):
    samples = features_dataset["results"]
    all_dicts = [d for row in samples for d in row]

    df = pd.json_normalize(all_dicts).fillna(0)

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():  # type: ignore Invalid series conditional
            df[col] = df[col].apply(lambda x: np.mean(x, dtype=spec.np_dtype) if isinstance(x, (list, np.ndarray)) else float(x))

    feature_matrix = df.to_numpy(dtype=spec.np_dtype)
    feature_matrix = np.where(np.isfinite(feature_matrix), feature_matrix, 0)
    return feature_matrix


def grade(features_dataset: Dataset, spec: Spec) -> TrainResult:
    """Train an XGBoost binary classifier on wavelet feature dataset.\n
    :param features_dataset: HuggingFace Dataset with 'features' (numpy array) and 'label' (int) columns.
    :return: TrainResult containing model, PCA transformer, data matrices, and training metadata.
    :raises RuntimeError: If xgboost is not installed. Install with ``pip install negate[xgb]``.
    :raises ValueError: If dataset is empty or missing required columns.
    :raises numpy.linalg.LinAlgError: If PCA fails (e.g., constant features).\n

    .. note::
        The function uses 80-20 train-test split with stratification to preserve
        class distribution. Early stopping triggers after 10 rounds without improvement.
    """
    feature_matrix = prepare_dataset(features_dataset, spec)

    labels = np.array(features_dataset["label"])

    rng = default_rng(1)
    random_state = lambda: int(np.round(rng.random() * spec.train_rounds.max_rnd))
    seed = spec.hyper_param.seed if spec.hyper_param.seed > 0 else random_state()
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        labels,
        test_size=spec.train_rounds.test_size,
        stratify=labels,
        random_state=seed,
    )

    pca: PCA = PCA(n_components=spec.train_rounds.n_components, random_state=seed)  # dimensionality .95
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(
        f"imbalance ratio: {np.sum(labels == 0) / np.sum(labels == 1):.2f} \
        0: {np.sum(labels == 0)} samples ({np.sum(labels == 0) / len(labels) * 100:.1f}%) \
        1: {np.sum(labels == 1)} samples ({np.sum(labels == 1) / len(labels) * 100:.1f}%)"
    )
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    d_matrix_train = xgb.DMatrix(X_train_pca, label=y_train)
    d_matrix_test = xgb.DMatrix(X_test_pca, label=y_test)

    training_parameters = asdict(spec.hyper_param) | {"scale_pos_weight": scale_pos_weight, "seed": seed}

    evaluation_parameters = [(d_matrix_train, "train"), (d_matrix_test, "test")]
    evaluation_result = {}
    model = xgb.train(
        training_parameters,
        d_matrix_train,
        num_boost_round=spec.train_rounds.num_boost_round,
        evals=evaluation_parameters,
        early_stopping_rounds=spec.train_rounds.early_stopping_rounds,
        evals_result=evaluation_result,
        verbose_eval=spec.train_rounds.verbose_eval,
    )

    return TrainResult(
        X_train=X_train,  # type: ignore
        pca=pca,  # type: ignore
        d_matrix_test=d_matrix_test,  # type: ignore
        model=model,  # type: ignore
        scale_pos_weight=scale_pos_weight,
        X_train_pca=X_train_pca,
        y_test=y_test,  # type: ignore
        labels=labels,
        feature_matrix=feature_matrix,
        seed=seed,
        num_features=model.num_features(),
    )
