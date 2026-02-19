# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from datasets import Dataset
from numpy.random import default_rng
from numpy.typing import NDArray

from negate.config import hyperparam_config as hyper_param

get_time = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
datestamped_folder = Path("models", get_time())
model_path = Path(__file__).parent.parent / "models"

timestamp = get_time()
result_path = Path(__file__).parent.parent / "results" / timestamp


def generate_datestamp_path(file_name) -> str:
    datestamped_folder.mkdir(parents=True, exist_ok=True)
    generated_path = str(datestamped_folder / file_name)
    return generated_path


@dataclass
class TrainingParameters:
    """Container holding main model parameters"""

    seed: int = hyper_param.seed
    colsample_bytree: float = hyper_param.colsample_bytree
    eval_metric: list = field(default_factory=lambda: hyper_param.eval_metric)
    learning_rate: float = hyper_param.learning_rate
    max_depth: int = hyper_param.max_depth
    objective: str = hyper_param.objective
    scale_pos_weight: float | None = hyper_param.scale_pos_weight
    subsample: float = hyper_param.subsample


@dataclass
class TrainResult:
    """Container holding all artifacts produced by :func:`grade`."""

    d_matrix_test: NDArray
    feature_matrix: NDArray
    labels: Any
    model: Callable
    num_features: int
    pca: Callable
    scale_pos_weight: float | None
    seed: int
    X_train_pca: NDArray
    X_train: NDArray
    y_test: NDArray


def grade(features_dataset: Dataset) -> TrainResult:
    """Train an XGBoost model from a feature dataset.\n
    :param features_dataset: Dataset of samples containing ``features`` and ``label``.
    :return: TrainResult holding the trained model, PCA, data matrices and metadata."""

    try:
        import xgboost as xgb
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
    except (ImportError, ModuleNotFoundError, Exception):
        raise RuntimeError("missing dependencies for xgboost. Please install using 'negate[xgb]'")

    feature_matrix = np.array([sample["features"] for sample in features_dataset]).astype(np.float32)  # type: ignore
    labels = np.array([sample["label"] for sample in features_dataset])  # type: ignore no overloads

    rng = default_rng(1)
    random_state = lambda: int(np.round(rng.random() * 0xFFFFFFFF))
    seed = random_state()
    params = TrainingParameters(
        seed=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, stratify=labels, random_state=params.seed)

    pca: PCA = PCA(n_components=0.95, random_state=params.seed)  # dimensionality .95
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    params.scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    d_matrix_train = xgb.DMatrix(X_train_pca, label=y_train)
    d_matrix_test = xgb.DMatrix(X_test_pca, label=y_test)

    training_parameters = asdict(params)
    evaluation_parameters = [(d_matrix_train, "train"), (d_matrix_test, "test")]
    evaluation_result = {}

    model = xgb.train(
        training_parameters, d_matrix_train, num_boost_round=200, evals=evaluation_parameters, early_stopping_rounds=10, evals_result=evaluation_result, verbose_eval=20
    )

    return TrainResult(
        X_train=X_train,  # type: ignore
        pca=pca,  # type: ignore
        d_matrix_test=d_matrix_test,  # type: ignore
        model=model,  # type: ignore
        scale_pos_weight=params.scale_pos_weight,
        X_train_pca=X_train_pca,
        y_test=y_test,  # type: ignore
        labels=labels,
        feature_matrix=feature_matrix,
        seed=params.seed,
        num_features=model.num_features(),
    )
