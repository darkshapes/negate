# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb
from datasets import Dataset
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    """Container holding all artifacts produced by :func:`grade`."""

    X_train: Any
    pca: Any
    d_matrix_test: Any
    model: Any
    scale_pos_weight: float
    X_train_pca: Any
    y_test: Any
    labels: Any
    feature_matrix: Any
    seed: int
    num_features: int


def grade(features_dataset: Dataset) -> TrainResult:
    """Train an XGBoost model from a feature dataset.\n
    :param features_dataset: Dataset of samples containing ``features`` and ``label``.
    :return: TrainResult holding the trained model, PCA, data matrices and metadata."""
    feature_matrix = np.array([sample["features"] for sample in features_dataset])  # type: ignore no overloads
    labels = np.array([sample["label"] for sample in features_dataset])  # type: ignore no overloads

    rng = default_rng(1)
    random_state = lambda: int(np.round(rng.random() * 0xFFFFFFFF))
    seed = random_state()
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, stratify=labels, random_state=seed)

    pca: PCA = PCA(n_components=0.95, random_state=seed)  # dimensionality .95
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    d_matrix_train = xgb.DMatrix(X_train_pca, label=y_train)
    d_matrix_test = xgb.DMatrix(X_test_pca, label=y_test)

    training_parameters = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": 4,
        "learning_rate": 0.1,  # keep as float
        "subsample": 0.8,  # keep as float
        "colsample_bytree": 0.8,  # keep as float
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
    }
    evaluation_parameters = [(d_matrix_train, "train"), (d_matrix_test, "test")]
    evaluation_result = {}

    model = xgb.train(
        training_parameters, d_matrix_train, num_boost_round=200, evals=evaluation_parameters, early_stopping_rounds=10, evals_result=evaluation_result, verbose_eval=20
    )

    return TrainResult(
        X_train=X_train,
        pca=pca,
        d_matrix_test=d_matrix_test,
        model=model,
        scale_pos_weight=scale_pos_weight,
        X_train_pca=X_train_pca,
        y_test=y_test,
        labels=labels,
        feature_matrix=feature_matrix,
        seed=seed,
        num_features=model.num_features(),
    )
