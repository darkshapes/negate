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

import argparse
import json
import os
from dataclasses import asdict

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from datasets import Dataset


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


from negate.inference import preprocessing
from negate.io.config import root_folder, random_state, get_time, result_path
from negate.io.datasets import build_datasets, prepare_dataset
from negate.io.spec import Spec, fetch_spec_data, adjust_spec, TrainResult
from negate.io.config import datestamped_folder


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
    # labels = np.repeat(labels, 3)  # adjust multiplier based on your data structure
    feature_matrix = prepare_dataset(features_dataset, spec)
    seed = spec.hyper_param.seed if spec.hyper_param.seed > 0 else random_state(spec.train_rounds.max_rnd)
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


def build_train_call(args: argparse.Namespace, path_result: Path, spec: Spec) -> Dataset:
    """Prepare CLI command input for function call.\n
    :param args: Parsed command-line arguments.
    :param path_result: Directory containing training outputs.
    :param spec: Model specification container.
    :returns: A dataset of raw images or precalculated features
    """
    kwargs = {}
    if args.features is not None:
        file_feat = str(path_result / args.features / f"features_{args.features}.json")
        with open(file_feat) as handle:
            features = json.load(handle)
        features_df = pd.DataFrame.from_dict(features)
        features_ds: Dataset = Dataset.from_pandas(features_df)
    else:
        kwargs["genuine_path"] = args.path
        if args.syn is not None:
            kwargs["synthetic_path"] = Path(args.syn)
        spec.model = args.model
        try:
            spec.vae = next(iter(x for x in spec.model_config.list_vae if args.ae in x))
        except StopIteration:
            raise ValueError(f"Invalid VAE choice: {args.ae}")
        kwargs["spec"] = spec
        origin_ds: Dataset = build_datasets(**kwargs)
        features_ds = pretrain(origin_ds, spec)
    return features_ds


def pretrain(image_ds: Dataset, spec: Spec) -> Dataset:
    """Calibration of computing wavelet energy features.\n
    :param ds_orig: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :returns: Dataset of extracted image features
    """
    features_ds = preprocessing(image_ds, spec=spec)
    return features_ds


def train_model(spec: Spec, features_ds: Dataset) -> TrainResult:
    """Train XGBoost model on preprocessed image features.\n
    :param spec: Specification container.
    """
    train_result = grade(features_ds, spec)
    return train_result


def training_loop(image_ds: Dataset, spec: Spec) -> None:
    """Train models across a range of hyperparameter values.\n
    :param ds_orig: Input dataset for training.
    """
    print("looping")

    def parse_num(val):
        """Try int first, fallback to float."""
        try:
            return int(val)
        except ValueError:
            return float(val)

    hyper_param = input("enter name of hyperparameter:")
    step_val = parse_num(input("enter increment"))
    start, end = map(parse_num, input("enter start and end values separated by comma").split(","))

    param_value = start
    metadata = fetch_spec_data()
    spec = adjust_spec(metadata=metadata, param_value=param_value, hyper_param=hyper_param)
    while param_value < end:
        path_loop = root_folder / "results" / get_time()
        features_ds = pretrain(image_ds=image_ds, spec=spec)
        train_model(features_ds=features_ds, spec=spec)
        os.rename(result_path, path_loop)
        param_value += step_val
