# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Command-line interface entry point for the Negate package.

This module provides the CLI parser and main execution logic for training
and inference workflows. It handles argument parsing, dataset loading,
preprocessing orchestration, and result saving.

The CLI supports a 'predict' subcommand for running inference on image datasets.
Results are automatically timestamped and saved to the results directory.

Functions:
    preprocessing: Apply wavelet analysis transformations to dataset.
    multi_prediction: Main prediction workflow orchestrator.
    main: CLI argument parser and command dispatcher.
"""

import argparse
import datetime
import pickle
import re
import shutil
import time as timer_module
from pathlib import Path
from sys import argv
from typing import Any

import numpy as np
import onnxruntime as ort
from datasets import Dataset
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ONNXRuntimeError
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from negate import (
    Spec,
    WaveletAnalyze,
    WaveletContext,
    build_datasets,
    chart_decompositions,
    generate_dataset,
    grade,
    load_config_options,
    prepare_dataset,
    result_path,
    save_metadata,
    save_models,
)
from negate.save import save_to_onnx
from negate.track import accuracy

start_ns = timer_module.perf_counter()
process_time = lambda: print(str(datetime.timedelta(seconds=timer_module.process_time())), end="")


def run_native(features_dataset: np.ndarray, model_version: Path) -> np.ndarray:
    """Run inference using XGBoost with PCA pre-processing.\n
    :param features_dataset: Extracted features.\n
    :param model_version: Model to use for the prediction\n
    :return: Prediction array."""

    import xgboost as xgb

    model_file_path_named = model_version / "negate.ubj"

    if not model_file_path_named.exists():
        raise FileNotFoundError(f"Model file not found: {str(model_file_path_named)}. Please run 'train' first to create the model.")
    else:
        model_file_path_named = str(model_file_path_named)

    pca_file_path_named = model_version / "negate_pca.pkl"
    with open(pca_file_path_named, "rb") as pca_file:
        pca = pickle.load(pca_file)

    features_pca = pca.transform(features_dataset)

    model = xgb.Booster()
    model.load_model(model_file_path_named)

    result = model.predict(xgb.DMatrix(features_pca))
    print(result)
    return result


def run_onnx(features_dataset: np.ndarray, model_version: Path) -> np.ndarray | Any:
    """Run inference using ONNX Runtime with PCA pre-processing.\n
    :param features_array: Feature array.\n
    :return: Prediction array."""

    model_file_path_named = model_version / "negate.onnx"
    if not model_file_path_named.exists():
        raise FileNotFoundError(f"Model file not found: {str(model_file_path_named)}. Please run 'train' first to create the model.")
    else:
        model_file_path_named = str(model_file_path_named)

    pca_file_path_named = model_version / "negate_pca.onnx"
    session_pca = ort.InferenceSession(pca_file_path_named)
    input_name_pca = session_pca.get_inputs()[0].name
    features_pca = session_pca.run(None, {input_name_pca: features_dataset})[0]

    # # input_name = ort.get_available_providers()[0]
    # pca_file_path_named = model_version / "negate_pca.pkl"
    # with open(pca_file_path_named, "rb") as pca_file:
    #     pca = pickle.load(pca_file)

    # features_pca = pca.transform(features_dataset)
    features_model = features_pca.astype(np.float32)  # type: ignore

    session = ort.InferenceSession(model_file_path_named)
    print(f"Model '{model_file_path_named}' loaded.")
    input_name = session.get_inputs()[0].name
    try:
        result = session.run(None, {input_name: features_model})[0]  # type: ignore
        print(result)
        return result
    except (InvalidArgument, ONNXRuntimeError) as error_log:
        import sys

        print(error_log)
        sys.exit()


def preprocessing(dataset: Dataset, spec: Spec) -> Dataset:
    """Apply wavelet decomposition analysis to dataset images.\n
    :param dataset: HuggingFace Dataset containing 'image' column with PIL images.
    :param spec: Specification container with analysis configuration.
    :return: Transformed dataset with 'features' column containing wavelet coefficients.\n
    :raises KeyError: If dataset missing required 'image' column.
    :raises RuntimeError: If wavelet analyzer fails to initialize.

    .. note::
        The preprocessing is memory-efficient and supports batched processing
        when config.toml `batch_size` > 0.
    """
    kwargs = {}
    if spec.opt.batch_size > 0:
        kwargs["batched"] = True
        kwargs["batch_size"] = spec.opt.batch_size

    context = WaveletContext(spec)
    with WaveletAnalyze(context) as analyzer:  # type: ignore
        dataset = dataset.map(
            analyzer,
            remove_columns=["image"],
            desc="Computing wavelets...",
            **kwargs,
        )
    result_path.mkdir(parents=True, exist_ok=True)
    config_name = "config.toml"
    shutil.copy(str(Path(__file__).parent.parent / "config" / config_name), str(result_path / config_name))

    return dataset


def infer_origin(image_path: Path, model_version: Path) -> tuple:
    """Predict synthetic or original for given image. (0 = genuine, 1 = synthetic)\n
    :param image_path: Path to image file or folder.
    :param vae_type: VAE model to use for feature extraction.
    :return: Prediction array."""

    print(f"""Checking path '{image_path}' with {model_version.stem}""")

    dataset: Dataset = generate_dataset(image_path)
    config_path = str(Path("results") / model_version.stem / "config.toml")
    config_options = load_config_options(config_path)
    spec = Spec(*config_options)
    features_dataset = preprocessing(dataset, spec)
    features_matrix = prepare_dataset(features_dataset, spec)
    result = run_onnx(features_matrix, model_version) if spec.opt.load_onnx else run_native(features_matrix, model_version)

    thresh = 0.5
    predictions = (result > thresh).astype(int)
    # ground_truth = np.full(predictions.shape, true_label, dtype=int)
    # acc = float(np.mean(predictions == ground_truth))
    # print(f"Accuracy: {acc:.2%}")
    print(result)
    print(predictions)
    return result, predictions  # type: ignore[return-value]


def pretrain(spec: Spec, file_or_folder_path: Path | None = None) -> None:
    """Calibration of computing wavelet energy features.\n
    :param file_or_folder_path: Additional datasets folder."""

    print("Metrics selected.")
    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    print("Beginning preprocessing.")
    features_dataset = preprocessing(dataset, spec=spec)
    timecode = timer_module.perf_counter() - start_ns
    chart_decompositions(features_dataset=features_dataset, spec=spec)
    print(f"Metrics completed in {timecode}")


def train_model(spec: Spec, file_or_folder_path: Path | None = None) -> None:
    print("Training selected.")

    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    print("Beginning preprocessing.")
    features_dataset = preprocessing(dataset, spec=spec)
    print("Training decision tree.")
    train_result = grade(features_dataset, spec)
    timecode = timer_module.perf_counter() - start_ns
    print(f"Training completed in {timecode}")
    save_metadata(train_result)
    save_models(train_result, compare=False)
    save_to_onnx(train_result)
    accuracy(train_result=train_result, timecode=timecode)
    chart_decompositions(features_dataset=features_dataset, spec=spec)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""

    pretrain_blurb = "Analyze and graph performance of image preprocessing on the image dataset at the provided path from CLI and config paths, default `assets/`."
    train_blurb = "Train XGBoost model on preprocessed image features using the image dataset in the provided path or `assets/`. The resulting model will be saved to disk."
    infer_blurb = "Infer whether an image at the provided path is synthetic or original."
    dataset_blurb = "Genunie/Human-original image dataset path"
    infer_path_blurb = "Path to the image or directory containing images of unknown origin"

    spec = Spec()

    model_blurb = f"Model to use. Default :{spec.model}"  # type: ignore
    model_choices = [repo for repo in spec.models]
    ae_choices = [ae[0] for ae in spec.model_config.list_vae]
    ae_choices.append("")
    trained_model_folder = Path(__file__).parent.parent / "models"
    trained_model_list = [
        str(folder.stem)  # for formatting
        for folder in Path(trained_model_folder).iterdir()
        if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)
    ]
    trained_model_list.sort(reverse=True)

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pretrain_parser = subparsers.add_parser("pretrain", help=pretrain_blurb)
    pretrain_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    pretrain_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore
    pretrain_parser.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)  # type: ignore

    train_parser = subparsers.add_parser("train", help=train_blurb)
    train_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
    train_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore
    train_parser.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)  # type: ignore

    infer_parser = subparsers.add_parser("infer", help=infer_blurb)
    infer_parser.add_argument("path", help=infer_path_blurb)
    infer_parser.add_argument("-m", "--model", choices=trained_model_list, default=trained_model_list[0])

    args = parser.parse_args(argv[1:])

    def build_call():
        if args.path:
            dataset_location: Path | None = Path(args.path)
        else:
            dataset_location: Path | None = None
        spec.model = args.model
        try:
            spec.vae = next(iter(x for x in spec.model_config.list_vae if args.ae in x))
        except StopIteration:
            raise ValueError(f"Invalid VAE choice: {args.ae}")
        return dataset_location

    match args.cmd:
        case "pretrain":
            dataset_location = build_call()
            pretrain(file_or_folder_path=dataset_location, spec=spec)
        case "train":
            dataset_location = build_call()
            train_model(file_or_folder_path=dataset_location, spec=spec)
        case "infer":
            if args.path is None:
                raise ValueError("Check requires an image path.")

            image_path: Path = Path(args.path)
            model_version = trained_model_folder / args.model
            infer_origin(image_path=image_path, model_version=model_version)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
