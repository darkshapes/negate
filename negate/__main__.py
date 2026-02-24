# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Command-line interface entry point for Negate package.\n
Handles CLI parsing, dataset loading, preprocessing, and result saving.
Supports 'predict' subcommand with automatic timestamping.
"""

import argparse
import json
import pickle
import re
import shutil
import time as timer_module
from dataclasses import asdict
from pathlib import Path
from sys import argv
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
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
    graph_train_variance,
    load_config_options,
    prepare_dataset,
    result_path,
    save_features,
    save_metadata,
    save_models,
)
from negate.save import save_to_onnx
from negate.track import accuracy

start_ns = timer_module.perf_counter()


def run_native(features_dataset: np.ndarray, model_version: Path, parameters: dict) -> np.ndarray:
    """Run inference using XGBoost with PCA pre-processing.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Result array of predicted origins."""

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

    model = xgb.Booster(params=parameters)
    model.load_model(model_file_path_named)

    result = model.predict(xgb.DMatrix(features_pca))
    return result


def run_onnx(features_dataset: np.ndarray, model_version: Path, parameters: dict) -> np.ndarray | Any:
    """Run inference using ONNX Runtime with PCA pre-processing.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Result array of predicted origins."""

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


def preprocessing(dataset: Dataset, spec: Spec, forward: bool = False) -> Dataset:
    """Apply wavelet analysis transformations to dataset.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Transformed dataset with 'features' column."""

    kwargs = {}
    if spec.opt.batch_size > 0:
        kwargs["batched"] = True
        kwargs["batch_size"] = spec.opt.batch_size

    context = WaveletContext(spec)
    with WaveletAnalyze(context) as analyzer:  # type: ignore
        if forward:
            dataset = dataset.map(
                analyzer.forward,
                remove_columns=["image"],
                desc="Computing wavelets...",
                **kwargs,
            )
        else:
            dataset = dataset.map(
                analyzer,
                remove_columns=["image"],
                desc="Computing wavelets...",
                **kwargs,
            )
    save_features(dataset)
    return dataset


def measure_origin(image_path: Path, label: bool | None = None) -> None:
    """\nMeasure model performance on a single image.\n
    :param image_path: Path to input image.
    :param label: Optional ground truth label.
    """
    from negate import classify_gnf_or_syn

    dataset: Dataset = generate_dataset(image_path)
    spec = Spec()
    features_dataset = preprocessing(dataset, spec=spec, forward=True)
    json_path = save_features(features_dataset)
    probabilities = classify_gnf_or_syn(json_path)
    print(features_dataset.description)
    print(probabilities)


def infer_origin(image_path: Path, model_version: Path, label: bool | None = None) -> tuple[np.ndarray, ...]:
    """Predict synthetic or original for given image.\n
    :param image_path: Path to image file or folder.
    :param model_version: Model version path.
    :return: Prediction arrays (0=genuine, 1=synthetic)."""

    print(f"""Checking path '{image_path}' using model date {model_version.stem}""")

    dataset: Dataset = generate_dataset(image_path)
    results_path = Path("results") / model_version.stem
    config_path = str(results_path / "config.toml")
    config_options = load_config_options(config_path)
    spec = Spec(*config_options)
    results_file_path = str(results_path / f"results_{model_version.stem}.json")
    with open(results_file_path) as result_metadata:
        train_metadata = json.load(result_metadata)
    spec.hyper_param.seed = train_metadata["seed"]
    features_dataset = preprocessing(dataset, spec)
    features_matrix = prepare_dataset(features_dataset, spec)

    parameters = asdict(spec.hyper_param) | {"scale_pos_weight": train_metadata["scale_pos_weight"]}

    result = run_onnx(features_matrix, model_version, parameters) if spec.opt.load_onnx else run_native(features_matrix, model_version, parameters=parameters)

    thresh = 0.5
    predictions = (result > thresh).astype(int)
    if label is not None:
        ground_truth = np.full(predictions.shape, label, dtype=int)
        acc = float(np.mean(predictions == ground_truth))
        print(f"Accuracy: {acc:.2%}")
    print(result)
    print(predictions)
    return result, predictions  # type: ignore[return-value]


def end_processing(process_name: str) -> float:
    """Backup config file and complete process timer.\n
    :param process_name: The type of process completing.
    :returns: Timecode of the elapsed computation time."""

    timecode = timer_module.perf_counter() - start_ns
    result_path.mkdir(parents=True, exist_ok=True)
    config_name = "config.toml"
    shutil.copy(str(Path(__file__).parent.parent / "config" / config_name), str(result_path / config_name))
    print(f"{process_name} completed in {timecode}")
    return timecode


def pretrain(spec: Spec, file_or_folder_path: Path | None = None) -> None:
    """Calibration of computing wavelet energy features.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :param file_or_folder_path: Additional datasets folder."""

    print("Metrics selected.")
    dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
    print("Beginning preprocessing.")
    features_dataset = preprocessing(dataset, spec=spec)
    end_processing("Pretraining")
    chart_decompositions(features_dataset=features_dataset, spec=spec)


def train_model(spec: Spec, file_or_folder_path: Path | None = None, features: Dataset | None = None) -> None:
    """Train XGBoost model on preprocessed image features.\n
    :param spec: Specification container.
    :param file_or_folder_path: Optional dataset path."""

    print("Training selected.")

    if features is None:
        dataset: Dataset = build_datasets(genuine_folder=file_or_folder_path, spec=spec)
        print("Beginning preprocessing.")
        features_dataset = preprocessing(dataset, spec=spec, forward=True)
        print("Training decision tree.")
    else:
        features_dataset = features
    train_result = grade(features_dataset, spec)
    timecode = end_processing("Training")
    save_metadata(train_result)
    save_models(train_result, compare=False)
    save_to_onnx(train_result)

    accuracy(train_result=train_result, timecode=timecode)
    chart_decompositions(features_dataset=features_dataset, spec=spec)
    graph_train_variance(train_result=train_result, spec=spec)


def training_loop(dataset_location):
    """Train models across a range of hyperparameter values.\n
    :param dataset_location: Path to training dataset.
    """
    print("looping")
    import os
    import tomllib

    from negate.config import NegateConfig
    from negate.train import get_time

    def parse_num(val):
        """Try int first, fallback to float."""
        try:
            return int(val)
        except ValueError:
            return float(val)

    hyper_param = input("enter name of hyperparameter:")
    increment = parse_num(input("enter increment"))
    start, end = map(parse_num, input("enter start and end values separated by comma").split(","))

    param_value = start
    while param_value < end:
        config_path = Path(__file__).parent.parent / "config" / "config.toml"
        with open(config_path, "rb") as config_file:
            data = tomllib.load(config_file)
        for label in ["model", "vae", "param", "datasets", "library", "rounds", hyper_param]:
            data.pop(label)
        config_replacement = NegateConfig(**{str(hyper_param): param_value}, **data)
        config_options = load_config_options()
        spec = Spec(config_replacement, *config_options[1:])
        loop_result_path = Path(__file__).parent.parent / "results" / get_time()
        train_model(file_or_folder_path=dataset_location, spec=spec)
        os.rename(result_path, loop_result_path)
        param_value += increment


def main() -> None:
    """CLI argument parser and command dispatcher.\n
    :raises ValueError: Missing image path.
    :raises ValueError: Invalid VAE choice.
    :raises NotImplementedError: Unsupported command passed."""

    pretrain_blurb = "Analyze and graph performance of image preprocessing on the image dataset at the provided path from CLI and config paths, default `assets/`."
    train_blurb = "Train XGBoost model on preprocessed image features using the image dataset in the provided path or `assets/`. The resulting model will be saved to disk."
    infer_blurb = "Infer whether an image at the provided path is synthetic or original."
    dataset_blurb = "Genunie/Human-original image dataset path"
    infer_path_blurb = "Path to the image or directory containing images of unknown origin"
    measure_blurb = "Measure defining features of synthetic or genuine images at the provided path."
    spec = Spec()

    model_blurb = f"Model to use. Default :{spec.model}"  # type: ignore
    model_choices = [repo for repo in spec.models]
    ae_choices = [ae[0] for ae in spec.model_config.list_vae]
    ae_choices.append("")
    trained_model_folder = Path(__file__).parent.parent / "models"
    results_folder = Path(__file__).parent.parent / "results"
    trained_model_list = [
        str(folder.stem)  # for formatting
        for folder in Path(trained_model_folder).iterdir()
        if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)
    ]
    results_list = [
        str(folder.stem)  # for formatting
        for folder in Path(results_folder).iterdir()
        if folder.is_dir() and re.fullmatch(r"\d{8}_\d{6}", folder.stem)
    ]
    trained_model_list.sort(reverse=True)

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pretrain_parser = subparsers.add_parser("pretrain", help=pretrain_blurb)
    train_parser = subparsers.add_parser("train", help=train_blurb)
    train_parser.add_argument("-l", "--loop", action="store_true", help="Loop training iterations across hyperparameter settings")
    train_parser.add_argument("-f", "--features", choices=results_list, default=None, help="Train from an existing set of features")

    measure_parser = subparsers.add_parser("measure", help=measure_blurb)
    measure_parser.add_argument("path", help=infer_path_blurb)

    infer_parser = subparsers.add_parser("infer", help=infer_blurb)
    infer_parser.add_argument("path", help=infer_path_blurb)
    infer_parser.add_argument("-m", "--model", choices=trained_model_list, default=trained_model_list[0])

    for sub_parser in [pretrain_parser, train_parser]:
        sub_parser.add_argument("path", help=dataset_blurb, nargs="?", default=None)
        sub_parser.add_argument("-m", "--model", choices=model_choices, default=spec.model, help=model_blurb)  # type: ignore
        sub_parser.add_argument("-a", "--ae", choices=ae_choices, default=spec.model_config.auto_vae[0], help=model_blurb)  # type: ignore

    for sub_parser in [infer_parser, measure_parser]:
        label_grp = sub_parser.add_mutually_exclusive_group()
        label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help="Mark image as synthetic (label = 1) for evaluation.")
        label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help="Mark image as genuine (label = 0) for evaluation.")

    args = parser.parse_args(argv[1:])

    def build_call():
        """Prepare CLI command input for function call"""
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
            features = None
            dataset_location = build_call()
            if args.loop is True:
                training_loop(dataset_location)
            elif args.features is not None:
                features_file_path = str(results_folder / args.features / f"features_{args.features}.json")
                with open(features_file_path) as features_file:
                    features = json.load(features_file)
                features_dataframe = pd.DataFrame.from_dict(features)
                features = Dataset.from_pandas(features_dataframe)
            else:
                train_model(file_or_folder_path=dataset_location, spec=spec, features=features)
        case "infer":
            if args.path is None:
                raise ValueError("Infer requires an image path.")

            image_path: Path = Path(args.path)
            model_version = trained_model_folder / args.model
            infer_origin(image_path=image_path, model_version=model_version, label=args.label)
        case "measure":
            if args.path is None:
                raise ValueError("Measure requires an image path.")

            image_path: Path = Path(args.path)
            measure_origin(image_path=image_path, label=args.label)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
