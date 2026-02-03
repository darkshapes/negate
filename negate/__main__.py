# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
from typing import Any

import numpy as np

from negate import TrainResult, VAEModel, build_datasets, datestamped_folder, features, generate_dataset, grade, in_console, model_path, on_graph, save_to_onnx


def evaluate(prediction: np.ndarray, ground_truth: np.ndarray) -> None:
    """Print accuracy and class distribution.\n
    :param prediction: Model outputs (0 = genuine, 1 = synthetic).\n
    :param ground_truth: Ground-truth labels.\n
    :return: None."""

    prediction = prediction.astype(int)
    ground_truth = ground_truth.astype(int)

    acc = float(np.mean(prediction == ground_truth))

    print(f"Accuracy: {acc:.2%}")


def predict(image_path: Path, vae_type: VAEModel, metadata: dict[str, Any], true_label: int | None = None) -> np.ndarray:
    """Predict synthetic or original for given image. (0 = genuine, 1 = synthetic)\n
    :param image_path: Path to image file or folder.
    :param vae_type: VAE model to use for feature extraction.
    :return: Prediction array."""

    import onnxruntime as ort
    from datasets import Dataset
    from onnxruntime import SparseTensor
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ONNXRuntimeError
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    pca_file_path_named = model_path / "negate_pca.onnx"
    model_file_path_named = model_path / "negate.onnx"
    if not model_file_path_named.exists():
        raise FileNotFoundError(f"Model file not found: {str(model_file_path_named)}. Please run 'train' first to create the model.")
    else:
        model_file_path_named = str(model_file_path_named)
    print(f"""{"Evaluation" if true_label is not None else "Detection"} selected.
Checking path '{image_path}' with {vae_type.value}""")

    dataset: Dataset = generate_dataset(image_path)
    features_dataset: Dataset = features(dataset, vae_type)
    features_array = np.array(features_dataset["features"]).astype(np.float32)  # type: ignore[arg-type]

    session_pca = ort.InferenceSession(pca_file_path_named)
    input_name_pca = session_pca.get_inputs()[0].name
    features_pca = session_pca.run(None, {input_name_pca: features_array})[0]

    scale_pos_weight = metadata["scale_pos_weight"]
    print(metadata["scale_pos_weight"])
    scale_pos_weight = float(scale_pos_weight)
    input_name = ort.get_available_providers()[0]
    features_model = features_pca.astype(np.float32)

    session = ort.InferenceSession(model_file_path_named)
    print(f"Model '{model_file_path_named}' loaded.")
    input_name = session.get_inputs()[0].name
    try:
        result: SparseTensor = session.run(None, {input_name: features_model})[0]  # type: ignore
        print(result)
    except (InvalidArgument, ONNXRuntimeError) as error_log:
        match error_log:
            case _ if "Expected: 16384" in str(error_log):
                print(f"Training feature extractor does not match current feature extractor{vae_type.value} : {error_log}")
                predict(image_path, VAEModel.MITSUA_FP16, true_label)
            case _ if "Expected: 65536" in str(error_log):
                print(f"Training feature extractor does not match current feature extractor {vae_type.value} : {error_log}")
                predict(image_path, VAEModel.FLUX2_FP32, true_label)
            case _ if "Expected: 131072" in str(error_log):
                print(f"Training feature extractor does not match current feature extractor {vae_type.value} : {error_log}")
                predict(image_path, VAEModel.GLM_BF16, true_label)

    match true_label:
        case None:
            for prediction in result:  # type: ignore
                if prediction == 0:
                    print("Image is GENUINE")
                else:
                    print("image is SYNTHETIC")
        case _:
            evaluate(result, np.array([true_label]))  # type: ignore

    return result  # type: ignore[return-value]


def training_run(vae_type: VAEModel, file_or_folder_path: Path | None = None) -> None:
    """Train \n
    # XGB00st\n
    model using dataset at path.\n
    :param path: Dataset root folder."""
    from datasets import Dataset

    print("Training selected.")
    dataset: Dataset = build_datasets(file_or_folder_path)
    features_dataset: Dataset = features(dataset, vae_type)
    train_result: TrainResult = grade(features_dataset)
    save_to_onnx(train_result)
    in_console(train_result, vae_type)
    on_graph(train_result)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command passed."""
    import argparse
    from sys import argv

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    train_parser = subparsers.add_parser("train", help="Train model on the dataset in the provided path or `assets/`. The resulting model will be saved to disk.")
    train_parser.add_argument("path", help="Genunie/Human-original dataset path", nargs="?", default=None)
    train_parser.add_argument(
        "-m",
        "--model",
        choices=[m.value for m in VAEModel],
        default=VAEModel.GLM_BF16.value,
        help="Change the VAE model to use for training to a supported HuggingFace repo. Accuracy and memory use decrease from left to right",
    )
    check_parser = subparsers.add_parser(
        "check",
        help="Check whether an image at the provided path is synthetic or original.",
    )
    check_parser.add_argument("path", help="Image or folder path")
    label_grp = check_parser.add_mutually_exclusive_group()
    label_grp.add_argument("-s", "--synthetic", action="store_const", const=1, dest="label", help="Mark image as synthetic (label = 1) for evaluation.")
    label_grp.add_argument("-g", "--genuine", action="store_const", const=0, dest="label", help="Mark image as genuine (label = 0) for evaluation.")

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "train":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None
            datestamped_folder.mkdir(parents=True, exist_ok=True)

            vae_type = VAEModel(args.model)
            training_run(
                vae_type=vae_type,
                file_or_folder_path=dataset_location,
            )
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")
            import json

            results_file_path = model_path / "results.json"
            with open(results_file_path) as result_metadata:
                train_metadata = json.load(result_metadata)
            vae_type = VAEModel(train_metadata["vae_type"])
            predict(Path(args.path), vae_type=vae_type, metadata=train_metadata, true_label=args.label)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
