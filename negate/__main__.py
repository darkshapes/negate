# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
from sys import argv
import numpy as np

from negate import TrainResult, build_datasets, dataset_to_nparray, features, generate_dataset, grade, in_console, save_to_onnx, on_graph, VAEModel


def predict(image_path: Path) -> np.ndarray:
    """Predict synthetic or original for given image.\n
    :param image_path: Path to image file.
    :return: Prediction array.
    """
    from datasets import Dataset
    import onnxruntime as ort

    print("Detection selected.")

    dataset: Dataset = generate_dataset(image_path)
    dataset_np: np.ndarray = dataset_to_nparray(dataset)
    features_dataset: Dataset = features(dataset_np)

    session_pca = ort.InferenceSession("negate_pca.onnx")
    input_name_pca = session_pca.get_inputs()[0].name
    features_pca = session_pca.run(None, {input_name_pca: np.array(features_dataset).astype(np.float32)})[0]

    input_name = ort.get_available_providers()[0]
    inputs = {input_name: features_pca.astype(np.float32)}

    session = ort.InferenceSession("negate.onnx")
    return session.run(None, inputs)[0]


def training_run(file_or_folder_path: Path | None = None, vae_type: VAEModel = VAEModel.FLUX2_FP32) -> None:
    """Train model using dataset at path.\n
    :param path: Dataset root."""
    from datasets import Dataset

    print("Training selected.")
    dataset: Dataset = build_datasets(file_or_folder_path)
    features_dataset: Dataset = features(dataset, vae_type)
    train_result: TrainResult = grade(features_dataset)
    save_to_onnx(train_result)
    in_console(train_result)
    on_graph(train_result)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command."""
    import argparse

    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    train_parser = subparsers.add_parser("train", help="Train model on the dataset in the provided path or `assets/`. The resulting model will be saved to disk.")
    train_parser.add_argument("path", help="Dataset path", nargs="?", default=None)
    train_parser.add_argument(
        "--model",
        choices=[m.value for m in VAEModel],
        default=VAEModel.FLUX2_FP32.value,
        help="Change the VAE model to use for training to a supported HuggingFace repo. Accuracy and memory use decrease from left to right",
    )
    check_parser = subparsers.add_parser(
        "check",
        help="Check whether an image at the provided path is synthetic or original.",
    )
    check_parser.add_argument("path", help="Image path")

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "train":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None
            vae_type = VAEModel(args.model)
            training_run(dataset_location, vae_type)
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")
            predict(Path(args.path))
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
