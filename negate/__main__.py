# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
import numpy as np

from negate import TrainResult, build_datasets, features, generate_dataset, grade, in_console, save_to_onnx, on_graph, VAEModel


def predict(image_path: Path, vae_type: VAEModel = VAEModel.MITSUA_FP16) -> np.ndarray:
    """Predict synthetic or original for given image.\n
    :param image_path: Path to image file.
    :param vae_type: VAE model to use for feature extraction.
    :return: Prediction array.
    """
    from datasets import Dataset
    import onnxruntime as ort

    print("Detection selected.")

    # Check if model files exist
    model_path = Path("models") / "negate.onnx"
    pca_model_path = Path("negate_pca.onnx")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run 'train' first to create the model.")
    if not pca_model_path.exists():
        raise FileNotFoundError(f"PCA model file not found: {pca_model_path}. Please run 'train' first to create the model.")

    # Generate dataset and extract features
    dataset: Dataset = generate_dataset(image_path)
    features_dataset: Dataset = features(dataset, vae_type)

    # Convert to numpy array
    features_array = np.array(features_dataset["features"]).astype(np.float32)  # type: ignore[arg-type]

    # Run PCA transformation
    session_pca = ort.InferenceSession(str(pca_model_path))
    input_name_pca = session_pca.get_inputs()[0].name
    features_pca = session_pca.run(None, {input_name_pca: features_array})[0]

    # Run classifier
    input_name = session_pca.get_inputs()[0].name
    inputs = {input_name: features_pca.astype(np.float32)}  # type: ignore[union-attr]

    session = ort.InferenceSession(str(model_path))
    return session.run(None, inputs)[0]  # type: ignore[return-value]


def training_run(vae_type: VAEModel, file_or_folder_path: Path | None = None) -> None:
    """Train model using dataset at path.\n
    :param path: Dataset root."""
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
    train_parser.add_argument("path", help="Dataset path", nargs="?", default=None)
    train_parser.add_argument(
        "-m",
        "--model",
        choices=[m.value for m in VAEModel],
        default=VAEModel.MITSUA_FP16,
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
            training_run(file_or_folder_path=dataset_location, vae_type=vae_type)
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")
            predict(Path(args.path))
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
