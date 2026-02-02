# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
import numpy as np

from negate import TrainResult, build_datasets, features, generate_dataset, grade, in_console, save_to_onnx, on_graph, VAEModel


def evaluate(prediction: np.ndarray, ground_truth: np.ndarray) -> None:
    """Print accuracy and class distribution.\n
    :param prediction: Model outputs (0 = genuine, 1 = synthetic).\n
    :param ground_truth: Ground-truth labels.\n
    :return: None."""

    prediction = prediction.astype(int)
    ground_truth = ground_truth.astype(int)

    acc = float(np.mean(prediction == ground_truth))

    genu_cnt = int(np.sum(ground_truth == 0))
    synth_cnt = int(np.sum(ground_truth == 1))

    print(f"Accuracy: {acc:.2%}")
    print(f"Genuine: {genu_cnt}  Synthetic: {synth_cnt}")


def predict(image_path: Path, vae_type: VAEModel = VAEModel.MITSUA_FP16, true_label: int | None = None) -> np.ndarray:
    """Predict synthetic or original for given image. (0 = genuine, 1 = synthetic)\n
    :param image_path: Path to image file or folder.
    :param vae_type: VAE model to use for feature extraction.
    :return: Prediction array.
    """
    from datasets import Dataset
    import onnxruntime as ort
    from onnxruntime import SparseTensor

    print(f"{'Evaluation' if true_label is not None else 'Detection'} selected.")

    models_location = Path(__file__).parent.parent / "models"
    model_file = models_location / "negate.onnx"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}. Please run 'train' first to create the model.")

    dataset: Dataset = generate_dataset(image_path)
    features_dataset: Dataset = features(dataset, vae_type)

    features_array = np.array(features_dataset["features"]).astype(np.float32)  # type: ignore[arg-type]

    session = ort.InferenceSession(str(model_file))
    input_name = session.get_inputs()[0].name
    result: SparseTensor = session.run(None, {input_name: features_array})[0]  # type: ignore
    print(result)
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
            vae_type = VAEModel(args.model)
            training_run(file_or_folder_path=dataset_location, vae_type=vae_type)
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")
            predict(Path(args.path), true_label=args.label)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
