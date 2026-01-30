# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import argparse
import pickle
from pathlib import Path
from sys import argv

import numpy as np
import xgboost as xgb
from datasets import Dataset

from negate import TrainResult, build_datasets, dataset_to_nparray, features, generate_dataset, grade, in_console, save_model, save_to_onnx


def predict(image_path: Path) -> np.ndarray:
    """Predict synthetic or original for given image.\n
    :param image_path: Path to image file.
    :return: Prediction array.
    """
    import numpy as np

    print("Detection selected.")

    dataset: Dataset = generate_dataset(image_path)
    dataset_np: np.ndarray = dataset_to_nparray(dataset)
    features_dataset: Dataset = features(dataset_np)
    model = xgb.Booster()
    model.load_model("model.xgb")

    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    meta = np.load("meta.npz")
    scale_pos_weight = meta["scale_pos_weight"]

    features_pca = pca.transform(features_dataset)
    dmat = xgb.DMatrix(features_pca)
    return model.predict(dmat)


def training_run(file_or_folder_path: Path | None = None) -> None:
    """Train model using dataset at path.\n
    :param path: Dataset root.
    :return: None."""
    print("Training selected.")
    dataset: Dataset = build_datasets(file_or_folder_path)
    features_dataset: Dataset = features(dataset)
    train_result: TrainResult = grade(features_dataset)
    save_model(train_result)
    save_to_onnx(train_result)
    in_console(train_result)


def main() -> None:
    """CLI entry point.\n
    :raises ValueError: Missing image path.
    :raises NotImplementedError: Unsupported command.
    :return: None."""
    parser = argparse.ArgumentParser(description="Negate CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    train_parser = subparsers.add_parser("train", help="Train model. the model will be trained on the dataset and the resulting model will be saved to disk.")
    train_parser.add_argument("path", help="Dataset path", nargs="?", default=None)

    check_parser = subparsers.add_parser(
        "check",
        help="Check whether an image is synthetic or original. - The command will read the image file at the provided path and output a prediction of whether the image is synthetic or not.",
    )
    check_parser.add_argument("path", help="Image path")

    args = parser.parse_args(argv[1:])

    match args.cmd:
        case "train":
            if args.path:
                dataset_location: Path | None = Path(args.path)
            else:
                dataset_location: Path | None = None

            training_run(dataset_location)
        case "check":
            if args.path is None:
                raise ValueError("Check requires an image path.")
            predict(Path(args.path))
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    main()
