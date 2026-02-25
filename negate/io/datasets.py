# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from pathlib import Path

import numpy as np
from datasets import Dataset, Image, concatenate_datasets, load_dataset
from PIL import Image as PillowImage
from tqdm import tqdm

from negate.io.config import Spec, root_folder


def detect_nans(dataset: Dataset) -> Dataset:
    """Detect and remove NaN labels.\n
    :param dataset: Dataset with a ``label`` column.
    :return: Dataset without rows containing NaN labels."""

    lbls = np.array(dataset["label"])
    if np.isnan(lbls).any():
        print("NaNs at indices:", np.where(np.isnan(lbls)))
        valid = ~np.isnan(lbls)
        dataset = dataset.select(valid)
    return dataset


def load_remote_dataset(repo: str, folder_path: Path, split="train", label: int | None = None) -> Dataset:
    """Load a remote dataset and attach a default label.\n
    :param repo: Repository ID of the dataset.
    :param folder_path: Local path to cache the dataset.
    :param label: The default label to assign to all images in the dataset
    :return: Dataset with a ``label`` column added and NaNs removed."""

    remote_dataset = load_dataset(repo, cache_dir=str(folder_path), split=split).cast_column("image", Image(decode=True, mode="RGB"))
    if label is not None:
        remote_dataset = remote_dataset.add_column("label", [label] * len(remote_dataset))
    remote_dataset = detect_nans(remote_dataset)
    return remote_dataset


def generate_dataset(file_or_folder_path: Path | list[dict[str, PillowImage.Image]], label: int | None = None) -> Dataset:
    """Generates a dataset from an image file or folder of images.\n
    :param folder_path: Path to the folder containing image files.
    :return: Dataset containing images and labels with NaNs removed."""

    if isinstance(file_or_folder_path, Path):
        validated_paths = []
        valid_extensions = {".jpg", ".webp", ".jpeg", ".png", ".tif", ".tiff"}
        assert isinstance(file_or_folder_path, Path)

        if file_or_folder_path.is_dir():
            for img_path in tqdm(file_or_folder_path.iterdir(), total=len(os.listdir(str(file_or_folder_path))), desc="Creating dataset..."):
                if not (img_path.is_file() and img_path.suffix.lower() in valid_extensions):
                    continue
                try:
                    with PillowImage.open(img_path) as _verification:
                        pass
                except Exception as _unreadable_file:
                    continue
                validated_paths.append({"image": str(img_path)})
        elif file_or_folder_path.is_file() and file_or_folder_path.suffix.lower() in valid_extensions:
            validated_paths.append({"image": str(file_or_folder_path)})
        else:
            raise ValueError(f"Invalid path {file_or_folder_path}")
    else:
        validated_paths = file_or_folder_path
    dataset = Dataset.from_list(validated_paths)  # NaN Prevention: decode separately

    try:  # Fallback: keep the raw bytes if decoding fails.
        dataset = dataset.cast_column("image", Image(decode=True, mode="RGB"))
    except Exception:
        dataset = dataset.cast_column("image", Image())

    if label is not None:
        dataset = dataset.add_column("label", [label] * len(validated_paths))
        dataset = detect_nans(dataset)
    return dataset


def build_datasets(
    spec: Spec,
    genuine_path: Path | None = None,
    synthetic_path: Path | None = None,
) -> Dataset:
    """Builds synthetic and genuine datasets.\n
    :param input_folder: Path to folder containing data. (optional)
    :return: Dataset containing synthetic and genuine images."""

    synthetic_input_folder = root_folder / ".datasets"
    synthetic_input_folder.mkdir(parents=True, exist_ok=True)
    synthetic_repos = []

    spec.data.synthetic_local.append(str(synthetic_path))
    spec.data.genuine_local.append(str(genuine_path))
    print(f"Using images from {spec.data}")

    if spec.data.synthetic_data is not None:
        for data_repo in spec.data.synthetic_data:
            if data_repo is not None:
                synthetic_repos.append(load_remote_dataset(data_repo, synthetic_input_folder, label=1))
    if spec.data.synthetic_local is not None:
        for data_folder in spec.data.synthetic_local:
            if data_folder is not None:
                synthetic_repos.append(generate_dataset(Path(data_folder), label=1))

    genuine_input_folder = root_folder / "assets"
    genuine_input_folder.mkdir(parents=True, exist_ok=True)
    genuine_repos = []
    if spec.data.genuine_data is not None:
        for data_repo in spec.data.genuine_data:
            if data_repo is not None:
                genuine_repos.append(load_remote_dataset(data_repo, genuine_input_folder, label=0))
    if spec.data.genuine_local is not None:
        for data_folder in spec.data.genuine_local:
            if data_folder is not None:
                genuine_repos.append(generate_dataset(Path(data_folder), label=0))

    dataset = concatenate_datasets([*genuine_repos, *synthetic_repos])
    return dataset
