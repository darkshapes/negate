# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

import numpy as np
from datasets import Dataset, Image, concatenate_datasets, load_dataset
from PIL import Image as PillowImage

from negate import negate_opt


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


def load_remote_dataset(repo: str, folder_path: Path, label: int, split="train") -> Dataset:
    """Load a remote dataset and attach a default label.\n
    :param repo: Repository ID of the dataset.
    :param folder_path: Local path to cache the dataset.
    :param label: The default label to assign to all images in the dataset
    :return: Dataset with a ``label`` column added and NaNs removed."""

    print(f"Using remote images from {repo}")
    remote_dataset = load_dataset(repo, cache_dir=str(folder_path), split=split).cast_column("image", Image(decode=True))
    remote_dataset = remote_dataset.add_column("label", [label] * len(remote_dataset))
    remote_dataset = detect_nans(remote_dataset)
    return remote_dataset


def generate_dataset(input_path: Path | list[dict[str, PillowImage.Image]], label: int | None = None) -> Dataset:
    """Generates a dataset from images in the given folder.\n
    :param input_path: Path to the folder containing image files.
    :return: Dataset containing images and labels with NaNs removed."""

    if isinstance(input_path, Path):
        print(f"Using local images from {input_path}")

        validated_paths = []

        valid_extensions = {".jpg", ".webp", ".jpeg", ".png", ".tif", ".tiff"}
        assert isinstance(input_path, Path)
        if input_path.is_dir():
            for img_path in input_path.iterdir():
                if not (img_path.is_file() and img_path.suffix.lower() in valid_extensions):
                    continue
                try:
                    with PillowImage.open(img_path) as _verification:
                        pass
                except Exception as _unreadable_file:
                    continue
                validated_paths.append({"image": str(img_path)})
        elif input_path.is_file() and input_path.suffix.lower() in valid_extensions:
            validated_paths.append({"image": str(input_path)})
        else:
            raise ValueError(f"Invalid path {input_path}")
    else:
        validated_paths = input_path
    dataset = Dataset.from_list(validated_paths)  # NaN Prevention: decode separately

    try:  # Fallback: keep the raw bytes if decoding fails.
        dataset = dataset.cast_column("image", Image(decode=True))
    except Exception:
        dataset = dataset.cast_column("image", Image())

    if label is not None:
        dataset = dataset.add_column("label", [label] * len(validated_paths))
        dataset = detect_nans(dataset)
    return dataset


def build_datasets(input_folder: Path | None = None) -> Dataset:
    """Builds synthetic and genuine datasets.\n
    :param input_folder: Path to folder containing data. (optional)
    :return: Dataset containing synthetic and genuine images."""

    synthetic_input_folder = Path(".datasets")
    synthetic_input_folder.mkdir(parents=True, exist_ok=True)
    synthetic_repos = []
    if negate_opt.synthetic_data is not None:
        for data_repo in negate_opt.synthetic_data:
            synthetic_repos.append(load_remote_dataset(data_repo, synthetic_input_folder, label=1))
    if negate_opt.synthetic_local is not None:
        for data_folder in negate_opt.synthetic_local:
            synthetic_repos.append(generate_dataset(Path(data_folder), label=1))

    genuine_input_folder = Path("assets")
    genuine_input_folder.mkdir(parents=True, exist_ok=True)
    genuine_repos = []
    if negate_opt.genuine_data is not None:
        for data_repo in negate_opt.genuine_data:
            genuine_repos.append(load_remote_dataset(data_repo, genuine_input_folder, label=0))
    if negate_opt.genuine_local is not None:
        for data_folder in negate_opt.genuine_local:
            genuine_repos.append(generate_dataset(Path(data_folder), label=0))

    dataset = concatenate_datasets([*genuine_repos, *synthetic_repos])
    return dataset
