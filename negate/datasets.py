# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

import numpy as np
from datasets import Dataset, Image, concatenate_datasets, load_dataset


def detect_nans(dataset: Dataset) -> Dataset:
    """Detect and remove NaN labels.\n
    :param dataset: Dataset with a ``label`` column.
    :return: Dataset without rows containing NaN labels."""

    import numpy as np

    lbls = np.array(dataset["label"])
    if np.isnan(lbls).any():
        print("NaNs at indices:", np.where(np.isnan(lbls)))
        valid = ~np.isnan(lbls)
        dataset = dataset.select(valid)
    return dataset


def load_remote_dataset(repo: str, folder_path: Path) -> Dataset:
    """Load a remote dataset and attach a default label.\n
    :param repo: Repository ID of the dataset.
    :param folder_path: Local path to cache the dataset.
    :return: Dataset with a ``label`` column added and NaNs removed."""

    remote_dataset = load_dataset(repo, cache_dir=str(folder_path), split="train").cast_column("image", Image(decode=True))
    remote_dataset = remote_dataset.add_column("label", [1] * len(remote_dataset))
    remote_dataset = detect_nans(remote_dataset)
    return remote_dataset


def generate_dataset(input_path: Path) -> Dataset:
    """Generates a dataset from images in the given folder.\n
    :param input_path: Path to the folder containing image files.
    :return: Dataset containing images and labels with NaNs removed."""

    from PIL import Image as PillowImage

    validated_paths = []
    valid_extensions = {".jpg", ".webp", ".jpeg", ".png", ".tif", ".tiff"}
    for img_path in input_path.iterdir():
        if not (img_path.is_file() and img_path.suffix.lower() in valid_extensions):
            continue
        try:
            with PillowImage.open(img_path) as _verification:
                pass
        except Exception as _unreadable_file:
            continue
        validated_paths.append({"image": str(img_path)})

    dataset = Dataset.from_list(validated_paths)  # NaN Prevention: decode separately

    try:  # Fallback: keep the raw bytes if decoding fails.
        dataset = dataset.cast_column("image", Image(decode=True))
    except Exception:
        dataset = dataset.cast_column("image", Image())

    dataset = dataset.add_column("label", [0] * len(validated_paths))
    dataset = detect_nans(dataset)
    return dataset


def build_datasets(input_folder: Path | None = None) -> Dataset:
    """Builds synthetic and original datasets.\n
    :param input_folder: Path to folder containing data. (optional)
    :return: Dataset containing synthetic and original images."""

    synthetic_input_folder = Path(".datasets")
    original_input_folder = Path(__file__).parent.parent / "assets" / "ph"

    slice_dataset = load_remote_dataset("exdysa/nano-banana-pro-generated-1k-clone", synthetic_input_folder)
    rnd_synthetic_dataset = load_remote_dataset("exdysa/rnd_synthetic_img", synthetic_input_folder)

    original_dataset = generate_dataset(original_input_folder)

    dataset = concatenate_datasets([slice_dataset, rnd_synthetic_dataset, original_dataset])
    return dataset


def dataset_to_nparray(dataset: Dataset, column_names: list[str] | None = None) -> np.ndarray:
    """Convert Dataset to ndarray.\n
    :param dataset: HuggingFace Dataset of images.
    :param columns: Columns to keep. If None all columns are used.
    :return: Array of shape (n_samples, n_features) or (n_samples,) if a single column."""

    if column_names is None:
        column_names = dataset.column_names

    data = {name: dataset[name] for name in column_names}

    if len(column_names) == 1:
        return np.array(data[column_names[0]])
    return np.vstack([np.array(data[name]) for name in column_names]).T
