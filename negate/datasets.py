# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

from datasets import Dataset, Image, concatenate_datasets, interleave_datasets, load_dataset


def build_datasets() -> Dataset:
    """Builds synthetic and original datasets.\n
    :returns: A dictionary containing synthetic and original datasets."""

    synthetic_input_folder = ".datasets"
    original_input_folder = Path(__file__).parent.parent / "assets" / "ph"

    slice_dataset = load_dataset("darkshapes/a_slice", cache_dir=str(synthetic_input_folder), split="train").cast_column("image", Image(decode=True))
    rnd_synthetic_dataset = load_dataset("exdysa/rnd_synthetic_img", cache_dir=str(synthetic_input_folder), split="train").cast_column("image", Image(decode=True))
    synthetic_dataset = interleave_datasets([slice_dataset, rnd_synthetic_dataset])

    slice_dataset = slice_dataset.add_column("label", [1] * len(slice_dataset))
    rnd_synthetic_dataset = rnd_synthetic_dataset.add_column("label", [1] * len(rnd_synthetic_dataset))

    synthetic_dataset = concatenate_datasets([slice_dataset, rnd_synthetic_dataset])

    original_folder_contents = [{"image": str(image)} for image in original_input_folder.iterdir() if image.is_file()]
    original_dataset = Dataset.from_list(original_folder_contents).cast_column("image", Image(decode=True))
    original_dataset = original_dataset.add_column("label", [0] * len(original_dataset))

    dataset = concatenate_datasets([synthetic_dataset, original_dataset])
    return dataset
