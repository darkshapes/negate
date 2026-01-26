# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

from datasets import Image, load_dataset
from sys import argv

from negate import ResidualExtractor
from negate.quantify import graph_comparison


def with_dataset(ground_truth_folder: str, synthetic_folder: str | None = None, verbose: bool = False) -> None:
    """Download the dataset from a remote source and store it in the input folder."""
    import asyncio

    input_folder = Path(__file__).resolve().parent.parent / "assets"
    output_folder = Path(__file__).resolve().parent.parent / ".output"

    async def async_main() -> tuple:
        residuals = await residual_extractor.process_residuals()
        return residuals

    if not synthetic_folder:
        synthetic_images = load_dataset("darkshapes/a_slice", cache_dir=str(input_folder)).cast_column("image", Image(decode=False))
        residual_extractor = ResidualExtractor(Path(synthetic_images["train"][0]["image"]["path"]), output_folder, verbose=verbose)
        for index in synthetic_images["train"]:
            image_path = Path(index["image"]["path"]).resolve()
            residual_extractor.input = image_path
            synthetic_residuals = asyncio.run(async_main())

    else:
        input_folder = Path(input_folder) / synthetic_folder
        residual_extractor = ResidualExtractor(synthetic_folder, output_folder, verbose=verbose)
        synthetic_residuals = asyncio.run(async_main())

    synthetic_residuals = residual_extractor.data_frame

    input_folder = Path(input_folder) / ground_truth_folder
    residual_extractor = ResidualExtractor(input_folder, output_folder, verbose=verbose)
    human_residuals = asyncio.run(async_main())

    human_residuals = residual_extractor.data_frame

    graph_comparison(synthetic_residuals, human_residuals)


if __name__ == "__main__":
    verbose = False
    if len(argv) > 1 and "-v" == argv[1]:
        verbose = True

    ground_truth_folder = "real"

    with_dataset(ground_truth_folder=ground_truth_folder, verbose=verbose)

    ground_truth_folder = "real_v2"
    synthetic_folder = "synthetic_v2"

    with_dataset(ground_truth_folder, synthetic_folder, verbose)
