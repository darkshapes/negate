# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

from datasets import Image, load_dataset
from sys import argv

from negate import ResidualExtractor
import pandas as pd


def with_dataset(verbose: bool = False) -> None:
    """Download the dataset from a remote source and store it in the input folder."""
    import asyncio
    from matplotlib import pyplot as plt

    input_folder = Path(__file__).resolve().parent.parent / "assets"
    output_folder = Path(__file__).resolve().parent.parent / ".output"
    ground_truth_folder = Path(input_folder) / "real"

    synthetic_images = load_dataset("darkshapes/a_slice", cache_dir=input_folder).cast_column("image", Image(decode=False))

    fractal_features = []
    fractal_synthetic = []
    texture_features = []
    texture_synthetic = []

    async def async_main() -> tuple:
        residuals = await residual_extractor.process_residuals()
        return residuals

    for index in synthetic_images["train"]:
        image_path = Path(index["image"]["path"]).resolve()
        residual_extractor = ResidualExtractor(image_path, output_folder, verbose=verbose)
        residuals = asyncio.run(async_main())
        fractal_synthetic.extend(residuals["fractal_complexity"])
        texture_synthetic.extend(residuals["texture_complexity"])

    residual_extractor = ResidualExtractor(ground_truth_folder, output_folder, verbose=verbose)
    residuals = asyncio.run(async_main())
    fractal_features.extend(residuals["fractal_complexity"])
    texture_features.extend(residuals["texture_complexity"])

    plt.figure(figsize=(8, 5))
    plt.hist(fractal_features, bins=30, alpha=0.7, label="Human")
    plt.hist(fractal_synthetic, bins=30, alpha=0.7, label="Synthetic")
    plt.title("Fractal Complexity Distributions for Human and Synthetic Images v1")
    plt.xlabel("FD Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(texture_features, bins=30, alpha=0.7, label="Human")
    plt.hist(texture_synthetic, bins=30, alpha=0.7, label="Synthetic")
    plt.title("Texture Complexity Distributions for Human and Synthetic Images v1")
    plt.xlabel("TC Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def with_dataset_v2(verbose: bool = False) -> None:
    """Download the dataset from a remote source and store it in the input folder."""
    import asyncio
    from matplotlib import pyplot as plt

    input_folder = Path(__file__).resolve().parent.parent / "assets"
    output_folder = Path(__file__).resolve().parent.parent / ".output"
    ground_truth_folder = Path(input_folder) / "real_v2"
    synthetic_folder = Path(input_folder) / "synthetic_v2"

    fractal_features = []
    fractal_synthetic = []
    texture_features = []
    texture_synthetic = []

    async def async_main() -> tuple:
        residuals = await residual_extractor.process_residuals()
        return residuals

    residual_extractor = ResidualExtractor(ground_truth_folder, output_folder, verbose=verbose)
    residuals = asyncio.run(async_main())
    fractal_features.extend(residuals["fractal_complexity"])
    texture_features.extend(residuals["texture_complexity"])

    residual_extractor = ResidualExtractor(synthetic_folder, output_folder, verbose=verbose)
    residuals = asyncio.run(async_main())
    fractal_synthetic.extend(residuals["fractal_complexity"])
    texture_synthetic.extend(residuals["texture_complexity"])

    plt.figure(figsize=(8, 5))
    plt.hist(fractal_features, bins=30, alpha=0.7, label="Human")
    plt.hist(fractal_synthetic, bins=30, alpha=0.7, label="Synthetic")
    plt.title("Fractal Complexity Distributions for Human and Synthetic Images v2")
    plt.xlabel("FD Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(texture_features, bins=30, alpha=0.7, label="Human")
    plt.hist(texture_synthetic, bins=30, alpha=0.7, label="Synthetic")
    plt.title("Texture Complexity Distributions for Human and Synthetic Images v2")
    plt.xlabel("TC Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    verbose = False
    if len(argv) > 1 and "-v" == argv[1]:
        verbose = True

    # pytest.main([__file__])
    with_dataset(verbose)
    with_dataset_v2(verbose)
