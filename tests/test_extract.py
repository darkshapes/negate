# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
from pathlib import Path

from datasets import Image, load_dataset
from sys import argv

from negate import ResidualExtractor
import pandas as pd


def test_with_dataset(verbose: bool = False) -> None:
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
        fractal, texture = await residual_extractor.process_residuals()
        return (fractal, texture)

    for index in synthetic_images["train"]:
        image_path = Path(index["image"]["path"]).resolve()
        residual_extractor = ResidualExtractor(image_path, output_folder, verbose=verbose)
        fractal, texture = asyncio.run(async_main())
        fractal_synthetic.extend(fractal)
        texture_synthetic.extend(texture)

    residual_extractor = ResidualExtractor(ground_truth_folder, output_folder, verbose=verbose)
    fractal, texture = asyncio.run(async_main())
    fractal_features.extend(fractal)
    texture_features.extend(texture)

    plt.figure(figsize=(8, 5))
    plt.hist(fractal_features, bins=30, alpha=0.7, label="Human")
    plt.hist(fractal_synthetic, bins=30, alpha=0.7, label="Synthetic")
    plt.title("Fractal Complexity Distributions for Human and Synthetic Images")
    plt.xlabel("FD Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(texture_features, bins=30, alpha=0.7, label="Human")
    plt.hist(texture_synthetic, bins=30, alpha=0.7, label="Synthetic")
    plt.title("Texture Complexity Distributions for Human and Synthetic Images")
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
    test_with_dataset(verbose)
