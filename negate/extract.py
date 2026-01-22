# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
"""
ResidualExtractor for Laplacian Residuals and Texture Analysis

This module provides a class to extract Laplacian residuals from images and compute
fractal and texture features for further analysis.

Classes:
    ResidualExtractor:
        Extracts Laplacian residuals and computes fractal and texture features.
        - Attributes:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to save extracted residuals.
            verbose (bool): If True, enables verbose output.
        - Methods:
            download_dataset(): Downloads the dataset.
            laplacian_residual(): Computes Laplacian residuals.
            box_count(): Calculates fractal dimension using box-counting.
            texture_complexity(): Computes texture complexity.
            extract_and_save_residuals(): Extracts and saves residuals.
            process_residuals(): Processes saved residuals to compute features.

Functions:
    main():
        Main entry point to run the extraction and analysis process.
"""

import asyncio
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class ResidualExtractor:
    """Extracts Laplacian residuals and computes fractal and texture features from images.\n

    Attributes:
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save the extracted residuals.

    Methods:
        download_dataset(): Downloads the dataset for processing.
        laplacian_residual(): Computes Laplacian residuals.
        box_count(): Calculates the fractal dimension using the box-counting method.
        texture_complexity(): Computes texture complexity from the residual image.
        extract_and_save_residuals(): Extracts residuals and saves them to disk.
        process_residuals(): Processes saved residuals to compute features."""

    def __init__(self, input: Path, output_folder: Path, verbose: bool = False) -> None:
        """Initializes the ResidualExtractor with input and output folders.\n
        :param input_folder: Path to the folder containing images.
        :param output_folder: Path to the folder for saving residuals.
        :param verbose: If True, enables verbose output. Defaults to False."""
        self.sep = " : "
        self.input: Path = input
        self.output_folder = output_folder
        self.verbose = verbose
        self.console(("input", self.input), ("output", self.output_folder))
        self.console(("type input", type(self.input)))
        self.console(("input is dir", self.input.is_dir()))
        self.console(("input is file", self.input.is_file()))

    async def _load_single_image(self, image_path: str) -> np.ndarray | None:
        """Load a single image in grayscale."""
        return await asyncio.get_running_loop().run_in_executor(None, cv2.imread, image_path, cv2.IMREAD_GRAYSCALE)

    async def _load_images(self) -> list[np.ndarray]:
        """Asynchronously loads images from the provided paths."""
        loop = asyncio.get_running_loop()
        images: list[np.ndarray] = []

        image_extensions = {".jpg", ".png", ".jpeg"}
        if Path(self.input).is_dir():
            self.image_paths = [str(file_path) for file_path in Path(self.input).iterdir() if file_path.suffix.lower() in image_extensions]
        elif Path(self.input).is_file():
            self.image_paths = [self.input]

        self.console(("extension", Path(self.input).suffix))

        # Choose the iterator based on verbosity
        iterator = tqdm(self.image_paths) if self.verbose else self.image_paths

        for image_path in iterator:
            self.console(("loading", image_path))
            image = await self._load_single_image(image_path)
            if image is not None:
                images.append(image)
            else:
                self.console(("Warning: Image not loaded from", image_path))

        return images

    def laplacian_residual(self) -> NDArray:
        """Computes Laplacian residuals using a 3x3 kernel."""

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return cv2.filter2D(self.image.astype(np.float32), -1, kernel)

    def box_count(self, Z, threshold=0.01) -> NDArray:
        """Calculates the fractal dimension using the box-counting method."""

        Z = np.abs(Z)
        sizes = np.logspace(2, 7, num=6, base=2, dtype=int)  # box sizes
        counts = []
        for size in sizes:
            S = cv2.resize(Z, (size, size), interpolation=cv2.INTER_NEAREST)
            count = np.sum(S > threshold)
            try:
                counts.append(count)
            except RuntimeWarning as _:
                if self.verbose:
                    self.console(("Divide by zero error for", self.image_path), ("/n", ".. continuing"))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]  # FD

    def texture_complexity(self, R, D=16) -> NDArray:
        """Computes texture complexity from the residual image."""

        R = np.abs(R)
        h, w = R.shape
        blocks = [R[i : i + D, j : j + D] for i in range(0, h - D + 1, D) for j in range(0, w - D + 1, D)]
        TCs = []
        for block in blocks:
            t = 1 - np.sum((2 ** (-block)) / (block + 1e-5)) / (D * D)
            # Clamp t value to avoid log(0) or negative
            t = np.clip(t, 1e-5, 1 - 1e-5)
            tc = np.log(t / (1 - t)) + 4
            TCs.append(tc)
        return np.array(TCs)

    async def process_residuals(self) -> tuple[list[NDArray], list[NDArray]]:
        """Asynchronously processes images to compute fractal and texture features.\n
        :returns: Tuple of lists containing fractal dimensions and texture
        complexity values for each input image."""
        self.console(("process", "running..."))
        self.fractal_features = []
        self.texture_features = []

        self.images = await self._load_images()

        async def process_single(idx: int) -> None:
            self.image = self.images[idx]
            residual = await asyncio.get_running_loop().run_in_executor(None, self.laplacian_residual)
            fractal_dimension = await asyncio.get_running_loop().run_in_executor(None, self.box_count, residual)
            self.fractal_features.append(fractal_dimension)
            tc = await asyncio.get_running_loop().run_in_executor(None, self.texture_complexity, residual)
            self.texture_features.append(tc.mean())

        await asyncio.gather(*[process_single(self.image_path) for self.image_path in range(len(self.image_paths))])

        return self.fractal_features, self.texture_features

    def console(self, *args: tuple[str, Any]) -> None:
        if self.verbose:
            for arg, pair in args:
                print(f"{arg}", self.sep, f"{pair}")


def main() -> None:
    """Main entry point to run the extraction and analysis process."""

    import argparse

    from matplotlib import pyplot as plt

    input_folder = Path(__file__).resolve().parent.parent / "assets"
    output_folder = Path(__file__).resolve().parent.parent / ".output"

    parser = argparse.ArgumentParser(description="Extract Laplacian residuals from images.")
    parser.add_argument("-i", "--input", type=str, default=input_folder, help="Input folder containing images or individual image.")
    parser.add_argument("-o", "--output", type=str, default=output_folder, help="Output folder for residuals.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    verbose = args.verbose
    residual_extractor = ResidualExtractor(input_folder, output_folder, verbose)

    async def async_main() -> None:
        fractal_features, texture_features = await residual_extractor.process_residuals()
        plt.figure(figsize=(8, 5))
        plt.hist(fractal_features, bins=30, alpha=0.7, label="Fractal Dimension")
        plt.hist(texture_features, bins=30, alpha=0.7, label="Complexity")
        plt.title("Fractal and Texture Complexity Distributions for Synthetic Images")
        plt.xlabel("FD + TC Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

    asyncio.run(async_main())
