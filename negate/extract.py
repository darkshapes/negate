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

import numpy as np
import cv2
from tqdm import tqdm


class ResidualExtractor:
    """
        Extracts Laplacian residuals and computes fractal and texture features from images.\n
    Attributes:
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save the extracted residuals.

    Methods:
        download_dataset(): Downloads the dataset for processing.
        laplacian_residual(): Computes Laplacian residuals.
        box_count(): Calculates the fractal dimension using the box-counting method.
        texture_complexity(): Computes texture complexity from the residual image.
        extract_and_save_residuals(): Extracts residuals and saves them to disk.
        process_residuals(): Processes saved residuals to compute features.
    """

    def __init__(self, input_folder, output_folder, verbose=False) -> None:
        """Initializes the ResidualExtractor with input and output folders.\n
        :param input_folder: Path to the folder containing images.
        :param output_folder: Path to the folder for saving residuals.
        :param verbose: If True, enables verbose output. Defaults to False."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.verbose = verbose

    def download_dataset(self) -> None:
        """Download the dataset from a remote source and store it in the input folder."""
        from datasets import load_dataset, Image, DownloadMode

        self.dataset = load_dataset("darkshapes/a_slice", cache_dir=self.input_folder, download_mode=DownloadMode.FORCE_REDOWNLOAD).cast_column("image", Image(decode=False))

    def laplacian_residual(self):
        """Computes Laplacian residuals using a 3x3 kernel."""
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return cv2.filter2D(self.image.astype(np.float32), -1, kernel)

    def box_count(self, Z, threshold=0.01):
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
                    print(f"Divide by zero error for {self.residual_path}.. continuing")
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]  # FD

    def texture_complexity(self, R, D=16):
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

    def extract_and_save_residuals(self):
        """Extracts Laplacian residuals and saves them to disk."""
        import os
        from pathlib import Path

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        self.residuals = []
        for index in tqdm(self.dataset["train"]):
            file_path_named = Path(index["image"]["path"]).resolve()
            out_file = os.path.basename(file_path_named) + ".npy"
            residual_path = os.path.join(self.output_folder, out_file)
            if not os.path.exists(residual_path):
                self.image = cv2.imread(file_path_named, cv2.IMREAD_GRAYSCALE)
                residual = self.laplacian_residual()
                np.save(residual_path, residual)
            self.residuals.append(residual_path)

    def process_residuals(self):
        """Processes saved residuals to compute fractal and texture features."""
        self.fractal_features = []
        self.texture_features = []
        for self.residual_path in tqdm(self.residuals, desc="Processing residuals..."):
            residual_in = np.load(self.residual_path)
            fractal_dimension = self.box_count(residual_in)
            self.fractal_features.append(fractal_dimension)
            tc = self.texture_complexity(residual_in)
            self.texture_features.append(tc.mean())


def main():
    """Main entry point to run the extraction and analysis process."""

    import os
    import argparse
    from matplotlib import pyplot as plt

    input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".output/")

    parser = argparse.ArgumentParser(description="Extract Laplacian residuals from images.")
    parser.add_argument("-i", "--input", type=str, default=input_folder, help="Input folder containing images.")
    parser.add_argument("-o", "--output", type=str, default=output_folder, help="Output folder for residuals.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    residual_extractor = ResidualExtractor(args.input, args.output)
    residual_extractor.download_dataset()
    residual_extractor.extract_and_save_residuals()
    residual_extractor.process_residuals()

    plt.figure(figsize=(8, 5))
    plt.hist(residual_extractor.fractal_features, bins=30, alpha=0.7, label="Fake")
    plt.hist(residual_extractor.texture_features, bins=30, alpha=0.7, label="Fake")
    plt.title("Fractal and Texture Complexity Distributions")
    plt.xlabel("FD + TC Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
