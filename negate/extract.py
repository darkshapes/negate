# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import numpy as np
import cv2


class ResidualExtractor:
    def __init__(self, input_folder, output_folder, verbose=False) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder

    def download_dataset(self) -> None:
        """Download the dataset from a remote source and store it in the input folder."""
        from datasets import load_dataset, Image, DownloadMode

        self.dataset = load_dataset("darkshapes/a_slice", cache_dir=self.input_folder, download_mode=DownloadMode.FORCE_REDOWNLOAD).cast_column("image", Image(decode=False))

    def laplacian_residual(self):
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return cv2.filter2D(self.image.astype(np.float32), -1, kernel)

    def extract_and_save_residuals(self):
        from tqdm import tqdm
        import os
        from pathlib import Path

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        for index in tqdm(self.dataset["train"]):
            file_path_named = Path(index["image"]["path"]).resolve()
            self.image = cv2.imread(file_path_named, cv2.IMREAD_GRAYSCALE)
            residual = self.laplacian_residual()
            out_file = os.path.basename(file_path_named) + ".npy"
            residual_path = os.path.join(self.output_folder, out_file)
            np.save(residual_path, residual)


def main():
    import os
    import argparse

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
