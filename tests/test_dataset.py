# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for generate_dataset function."""

from pathlib import Path
from PIL import Image as PillImage
import tempfile
import os
import pytest
from negate.io.datasets import generate_dataset


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = PillImage.new("RGB", size, color="red")
    img.save(path)


class TestGenerateDataset:
    """Test suite for generate_dataset function."""

    def test_directory_with_images(self):
        """Test generating dataset from directory containing images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            _create_test_image(image_dir / "img1.png")
            _create_test_image(image_dir / "img2.jpg")
            _create_test_image(image_dir / "img3.webp")

            dataset = generate_dataset(image_dir)

            assert len(dataset) == 3
            assert "image" in dataset.column_names

    def test_single_file_path(self):
        """Test generating dataset from single image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.webp"
            _create_test_image(img_path)

            dataset = generate_dataset(img_path)

            assert len(dataset) == 1
            assert "image" in dataset.column_names

    def test_invalid_directory_raises_error(self):
        """Test that invalid directory raises ValueError."""
        with pytest.raises(ValueError, match="unknown file extension: .txt"):
            with tempfile.TemporaryDirectory() as tmpdir:
                invalid_dir = Path(tmpdir) / "invalid"
                os.makedirs(invalid_dir, exist_ok=True)

                _create_test_image(Path(tmpdir) / "not_an_image.txt")

                dataset = generate_dataset(invalid_dir)

                assert len(dataset) == 0

    def test_list_of_dicts_input(self):
        """Test generating dataset from list of dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img1_path = Path(tmpdir) / "img1.png"
            img2_path = Path(tmpdir) / "img2.jpg"
            _create_test_image(img1_path)
            _create_test_image(img2_path)

            input_data = [
                {"image": str(img1_path)},
                {"image": str(img2_path)},
            ]

            dataset = generate_dataset(input_data)

            assert len(dataset) == 2

    def test_label_parameter(self):
        """Test that label parameter adds label column correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            for i in range(3):
                _create_test_image(image_dir / f"img{i}.png")

            dataset = generate_dataset(image_dir, label=42)

            assert len(dataset) == 3
            assert all(row["label"] == 42 for row in dataset)

    def test_skips_non_image_files(self):
        """Test that non-image files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            _create_test_image(image_dir / "valid.png")
            (image_dir / "readme.txt").write_text("text file")
            (image_dir / "script.py").write_text("# python")

            dataset = generate_dataset(image_dir)

            assert len(dataset) == 1

    def test_valid_extensions(self):
        """Test all supported image extensions are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions = [".jpg", ".webp", ".jpeg", ".png", ".tif", ".tiff"]

            for ext in extensions:
                img_path = Path(tmpdir) / f"img{ext}"
                _create_test_image(img_path, size=(10, 10))

                dataset = generate_dataset(img_path)
                assert len(dataset) == 1, f"Failed for extension {ext}"

    def test_uppercase_extensions(self):
        """Test that uppercase extensions are also accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.PNG"
            _create_test_image(img_path)

            dataset = generate_dataset(img_path)

            assert len(dataset) == 1
