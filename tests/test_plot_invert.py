# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for invert_image function."""

from PIL import Image
import tempfile
from pathlib import Path

from negate.metrics.plot_invert import invert_image


class TestInvertImage:
    """Test suite for invert_image function."""

    def test_invert_image_rgb(self):
        """Test invert_image with RGB image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"

            # Create test image
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(input_path)

            invert_image(str(input_path), str(output_path))

            # Check output
            output_img = Image.open(output_path)
            assert output_img.getpixel((0, 0)) == (0, 255, 255)

    def test_invert_image_grayscale(self):
        """Test invert_image with grayscale image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"

            # Create test image
            img = Image.new("L", (100, 100), color=128)
            img.save(input_path)

            invert_image(str(input_path), str(output_path))

            # Check output - grayscale becomes RGB after inversion
            output_img = Image.open(output_path)
            # Grayscale mode converted to RGB, so pixel is (127, 127, 127)
            assert output_img.getpixel((0, 0)) == (127, 127, 127)
