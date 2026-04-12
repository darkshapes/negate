# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Invert image colors for negative generation."""

from PIL import Image


def invert_image(input_path: str, output_path: str) -> None:
    """Invert colors of a PNG image (create negative).

    :param input_path: Path to source PNG.
    :param output_path: Path for inverted output.
    """

    img = Image.open(input_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    r, g, b = img.split()
    r = Image.eval(r, lambda x: 255 - x)
    g = Image.eval(g, lambda x: 255 - x)
    b = Image.eval(b, lambda x: 255 - x)

    inverted = Image.merge("RGB", (r, g, b))
    inverted.save(output_path)
