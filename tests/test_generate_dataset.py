# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
from PIL import Image as PillowImage

from negate.datasets import generate_dataset


def test_generate_dataset_creates_dataset(tmp_path: Path):
    """Test generate_dataset.\n"""

    for i in range(3):
        img = PillowImage.new("RGB", (10, 10), color=(i * 40, i * 40, i * 40))
        img.save(tmp_path / f"img{i}.png")

    (tmp_path / "note.txt").write_text("ignore")

    ds = generate_dataset(tmp_path)

    assert len(ds) == 3
    assert {"image", "label"}.issubset(set(ds.column_names))
    assert all(lbl == 0 for lbl in ds["label"])
