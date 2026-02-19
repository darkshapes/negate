# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from pathlib import Path
import json

from PIL import Image
from tqdm import tqdm

from negate.config import Spec
from negate.residuals import Residual
from negate.train import result_path, timestamp


def test_residual_processing():
    res_data = {}
    spec = Spec()
    residual = Residual(spec)
    image_file = [Path(__file__).parent / "x_p.webp"]

    for img_path in tqdm(image_file, total=len(image_file), desc="real.."):
        data = Image.open(str(img_path))
        data = data.convert("RGB")
        res_data[img_path.stem] = residual(image=data)

    result_path.mkdir(parents=True, exist_ok=True)
    results_file = str(result_path / f"results_real_{timestamp}.json")
    result_format = {k: str(v) for k, v in res_data.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)
    res_data = {}
