# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from pathlib import Path
import json

from PIL import Image
from tqdm import tqdm

from negate.config import Spec
from negate.residuals import Residual
from negate.track import result_path, timestamp


def run():
    res_data = {}
    spec = Spec()
    residual = Residual(spec)
    folder_path = Path("/Users/e6d64/Downloads/real/")

    for img_path in tqdm(folder_path.iterdir(), total=len(os.listdir(str(folder_path))), desc="real.."):
        data = Image.open(str(img_path))
        res_data[img_path.stem] = residual(image=data)

    result_path.mkdir(parents=True, exist_ok=True)
    results_file = str(result_path / f"results_real_{timestamp}.json")
    result_format = {k: str(v) for k, v in res_data.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)
    res_data = {}
    folder_path = Path("/Users/e6d64/Downloads/syn/")

    for img_path in tqdm(folder_path.iterdir(), total=len(os.listdir(str(folder_path))), desc="synth.."):
        data = Image.open(str(img_path))
        res_data[img_path.stem] = residual(image=data)

    results_file = str(result_path / f"results_syn_{timestamp}.json")
    result_format = {k: str(v) for k, v in res_data.items()}
    with open(results_file, "tw", encoding="utf-8") as out_file:
        json.dump(result_format, out_file, ensure_ascii=False, indent=4, sort_keys=True)


if __name__ == "__main__":
    run()
