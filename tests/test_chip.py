# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest
from negate.io.config import Chip


def test_chip():
    import torch
    import numpy as np

    chip = Chip()

    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    print(expected)
    assert chip.device == torch.device(expected)

    if expected != "cpu":
        expected = torch.float16
        assert chip.dtype == expected
    else:
        expected = torch.float32
        assert chip.dtype == expected

    expected = NotImplementedError
    with pytest.raises(expected):
        chip.dtype = "bfloat16"

    chip.dtype = "float32"
    expected = np.float32
    assert chip.np_dtype == expected


test_chip()
