# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from negate.extract import VAEModel, MODEL_MAP


def test_vae_model_map_entries() -> None:
    """Test mapping coverage.
    :return: None"""
    for model in VAEModel:
        assert model in MODEL_MAP, f"{model} missing in MODEL_MAP"
