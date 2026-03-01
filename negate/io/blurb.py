# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a pes */ -->

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from negate.io.config import config_path
from negate.io.spec import Spec


@dataclass
class Blurb:
    """CLI help text and argument metadata.\n
    Bundles all help strings, defaults, and choices for the argument parser.
    Loaded from `blurb.toml` if present, otherwise uses class-defined defaults.
    """

    _dynamic_attrs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # Models - set post-init via Spec
    model_desc: str = field(default="model to use. Default : ")
    ae_choices: list[str] = field(default_factory=list)
    model_choices: list[str] = field(default_factory=list)

    def __init__(self, spec: Spec):
        self.blurb_path = config_path / "blurb.toml"
        self.spec = spec
        self._load_toml()
        self.default_vit = self.spec.model
        self.default_vae = self.spec.model_config.auto_vae[0]
        self.ae_choices = [ae[0] for ae in self.spec.model_config.list_vae]
        self.ae_choices.append("")
        self.model_choices = list(self.spec.models)

    def _load_toml(self) -> None:
        """Load string fields from blurb.toml if it exists."""
        path = config_path / "blurb.toml"
        if path.exists():
            with open(path, "rb") as blurb_file:
                data = tomllib.load(blurb_file)
            for key, value in data.items():
                setattr(self, key, value)

    @classmethod
    def from_toml(cls, path: Path | str) -> "Blurb":
        """Create Blurb instance with explicit TOML path override."""
        with open(path, "rb") as blurb_file:
            data = tomllib.load(blurb_file)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def infer_model_blurb(self, pair: list[str]) -> str:
        return f"Trained {self.model_desc} {pair}"

    def ae_model_blurb(self) -> str:
        return f"Autoencoder {self.model_desc} {self.default_vae}"

    def vit_model_blurb(self) -> str:
        return f"Vison {self.model_desc} {self.default_vit}"
