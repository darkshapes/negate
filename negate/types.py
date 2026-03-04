# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class OriginLabel(IntEnum):
    """Discrete model output labels for image origin."""

    GNE = 0
    SYN = 1

    @classmethod
    def coerce(cls, value: OriginLabel | int) -> OriginLabel:
        """Normalize ints and enum instances to OriginLabel."""
        if isinstance(value, cls):
            return value
        return cls(int(value))

    @classmethod
    def from_probability(cls, probability: float, threshold: float = 0.5) -> OriginLabel:
        """Map model probability to a discrete origin label."""
        return cls.SYN if probability > threshold else cls.GNE


class InferenceModel(str, Enum):
    """Inference model role used in weighted certainty output."""

    AE = "ae"
    DC = "dc"


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Typed model output carrying probability and enum label."""

    probability: float
    origin: OriginLabel

    @classmethod
    def from_probability(cls, probability: float, threshold: float = 0.5) -> ModelOutput:
        prob = float(probability)
        return cls(probability=prob, origin=OriginLabel.from_probability(prob, threshold=threshold))
