from __future__ import annotations

import numpy as np

from negate.types import ModelOutput, OriginLabel
from negate.metrics.heuristics import compute_weighted_certainty, model_accuracy


def test_model_output_from_probability_uses_origin_enum() -> None:
    low = ModelOutput.from_probability(0.49)
    high = ModelOutput.from_probability(0.51)

    assert low.origin is OriginLabel.GNE
    assert high.origin is OriginLabel.SYN


def test_model_accuracy_returns_origin_enum_labels() -> None:
    result = model_accuracy(np.array([0.2, 0.8]), label=OriginLabel.SYN)

    assert result[0][0] is OriginLabel.GNE
    assert result[1][0] is OriginLabel.SYN


def test_compute_weighted_certainty_accepts_model_output_enums(capsys) -> None:
    ae = {"pred": [ModelOutput.from_probability(0.2)]}
    dc = {"pred": [ModelOutput.from_probability(0.8)]}

    compute_weighted_certainty(ae, dc, label=OriginLabel.SYN)

    output = capsys.readouterr().out
    assert "Model Accuracy:" in output
    assert " :GNE " in output or " :SYN " in output


def test_compute_weighted_certainty_accepts_legacy_float_predictions(capsys) -> None:
    ae = {"pred": [0.2]}
    dc = {"pred": [0.8]}

    compute_weighted_certainty(ae, dc)

    output = capsys.readouterr().out
    assert " :GNE " in output or " :SYN " in output
