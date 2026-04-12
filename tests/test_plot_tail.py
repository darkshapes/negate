# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for tail separation plots."""

import inspect
import pandas as pd
import pytest

from negate.metrics import plot_tail


class TestTailSeparationKeys:
    """Test suite for tail separation keys."""

    def test_wavelet_keys_count(self):
        """Test wavelet_keys has correct number of keys."""
        assert len(plot_tail.wavelet_keys) == 4

    def test_residual_keys_count(self):
        """Test residual_keys has correct number of keys."""
        assert len(plot_tail.residual_keys) == 19


class TestGraphWavelet:
    """Test suite for graph_wavelet function."""

    def test_graph_wavelet_signature(self):
        """Test graph_wavelet has correct signature."""
        sig = inspect.signature(plot_tail.graph_wavelet)
        params = list(sig.parameters.keys())
        assert "spec" in params
        assert "wavelet_dataframe" in params


class TestGraphTailSeparations:
    """Test suite for graph_tail_separations function."""

    def test_graph_tail_separations_signature(self):
        """Test graph_tail_separations has correct signature."""
        sig = inspect.signature(plot_tail.graph_tail_separations)
        params = list(sig.parameters.keys())
        assert "spec" in params
        assert "scores_dataframe" in params
