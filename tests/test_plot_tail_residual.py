# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for residual plots."""

import inspect
import pandas as pd
import pytest

from negate.metrics import plot_tail_residual


class TestGraphResidual:
    """Test suite for graph_residual function."""

    def test_graph_residual_signature(self):
        """Test graph_residual has correct signature."""
        sig = inspect.signature(plot_tail_residual.graph_residual)
        params = list(sig.parameters.keys())
        assert "spec" in params
        assert "residual_dataframe" in params


class TestGraphKDE:
    """Test suite for graph_kde function."""

    def test_graph_kde_signature(self):
        """Test graph_kde has correct signature."""
        sig = inspect.signature(plot_tail_residual.graph_kde)
        params = list(sig.parameters.keys())
        assert "spec" in params
        assert "residual_dataframe" in params


class TestGraphCohen:
    """Test suite for graph_cohen function."""

    def test_graph_cohen_signature(self):
        """Test graph_cohen has correct signature."""
        sig = inspect.signature(plot_tail_residual.graph_cohen)
        params = list(sig.parameters.keys())
        assert "spec" in params
        assert "residual_dataframe" in params
