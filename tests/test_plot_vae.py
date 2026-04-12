# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for VAE plots."""

import inspect
import numpy as np
import pandas as pd
import pytest

from negate.metrics import plot_vae


class TestVAELossKeys:
    """Test suite for VAE loss keys."""

    def test_vae_loss_keys_count(self):
        """Test vae_loss_keys has correct number of keys."""
        assert len(plot_vae.vae_loss_keys) == 8


class TestGraphVAELoss:
    """Test suite for graph_vae_loss function."""

    def test_graph_vae_loss_signature(self):
        """Test graph_vae_loss has correct signature."""
        sig = inspect.signature(plot_vae.graph_vae_loss)
        params = list(sig.parameters.keys())
        assert "spec" in params
        assert "vae_dataframe" in params


class TestGraphTrainVariance:
    """Test suite for graph_train_variance function."""

    def test_graph_train_variance_signature(self):
        """Test graph_train_variance has correct signature."""
        sig = inspect.signature(plot_vae.graph_train_variance)
        params = list(sig.parameters.keys())
        assert "train_result" in params
        assert "spec" in params
