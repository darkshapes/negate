# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for plot save/load functions."""

import inspect
import pandas as pd
import pytest

from negate.metrics import plot_save


class TestSaveFrames:
    """Test suite for save_frames function."""

    def test_save_frames_signature(self):
        """Test save_frames has correct signature."""
        sig = inspect.signature(plot_save.save_frames)
        params = list(sig.parameters.keys())
        assert "data_frame" in params
        assert "model_name" in params


class TestLoadFrames:
    """Test suite for load_frames function."""

    def test_load_frames_signature(self):
        """Test load_frames has correct signature."""
        sig = inspect.signature(plot_save.load_frames)
        params = list(sig.parameters.keys())
        assert "folder_path_name" in params
