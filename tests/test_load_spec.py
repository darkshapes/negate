"""Tests for loading specifications from config files."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


def test_load_spec_from_temp_datestamped_folder():
    """Test that load_spec can load a config from a datestamped results folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results"
    with tempfile.TemporaryDirectory(prefix=timestamp, dir=str(results_dir)) as tmpdir:
        tmp_path = Path(tmpdir)

        source_config = Path(__file__).parent.parent / "tests" / "test_config.toml"
        dest_config = tmp_path / "config.toml"
        shutil.copy(source_config, dest_config)

        from negate.__main__ import load_spec

        loaded_spec = load_spec(Path(tmp_path.stem))

        assert loaded_spec is not None
        assert hasattr(loaded_spec, "opt")
        assert loaded_spec.opt.dim_factor == 666
        assert hasattr(loaded_spec, "model_config")


def test_load_wrong_spec_path():
    """Test that load_spec can load a config from a datestamped results folder."""

    from negate.__main__ import load_spec

    with pytest.raises(FileNotFoundError):
        loaded_spec = load_spec(Path("92525"))

    with pytest.raises(FileNotFoundError):
        loaded_spec = load_spec(Path("config.to"))
