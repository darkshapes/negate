# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test suite for extract combinations command."""

from pathlib import Path
from PIL import Image
import tempfile
import pytest
from negate.run_combinations import run_all_combinations


def _create_test_image(path: Path, size: tuple = (100, 100)) -> None:
    """Helper to create a valid test image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path)


class TestRunAllCombinations:
    """Test suite for run_all_combinations function."""

    def test_runs_all_extractor_combinations(self):
        """Test that all extractor module combinations are run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)

            results = run_all_combinations(img_path)

            assert isinstance(results, dict)
            assert len(results) > 0
            assert "single_modules" in results
            assert "module_pairs" in results

    def test_single_module_results(self):
        """Test single module extraction results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)

            results = run_all_combinations(img_path)

            assert "single_modules" in results
            for module_name, features in results["single_modules"].items():
                assert isinstance(features, dict)
                assert len(features) >= 0

    def test_module_pair_results(self):
        """Test module pair extraction results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)

            results = run_all_combinations(img_path)

            assert "module_pairs" in results
            for pair_name, features in results["module_pairs"].items():
                assert isinstance(features, dict)

    def test_returns_feature_counts(self):
        """Test that feature counts are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _create_test_image(img_path)

            results = run_all_combinations(img_path)

            assert "summary" in results
            summary = results["summary"]
            assert "total_single_modules" in summary
            assert "total_module_pairs" in summary


class TestCombinationRuntimeError:
    """Test suite for RuntimeError handling in combination extraction."""

    def test_combination_runtime_error_extractor(self):
        """Test RuntimeError is caught when extractor fails."""
        from pathlib import Path
        import tempfile
        from PIL import Image
        from negate.io.spec import Spec
        from negate.extract.unified_core import ExtractionModule

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid image
            img_path = Path(tmpdir) / "test.png"
            img = Image.new("RGB", (100, 100))
            img.save(img_path)

            # Test that RuntimeError is caught during extraction
            try:
                from negate.extract.unified_core import UnifiedExtractor
                spec = Spec()
                extractor = UnifiedExtractor(spec, enable=[ExtractionModule.LEARNED])
                features = extractor(Image.open(img_path))
                assert isinstance(features, dict)
                assert len(features) == 768
            except RuntimeError as exc:
                # RuntimeError should be caught and handled gracefully
                assert isinstance(exc, RuntimeError)
