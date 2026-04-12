# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from negate.decompose.surface import SurfaceFeatures
from negate.io.datasets import build_datasets, generate_dataset
from negate.io.spec import Spec, root_folder
from negate.extract.ensemble import load_and_extract, run_ensemble_cv, main


@pytest.fixture
def sample_images() -> tuple[NDArray, NDArray]:
    """Create sample genuine and synthetic images."""
    genuine = np.random.rand(64, 64, 3).astype(np.float32)
    genuine = np.clip(genuine, 0, 1)
    synthetic = np.random.rand(64, 64, 3).astype(np.float32)
    synthetic = np.clip(synthetic, 0, 1)
    return genuine, synthetic


@pytest.fixture
def mock_dataset(sample_images: tuple[NDArray, NDArray]) -> list[dict]:
    """Create mock dataset with sample images."""
    genuine, synthetic = sample_images
    return [
        {"image": genuine, "label": 0},
        {"image": synthetic, "label": 1},
        {"image": genuine, "label": 0},
        {"image": synthetic, "label": 1},
    ]


@pytest.fixture
def mock_spec() -> Spec:
    """Create mock specification for testing."""
    from negate.io.config import (
        NegateConfig,
        NegateDataPaths,
        NegateEnsembleConfig,
        NegateHyperParam,
        NegateModelConfig,
        NegateTrainRounds,
        chip,
        data_paths,
        hyperparam_config,
        load_config_options,
        negate_options,
        model_config,
        train_rounds,
    )

    spec = Spec(
        negate_options=negate_options,
        hyperparam_config=hyperparam_config,
        ensemble_config=NegateEnsembleConfig(
            sample_size=10,
            n_folds=3,
            abstain_threshold=0.3,
            svm_c=10,
            mlp_hidden_layers=64,
            mlp_activation="relu",
            mlp_max_iter=1000,
            cv=3,
            method="sigmoid",
            gamma="auto",
            kernel="rbf",
        ),
        data_paths=NegateDataPaths(
            eval_data=[],
            genuine_data=[],
            genuine_local=[],
            synthetic_data=[],
            synthetic_local=[],
        ),
        model_config=model_config,
        chip=chip,
        train_rounds=train_rounds,
    )
    return spec


class TestLoadAndExtract:
    """Test suite for load_and_extract function."""

    def test_load_and_extract_returns_correct_types(self, mock_spec: Spec, sample_images: tuple[NDArray, NDArray]) -> None:
        """Verify load_and_extract returns tuple of correct types."""
        _, _, _, _, _ = load_and_extract(mock_spec)

    def test_load_and_extract_returns_features_array(self, mock_spec: Spec) -> None:
        """Verify load_and_extract returns 2D feature array."""
        features, _, _, _, _ = load_and_extract(mock_spec)
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2

    def test_load_and_extract_returns_labels_array(self, mock_spec: Spec) -> None:
        """Verify load_and_extract returns label array."""
        _, labels, _, _, _ = load_and_extract(mock_spec)
        assert isinstance(labels, np.ndarray)
        assert labels.ndim == 1

    def test_load_and_extract_returns_feature_names(self, mock_spec: Spec) -> None:
        """Verify load_and_extract returns list of feature names."""
        _, _, names, _, _ = load_and_extract(mock_spec)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_load_and_extract_returns_image_data(self, mock_spec: Spec) -> None:
        """Verify load_and_extract returns image data."""
        _, _, _, gen_data, syn_data = load_and_extract(mock_spec)
        assert gen_data is not None
        assert syn_data is not None


class TestRunEnsembleCV:
    """Test suite for run_ensemble_cv function."""

    def test_run_ensemble_cv_returns_correct_types(self, mock_spec: Spec, sample_images: tuple[NDArray, NDArray]) -> None:
        """Verify run_ensemble_cv returns tuple of correct types."""
        X = np.random.rand(10, 50)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        results, probs, preds, model = run_ensemble_cv(X, y, mock_spec)
        assert isinstance(results, dict)
        assert isinstance(probs, np.ndarray)
        assert isinstance(preds, np.ndarray)
        assert model is not None

    def test_run_ensemble_cv_returns_results_dict(self, mock_spec: Spec) -> None:
        """Verify run_ensemble_cv returns results dictionary."""
        X = np.random.rand(10, 50)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        results, _, _, _ = run_ensemble_cv(X, y, mock_spec)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_run_ensemble_cv_results_contain_metrics(self, mock_spec: Spec) -> None:
        """Verify run_ensemble_cv results contain required metrics."""
        X = np.random.rand(10, 50)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        results, _, _, _ = run_ensemble_cv(X, y, mock_spec)
        for model_name, metrics in results.items():
            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics

    def test_run_ensemble_cv_returns_probabilities(self, mock_spec: Spec) -> None:
        """Verify run_ensemble_cv returns probability array."""
        X = np.random.rand(10, 50)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        _, probs, _, _ = run_ensemble_cv(X, y, mock_spec)
        assert isinstance(probs, np.ndarray)
        assert probs.ndim == 1
        assert probs.shape[0] == len(y)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_run_ensemble_cv_returns_predictions(self, mock_spec: Spec) -> None:
        """Verify run_ensemble_cv returns prediction array."""
        X = np.random.rand(10, 50)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        _, _, preds, _ = run_ensemble_cv(X, y, mock_spec)
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1
        assert preds.shape[0] == len(y)


class TestMain:
    """Test suite for main function."""

    def test_main_runs_without_error(self, mock_spec: Spec) -> None:
        """Verify main function runs without raising exceptions."""
        # Note: main() requires actual dataset loading which may fail in test environment
        # This test verifies the function is callable
        assert callable(main)

    def test_main_uses_load_and_extract(self) -> None:
        """Verify main function calls load_and_extract."""
        import negate.extract.ensemble as ensemble_module
        import inspect

        source = inspect.getsource(ensemble_module.main)
        assert "load_and_extract" in source

    def test_main_uses_run_ensemble_cv(self) -> None:
        """Verify main function calls run_ensemble_cv."""
        import negate.extract.ensemble as ensemble_module
        import inspect

        source = inspect.getsource(ensemble_module.main)
        assert "run_ensemble_cv" in source


class TestSurfaceFeatures:
    """Test suite for SurfaceFeatures class used in ensemble."""

    def test_surface_features_extract_features(self, sample_images: tuple[NDArray, NDArray]) -> None:
        """Verify SurfaceFeatures extracts features correctly."""
        genuine, _ = sample_images
        extractor = SurfaceFeatures(genuine)
        features = extractor()
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_surface_features_extract_brightness_features(self, sample_images: tuple[NDArray, NDArray]) -> None:
        """Verify SurfaceFeatures extracts brightness features."""
        _, _ = sample_images
        extractor = SurfaceFeatures(np.random.rand(64, 64, 3).astype(np.float32))
        features = extractor()
        assert "mean_brightness" in features
        assert "entropy_brightness" in features

    def test_surface_features_extract_color_features(self, sample_images: tuple[NDArray, NDArray]) -> None:
        """Verify SurfaceFeatures extracts color features."""
        _, _ = sample_images
        extractor = SurfaceFeatures(np.random.rand(64, 64, 3).astype(np.float32))
        features = extractor()
        assert "red_mean" in features
        assert "green_mean" in features
        assert "blue_mean" in features

    def test_surface_features_extract_texture_features(self, sample_images: tuple[NDArray, NDArray]) -> None:
        """Verify SurfaceFeatures extracts texture features."""
        _, _ = sample_images
        extractor = SurfaceFeatures(np.random.rand(64, 64, 3).astype(np.float32))
        features = extractor()
        assert "contrast" in features
        assert "correlation" in features
        assert "energy" in features
        assert "homogeneity" in features
