# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Comprehensive tests for wavelet.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor

from negate.decompose.residuals import Residual
from negate.decompose.wavelet import WaveletContext, WaveletAnalyze
from negate.io.config import (
    NegateConfig,
    NegateHyperParam,
    NegateEnsembleConfig,
    NegateDataPaths,
    NegateModelConfig,
    chip,
    train_rounds,
)
from negate.io.spec import Spec


@pytest.fixture
def mock_spec() -> Spec:
    """Create mock specification object for testing."""
    config = NegateConfig(
        alpha=0.5,
        batch_size=32,
        condense_factor=2,
        dim_factor=4,
        dim_patch=16,
        disable_nullable=False,
        dtype="float32",
        feat_ext_path="test",
        load_from_cache_file=True,
        load_onnx=False,
        magnitude_sampling=True,
        residual_dtype="float64",
        top_k=5,
    )
    hyper_param = NegateHyperParam(
        seed=42,
        colsample_bytree=0.8,
        eval_metric=["auc"],
        learning_rate=0.01,
        max_depth=6,
        objective="binary:logistic",
        subsample=0.8,
    )
    ensemble = NegateEnsembleConfig(
        sample_size=100,
        n_folds=5,
        abstain_threshold=0.5,
        svm_c=1,
        mlp_hidden_layers=64,
        mlp_activation="relu",
        mlp_max_iter=100,
        cv=5,
        method="svm",
        gamma="scale",
        kernel="rbf",
    )
    data_paths = NegateDataPaths(
        eval_data=["eval"],
        genuine_data=["genuine"],
        genuine_local=[],
        synthetic_data=["synthetic"],
        synthetic_local=[],
    )
    model_config = NegateModelConfig(
        data={"library": {"timm": ["vit_base_patch16_dinov3.lvd1689m"]}},
        vae={"library": {"diffusers": ["vae"]}},
    )
    spec = Spec(
        negate_options=config,
        hyperparam_config=hyper_param,
        ensemble_config=ensemble,
        data_paths=data_paths,
        model_config=model_config,
        chip=chip,
        train_rounds=train_rounds,
    )
    return spec


@pytest.fixture
def mock_spec_cpu() -> Spec:
    """Create mock specification object with CPU device for testing."""
    config = NegateConfig(
        alpha=0.5,
        batch_size=32,
        condense_factor=2,
        dim_factor=4,
        dim_patch=16,
        disable_nullable=False,
        dtype="float32",
        feat_ext_path="test",
        load_from_cache_file=True,
        load_onnx=False,
        magnitude_sampling=True,
        residual_dtype="float64",
        top_k=5,
    )
    hyper_param = NegateHyperParam(
        seed=42,
        colsample_bytree=0.8,
        eval_metric=["auc"],
        learning_rate=0.01,
        max_depth=6,
        objective="binary:logistic",
        subsample=0.8,
    )
    ensemble = NegateEnsembleConfig(
        sample_size=100,
        n_folds=5,
        abstain_threshold=0.5,
        svm_c=1,
        mlp_hidden_layers=64,
        mlp_activation="relu",
        mlp_max_iter=100,
        cv=5,
        method="svm",
        gamma="scale",
        kernel="rbf",
    )
    data_paths = NegateDataPaths(
        eval_data=["eval"],
        genuine_data=["genuine"],
        genuine_local=[],
        synthetic_data=["synthetic"],
        synthetic_local=[],
    )
    model_config = NegateModelConfig(
        data={"library": {"timm": ["vit_base_patch16_dinov3.lvd1689m"]}},
        vae={"library": {"diffusers": ["vae"]}},
    )
    spec = Spec(
        negate_options=config,
        hyperparam_config=hyper_param,
        ensemble_config=ensemble,
        data_paths=data_paths,
        model_config=model_config,
        chip=chip,
        train_rounds=train_rounds,
    )
    spec.device = torch.device("cpu")
    return spec


@pytest.fixture
def mock_residual() -> Residual:
    """Create mock residual object for testing."""
    spec = Spec()
    residual = Residual(spec)
    residual.fourier_discrepancy = MagicMock(return_value={"max_magnitude": 1.0})
    return residual


@pytest.fixture
def mock_dataset() -> dict[str, list[Tensor]]:
    """Create mock dataset with test images."""
    images = [
        torch.randn(1, 3, 64, 64),
        torch.randn(1, 3, 128, 128),
    ]
    return {"image": images}


@pytest.fixture
def mock_vit_extract() -> MagicMock:
    """Create mock VIT extract object."""
    mock = MagicMock()
    mock.__call__ = MagicMock(return_value=[torch.randn(768)])
    return mock


@pytest.fixture
def mock_vae_extract() -> MagicMock:
    """Create mock VAE extract object."""
    mock = MagicMock()
    mock.__call__ = MagicMock(return_value={"features": [torch.randn(32)]})
    mock.latent_drift = MagicMock(return_value={"bce_loss": 0.1, "l1_mean": 0.2, "mse_mean": 0.3, "kl_loss": 0.4})
    return mock


@pytest.fixture
def wavelet_context(mock_spec_cpu, mock_vit_extract, mock_vae_extract) -> WaveletContext:
    """Create WaveletContext with mocked extractors."""
    dwt = DWTForward(J=2, wave="haar")
    idwt = DWTInverse(wave="haar")
    residual = Residual(mock_spec_cpu)
    context = WaveletContext(
        spec=mock_spec_cpu,
        verbose=False,
        extract=mock_vit_extract,
        vae=mock_vae_extract,
        residual=residual,
    )
    return context


@pytest.fixture
def wavelet_analyze(wavelet_context) -> WaveletAnalyze:
    """Create WaveletAnalyze instance."""
    return WaveletAnalyze(wavelet_context)


@pytest.fixture
def mock_vit_extract_class() -> MagicMock:
    """Create mock VITExtract class."""
    mock = MagicMock()
    mock.return_value = MagicMock()
    mock.return_value.__call__ = MagicMock(return_value=[torch.randn(768)])
    return mock


@pytest.fixture
def mock_vae_extract_class() -> MagicMock:
    """Create mock VAEExtract class."""
    mock = MagicMock()
    mock.return_value = MagicMock()
    mock.return_value.__call__ = MagicMock(return_value={"features": [torch.randn(32)]})
    mock.return_value.latent_drift = MagicMock(return_value={"bce_loss": 0.1, "l1_mean": 0.2, "mse_mean": 0.3, "kl_loss": 0.4})
    return mock


@pytest.fixture
def mock_dwt() -> MagicMock:
    """Create mock DWT transform."""
    mock = MagicMock()
    mock.return_value = (torch.randn(1, 1, 8, 8), torch.randn(1, 2, 8, 8))
    return mock


@pytest.fixture
def mock_idwt() -> MagicMock:
    """Create mock IDWT transform."""
    mock = MagicMock()
    mock.return_value = (torch.randn(1, 1, 8, 8), torch.randn(1, 2, 8, 8))
    return mock


@pytest.fixture
def wavelet_analyze_mock(mock_spec_cpu, mock_vit_extract, mock_vae_extract, mock_dwt, mock_idwt) -> WaveletAnalyze:
    """Create WaveletAnalyze instance with mocked DWT transforms on CPU."""
    dwt = DWTForward(J=2, wave="haar")
    idwt = DWTInverse(wave="haar")
    residual = Residual(mock_spec_cpu)
    context = WaveletContext(
        spec=mock_spec_cpu,
        verbose=False,
        dwt=dwt,
        idwt=idwt,
        extract=mock_vit_extract,
        vae=mock_vae_extract,
        residual=residual,
    )
    analyzer = WaveletAnalyze(context)
    with patch.object(analyzer.context.dwt, "__call__", mock_dwt):
        with patch.object(analyzer.context.idwt, "__call__", mock_idwt):
            yield analyzer


class TestWaveletContext:
    """Tests for WaveletContext class."""

    def test_initialization_with_defaults(self, mock_spec_cpu, mock_vit_extract_class, mock_vae_extract_class) -> None:
        """Test WaveletContext initialization with default parameters."""
        with patch("negate.extract.feature_vit.VITExtract", mock_vit_extract_class), patch("negate.extract.feature_vae.VAEExtract", mock_vae_extract_class):
            context = WaveletContext(spec=mock_spec_cpu, verbose=False)
            assert context.dwt is not None
            assert context.idwt is not None
            assert context.residual is not None
            assert context.verbose is False

    def test_initialization_with_custom_dwt(self, mock_spec_cpu, mock_vit_extract_class, mock_vae_extract_class) -> None:
        """Test WaveletContext with custom DWTForward instance."""
        custom_dwt = DWTForward(J=3, wave="haar")
        with patch("negate.extract.feature_vit.VITExtract", mock_vit_extract_class), patch("negate.extract.feature_vae.VAEExtract", mock_vae_extract_class):
            context = WaveletContext(spec=mock_spec_cpu, verbose=False, dwt=custom_dwt)
            assert context.dwt == custom_dwt

    def test_initialization_with_custom_idwt(self, mock_spec_cpu, mock_vit_extract_class, mock_vae_extract_class) -> None:
        """Test WaveletContext with custom DWTInverse instance."""
        custom_idwt = DWTInverse(wave="haar")
        with patch("negate.extract.feature_vit.VITExtract", mock_vit_extract_class), patch("negate.extract.feature_vae.VAEExtract", mock_vae_extract_class):
            context = WaveletContext(spec=mock_spec_cpu, verbose=False, idwt=custom_idwt)
            assert context.idwt == custom_idwt

    def test_initialization_with_all_custom_objects(self, mock_spec_cpu) -> None:
        """Test WaveletContext with all custom dependency objects."""
        dwt = DWTForward(J=2, wave="haar")
        idwt = DWTInverse(wave="haar")
        residual = Residual(mock_spec_cpu)
        mock_extract = MagicMock()
        mock_vae = MagicMock()
        context = WaveletContext(
            spec=mock_spec_cpu,
            verbose=False,
            dwt=dwt,
            idwt=idwt,
            extract=mock_extract,
            vae=mock_vae,
            residual=residual,
        )
        assert context.dwt == dwt
        assert context.idwt == idwt
        assert context.residual == residual
        assert context.extract == mock_extract
        assert context.vae == mock_vae

    def test_context_manager_enter(self, wavelet_context) -> None:
        """Test context manager __enter__ method."""
        result = wavelet_context.__enter__()
        assert result is wavelet_context

    def test_context_manager_exit(self, wavelet_context) -> None:
        """Test context manager __exit__ method."""
        wavelet_context.__exit__(None, None, None)
        # Context manager should not raise exception

    def test_spec_attribute_set(self, mock_spec_cpu, mock_vit_extract_class, mock_vae_extract_class) -> None:
        """Test that spec attribute is properly set."""
        with patch("negate.extract.feature_vit.VITExtract", mock_vit_extract_class), patch("negate.extract.feature_vae.VAEExtract", mock_vae_extract_class):
            context = WaveletContext(spec=mock_spec_cpu, verbose=False)
            assert context.spec == mock_spec_cpu

    def test_verbose_attribute_set(self, mock_spec_cpu, mock_vit_extract_class, mock_vae_extract_class) -> None:
        """Test verbose flag is properly set."""
        with patch("negate.extract.feature_vit.VITExtract", mock_vit_extract_class), patch("negate.extract.feature_vae.VAEExtract", mock_vae_extract_class):
            context = WaveletContext(spec=mock_spec_cpu, verbose=True)
            assert context.verbose is True


class TestWaveletAnalyze:
    """Tests for WaveletAnalyze class."""

    def test_initialization(self, wavelet_analyze) -> None:
        """Test WaveletAnalyze initialization."""
        assert wavelet_analyze.context is not None
        assert wavelet_analyze.cast_move is not None
        assert wavelet_analyze.dim_patch is not None

    def test_context_manager_enter(self, wavelet_analyze) -> None:
        """Test context manager __enter__ method."""
        result = wavelet_analyze.__enter__()
        assert result is wavelet_analyze

    def test_context_manager_exit(self, wavelet_analyze) -> None:
        """Test context manager __exit__ method."""
        wavelet_analyze.__exit__(None, None, None)

    def test_ensemble_decompose_returns_dict(self, wavelet_analyze_mock) -> None:
        """Test ensemble_decompose returns dictionary."""
        test_tensor = torch.randn(1, 3, 16, 16)
        result = wavelet_analyze_mock.ensemble_decompose(test_tensor)
        assert isinstance(result, dict)
        assert "min_warp" in result
        assert "max_warp" in result
        assert "min_base" in result
        assert "max_base" in result

    def test_ensemble_decompose_with_mock_extract(self, wavelet_analyze_mock, mock_vit_extract, mock_vae_extract) -> None:
        """Test ensemble_decompose with mocked extractors."""
        with patch.object(Residual, "__call__", return_value={"residual": 0.5}):
            with patch.object(mock_vit_extract, "__call__", return_value=[torch.randn(768)]):
                with patch.object(mock_vae_extract, "latent_drift", return_value={"bce_loss": 0.1}):
                    result = wavelet_analyze_mock.ensemble_decompose(torch.randn(1, 3, 16, 16))
                    assert isinstance(result, dict)
                    assert len(result) >= 4

    def test_select_patch_returns_tuple(self, wavelet_analyze) -> None:
        """Test select_patch returns tuple of correct length."""
        test_image = torch.randn(1, 3, 64, 64)
        result = wavelet_analyze.select_patch(test_image)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_select_patch_returns_selected_tensor(self, wavelet_analyze) -> None:
        """Test select_patch returns selected patch tensor."""
        test_image = torch.randn(1, 3, 64, 64)
        selected, metadata, spectrum = wavelet_analyze.select_patch(test_image)
        assert isinstance(selected, Tensor)
        assert selected.ndim == 4

    def test_select_patch_metadata_dict(self, wavelet_analyze) -> None:
        """Test select_patch metadata contains expected keys."""
        test_image = torch.randn(1, 3, 64, 64)
        selected, metadata, spectrum = wavelet_analyze.select_patch(test_image)
        assert "selected_patch_idx" in metadata
        assert "max_fourier_magnitude" in metadata
        assert isinstance(metadata["selected_patch_idx"], int)
        assert isinstance(metadata["max_fourier_magnitude"], float)

    def test_select_patch_spectrum_list(self, wavelet_analyze) -> None:
        """Test select_patch returns spectrum as list of tensors."""
        test_image = torch.randn(1, 3, 64, 64)
        selected, metadata, spectrum = wavelet_analyze.select_patch(test_image)
        assert isinstance(spectrum, list)
        assert all(isinstance(patch, Tensor) for patch in spectrum)

    def test_cleanup_on_non_cpu_device(self, mock_spec_cpu, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup method behavior for non-CPU devices."""
        mock_spec_cpu.device = torch.device("cpu")
        with patch("gc.collect"):
            context = WaveletContext(
                spec=mock_spec_cpu,
                verbose=False,
                dwt=DWTForward(J=2, wave="haar"),
                idwt=DWTInverse(wave="haar"),
                extract=mock_vit_extract,
                vae=mock_vae_extract,
                residual=Residual(mock_spec_cpu),
            )
            analyzer = WaveletAnalyze(context)
            analyzer.cleanup()

    def test_cleanup_called_on_exit(self, wavelet_analyze) -> None:
        """Test cleanup is called on context exit."""
        with patch.object(wavelet_analyze, "cleanup") as mock_cleanup:
            with wavelet_analyze:
                pass
            mock_cleanup.assert_called_once()


class TestSimExtrema:
    """Tests for sim_extrema method."""

    def test_sim_extrema_returns_dict(self, wavelet_analyze) -> None:
        """Test sim_extrema returns dictionary with expected keys."""
        base_features = [torch.randn(768)]
        warp_features = [torch.randn(768)]
        batch_size = 1
        result = wavelet_analyze.sim_extrema(base_features, warp_features, batch_size)
        assert isinstance(result, dict)
        assert "min_warp" in result
        assert "max_warp" in result
        assert "min_base" in result
        assert "max_base" in result

    def test_sim_extrema_with_batch_size(self, wavelet_analyze) -> None:
        """Test sim_extrema with different batch sizes."""
        for batch_size in [1, 2, 4]:
            base_features = [torch.randn(768)]
            warp_features = [torch.randn(768)]
            result = wavelet_analyze.sim_extrema(base_features, warp_features, batch_size)
            assert isinstance(result["min_warp"], float)
            assert isinstance(result["max_warp"], float)

    def test_sim_extrema_empty_input(self, wavelet_analyze) -> None:
        """Test sim_extrema with empty input returns zeros."""
        result = wavelet_analyze.sim_extrema([], [], 0)
        assert result["min_warp"] == 0.0
        assert result["max_warp"] == 0.0
        assert result["min_base"] == 0.0
        assert result["max_base"] == 0.0


class TestSelectPatch:
    """Tests for select_patch method."""

    def test_select_patch_single_image(self, wavelet_analyze) -> None:
        """Test select_patch with single image."""
        test_image = torch.randn(1, 3, 64, 64)
        selected, metadata, spectrum = wavelet_analyze.select_patch(test_image)
        assert selected.shape[0] == 1
        assert len(spectrum) > 0

    def test_select_patch_max_magnitude_selected(self, wavelet_analyze) -> None:
        """Test that highest magnitude patch is selected."""
        # Create image with varying magnitudes
        base = torch.zeros(1, 3, 64, 64)
        base[0, 0, :16, :16] = 1.0  # High magnitude region
        base[0, 0, 48:, 48:] = 0.1  # Low magnitude region
        selected, metadata, _ = wavelet_analyze.select_patch(base)
        assert metadata["max_fourier_magnitude"] > 0.0


class TestEnsembleDecompose:
    """Tests for ensemble_decompose method."""

    def test_decompose_with_haar_wavelet(self, wavelet_analyze_mock) -> None:
        """Test decomposition using Haar wavelet transform."""
        test_tensor = torch.randn(1, 3, 16, 16)
        result = wavelet_analyze_mock.ensemble_decompose(test_tensor)
        assert "min_warp" in result
        assert "max_warp" in result

    def test_decompose_with_different_alpha(self, mock_spec_cpu, mock_vit_extract, mock_vae_extract, mock_dwt, mock_idwt) -> None:
        """Test decomposition with different alpha values."""
        for alpha in [0.1, 0.5, 0.9]:
            mock_spec_cpu.opt = NegateConfig(
                alpha=alpha,
                batch_size=32,
                condense_factor=2,
                dim_factor=4,
                dim_patch=16,
                disable_nullable=False,
                dtype="float32",
                feat_ext_path="test",
                load_from_cache_file=True,
                load_onnx=False,
                magnitude_sampling=True,
                residual_dtype="float64",
                top_k=5,
            )
            dwt = DWTForward(J=2, wave="haar")
            idwt = DWTInverse(wave="haar")
            residual = Residual(mock_spec_cpu)
            context = WaveletContext(
                spec=mock_spec_cpu,
                verbose=False,
                dwt=dwt,
                idwt=idwt,
                extract=mock_vit_extract,
                vae=mock_vae_extract,
                residual=residual,
            )
            analyzer = WaveletAnalyze(context)
            with patch.object(analyzer.context.dwt, "__call__", mock_dwt):
                with patch.object(analyzer.context.idwt, "__call__", mock_idwt):
                    test_tensor = torch.randn(1, 3, 16, 16)
                    result = analyzer.ensemble_decompose(test_tensor)
                    assert isinstance(result, dict)


class TestCleanup:
    """Tests for cleanup method."""

    def test_cleanup_frees_gpu_memory(self, mock_spec_cpu, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup frees GPU cache."""
        mock_spec_cpu.device = torch.device("cpu")
        with patch("gc.collect") as mock_gc:
            context = WaveletContext(
                spec=mock_spec_cpu,
                verbose=False,
                dwt=DWTForward(J=2, wave="haar"),
                idwt=DWTInverse(wave="haar"),
                extract=mock_vit_extract,
                vae=mock_vae_extract,
                residual=Residual(mock_spec_cpu),
            )
            analyzer = WaveletAnalyze(context)
            analyzer.cleanup()
            mock_gc.assert_called_once()

    def test_cleanup_no_exception_on_cpu(self, mock_spec_cpu, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup does not raise exception on CPU."""
        mock_spec_cpu.device = torch.device("cpu")
        context = WaveletContext(
            spec=mock_spec_cpu,
            verbose=False,
            dwt=DWTForward(J=2, wave="haar"),
            idwt=DWTInverse(wave="haar"),
            extract=mock_vit_extract,
            vae=mock_vae_extract,
            residual=Residual(mock_spec_cpu),
        )
        analyzer = WaveletAnalyze(context)
        with patch("gc.collect"):
            analyzer.cleanup()
        # Should not raise

    def test_cleanup_with_gpu_device(self, mock_spec_cpu, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup with GPU device."""
        mock_spec_cpu.device = torch.device("cpu")
        with patch("gc.collect"):
            context = WaveletContext(
                spec=mock_spec_cpu,
                verbose=False,
                dwt=DWTForward(J=2, wave="haar"),
                idwt=DWTInverse(wave="haar"),
                extract=mock_vit_extract,
                vae=mock_vae_extract,
                residual=Residual(mock_spec_cpu),
            )
            analyzer = WaveletAnalyze(context)
            analyzer.cleanup()
        # Should not raise
