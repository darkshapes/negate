# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Tests for wavelet.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor

from negate.decompose.residuals import Residual
from negate.decompose.wavelet import WaveletAnalyze, WaveletContext
from negate.io.spec import Spec


@pytest.fixture
def mock_spec() -> Spec:
    """Create mock specification object for testing."""
    config = MagicMock()
    config.alpha = 0.5
    config.condense_factor = 2
    config.top_k = 4
    config.dim_factor = 3
    config.dim_patch = 256
    config.dtype = torch.float32
    config.disable_nullable = False
    config.load_from_cache_file = False
    config.load_onnx = False
    config.magnitude_sampling = True
    config.residual_dtype = "float64"

    hyper_param = MagicMock()
    hyper_param.seed = 42
    hyper_param.colsample_bytree = 0.8
    hyper_param.eval_metric = ["auc"]
    hyper_param.learning_rate = 0.01
    hyper_param.max_depth = 4
    hyper_param.objective = "binary:logistic"
    hyper_param.subsample = 0.8

    ensemble = MagicMock()
    ensemble.sample_size = 100
    ensemble.n_folds = 5
    ensemble.abstain_threshold = 0.3
    ensemble.svm_c = 10.0
    ensemble.mlp_hidden_layers = 100
    ensemble.mlp_activation = "relu"
    ensemble.mlp_max_iter = 1000
    ensemble.cv = 3
    ensemble.method = "sigmoid"
    ensemble.gamma = "scale"
    ensemble.kernel = "rbf"

    data_paths = MagicMock()
    data_paths.eval_data = ["eval"]
    data_paths.genuine_data = ["genuine"]
    data_paths.genuine_local = []
    data_paths.synthetic_data = ["synthetic"]
    data_paths.synthetic_local = []

    model_config = MagicMock()
    model_config.data = {"library": {"timm": ["vit_base_patch16_dinov3.lvd1689m"]}}
    model_config.vae = {"library": {"diffusers": ["vae"]}}

    spec = Spec(
        negate_options=config,
        hyperparam_config=hyper_param,
        ensemble_config=ensemble,
        data_paths=data_paths,
        model_config=model_config,
    )
    spec.device = torch.device("cpu")
    spec.opt = config
    return spec


@pytest.fixture
def mock_vit_extract() -> MagicMock:
    """Create mock VITExtract."""
    mock = MagicMock()
    mock.return_value = [torch.randn(768)]
    return mock


@pytest.fixture
def mock_vae_extract() -> MagicMock:
    """Create mock VAEExtract."""
    mock = MagicMock()
    mock.return_value = {"features": [torch.randn(32)]}
    mock.latent_drift = MagicMock(return_value={"bce_loss": 0.1, "l1_mean": 0.2, "mse_mean": 0.3, "kl_loss": 0.4})
    return mock


@pytest.fixture
def wavelet_context(mock_spec, mock_vit_extract, mock_vae_extract) -> WaveletContext:
    """Create WaveletContext instance with mocked extractors."""
    residual = Residual(mock_spec)
    return WaveletContext(
        spec=mock_spec,
        verbose=False,
        extract=mock_vit_extract,
        vae=mock_vae_extract,
        residual=residual,
    )


class TestWaveletContext:
    """Tests for WaveletContext class."""

    def test_initialization_with_defaults(self, mock_spec) -> None:
        """Test WaveletContext initialization with default parameters."""
        with patch("negate.extract.feature_vit.VITExtract", return_value=MagicMock(return_value=[torch.randn(768)])):
            with patch("negate.extract.feature_vae.VAEExtract", return_value=MagicMock(return_value={"features": [torch.randn(32)]})):
                with patch("negate.decompose.residuals.Residual", return_value=MagicMock()):
                    context = WaveletContext(spec=mock_spec, verbose=False)
                    assert context.dwt is not None
                    assert context.idwt is not None
                    assert context.residual is not None
                    assert context.extract is not None
                    assert context.vae is not None
                    assert context.verbose is False

    def test_initialization_with_custom_dwt(self, mock_spec) -> None:
        """Test WaveletContext with custom DWTForward instance."""
        custom_dwt = DWTForward(J=3, wave="haar")
        with patch("negate.extract.feature_vit.VITExtract", return_value=MagicMock(return_value=[torch.randn(768)])):
            with patch("negate.extract.feature_vae.VAEExtract", return_value=MagicMock(return_value={"features": [torch.randn(32)]})):
                with patch("negate.decompose.residuals.Residual", return_value=MagicMock()):
                    context = WaveletContext(spec=mock_spec, verbose=False, dwt=custom_dwt)
                    assert context.dwt == custom_dwt

    def test_initialization_with_custom_idwt(self, mock_spec) -> None:
        """Test WaveletContext with custom DWTInverse instance."""
        custom_idwt = DWTInverse(wave="haar")
        with patch("negate.extract.feature_vit.VITExtract", return_value=MagicMock(return_value=[torch.randn(768)])):
            with patch("negate.extract.feature_vae.VAEExtract", return_value=MagicMock(return_value={"features": [torch.randn(32)]})):
                with patch("negate.decompose.residuals.Residual", return_value=MagicMock()):
                    context = WaveletContext(spec=mock_spec, verbose=False, idwt=custom_idwt)
                    assert context.idwt == custom_idwt

    def test_initialization_with_all_custom_objects(self, mock_spec) -> None:
        """Test WaveletContext with all custom dependency objects."""
        dwt = DWTForward(J=2, wave="haar")
        idwt = DWTInverse(wave="haar")
        residual = Residual(mock_spec)
        mock_extract = MagicMock()
        mock_vae = MagicMock()
        context = WaveletContext(
            spec=mock_spec,
            verbose=False,
            dwt=dwt,
            idwt=idwt,
            extract=mock_extract,
            vae=mock_vae,
            residual=residual,
        )
        assert context.dwt == dwt
        assert context.idwt == idwt
        assert context.extract == mock_extract
        assert context.vae == mock_vae
        assert context.residual == residual

    def test_context_manager_enter(self, wavelet_context) -> None:
        """Test WaveletContext __enter__ method."""
        with wavelet_context as ctx:
            assert ctx is wavelet_context

    def test_context_manager_exit(self, wavelet_context) -> None:
        """Test WaveletContext __exit__ method."""
        with wavelet_context:
            pass

    def test_spec_attribute_set(self, wavelet_context) -> None:
        """Test spec attribute is set."""
        assert wavelet_context.spec is not None


class TestWaveletAnalyze:
    """Tests for WaveletAnalyze class."""

    def test_initialization(self, wavelet_context) -> None:
        """Test WaveletAnalyze initialization."""
        analyzer = WaveletAnalyze(wavelet_context)
        assert analyzer.context is wavelet_context

    def test_context_manager_enter(self, wavelet_context) -> None:
        """Test WaveletAnalyze __enter__ method."""
        with WaveletAnalyze(wavelet_context) as analyzer:
            assert analyzer is not None

    def test_context_manager_exit(self, wavelet_context) -> None:
        """Test WaveletAnalyze __exit__ method."""
        with WaveletAnalyze(wavelet_context):
            pass

    def test_cleanup_on_non_cpu_device(self, mock_spec, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup behavior on non-CPU device."""
        mock_spec.device = torch.device("cuda")
        residual = Residual(mock_spec)
        context = WaveletContext(
            spec=mock_spec,
            verbose=False,
            extract=mock_vit_extract,
            vae=mock_vae_extract,
            residual=residual,
        )
        with patch("torch.cuda.empty_cache"):
            with patch("gc.collect"):
                with WaveletAnalyze(context) as analyzer:
                    analyzer.cleanup()


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_frees_gpu_memory(self, mock_spec, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup frees GPU cache."""
        mock_spec.device = torch.device("cuda")
        residual = Residual(mock_spec)
        context = WaveletContext(
            spec=mock_spec,
            verbose=False,
            extract=mock_vit_extract,
            vae=mock_vae_extract,
            residual=residual,
        )
        with patch("torch.cuda.empty_cache") as mock_empty:
            with patch("gc.collect") as mock_gc:
                with WaveletAnalyze(context) as analyzer:
                    analyzer.cleanup()
                    mock_empty.assert_called_once()
                    mock_gc.assert_called_once()

    def test_cleanup_no_exception_on_cpu(self, mock_spec, mock_vit_extract, mock_vae_extract) -> None:
        """Test cleanup works without exception on CPU."""
        mock_spec.device = torch.device("cpu")
        residual = Residual(mock_spec)
        context = WaveletContext(
            spec=mock_spec,
            verbose=False,
            extract=mock_vit_extract,
            vae=mock_vae_extract,
            residual=residual,
        )
        with patch("torch.cuda.empty_cache") as mock_empty:
            with WaveletAnalyze(context) as analyzer:
                analyzer.cleanup()
                mock_empty.assert_not_called()
