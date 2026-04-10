# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Unified feature extraction interface with interchangeable analyzers.

This module provides a unified interface that allows users to select which
extraction modules to run and in what order. Each analyzer is independent
and can be enabled/disabled via configuration.

Usage:
    >>> from negate.extract import UnifiedExtractor
    >>> extractor = UnifiedExtractor(spec, enable=["artwork", "learned", "vae", "vit"])
    >>> features = extractor(image)
"""

from __future__ import annotations

import gc
from enum import Enum, auto
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from negate.decompose.residuals import Residual
from negate.decompose.wavelet import WaveletContext, WaveletAnalyze
from negate.extract.feature_artwork import ArtworkExtract
from negate.extract.feature_learned import LearnedExtract
from negate.extract.feature_vae import VAEExtract
from negate.extract.feature_vit import VITExtract
from negate.io.spec import Spec


class ExtractionModule(Enum):
    """Extraction module types."""

    ARTWORK = auto()
    LEARNED = auto()
    RESIDUAL = auto()
    WAVELET = auto()
    VAE = auto()
    VIT = auto()


DEFAULT_ENABLED_MODULES = {
    ExtractionModule.ARTWORK,
    ExtractionModule.LEARNED,
    ExtractionModule.RESIDUAL,
    ExtractionModule.WAVELET,
    ExtractionModule.VAE,
    ExtractionModule.VIT,
}


class UnifiedExtractor:
    """Unified feature extraction interface with interchangeable analyzers.

    This class manages multiple extraction modules and allows users to select
    which ones to run and in what order. Each analyzer produces its own set
    of features that are merged into the final result.

    Attributes:
        spec: Configuration specification containing device/dtype settings.
        enabled: Set of enabled extraction module names.
        extractors: Dictionary mapping module names to extractor instances.

    Example:
        >>> from negate.io.spec import Spec
        >>> spec = Spec()
        >>> extractor = UnifiedExtractor(spec, enable=["artwork", "learned"])
        >>> features = extractor(image)
    """

    def __init__(self, spec: Spec, enable: Sequence[ExtractionModule | str] | None = None) -> None:
        """Initialize the unified extractor with selected modules.\n
        :param spec: Specification container with model config and hardware settings.
        :param enable: Sequence of module names to enable. If None, all modules are enabled.
        """
        self.spec = spec
        self.enabled: set[ExtractionModule]
        if enable is None:
            self.enabled = DEFAULT_ENABLED_MODULES.copy()
        else:
            self.enabled = set()
            for mod in enable:
                if isinstance(mod, str):
                    self.enabled.add(ExtractionModule[mod.upper()])
                else:
                    self.enabled.add(mod)
        self.extractors: dict[ExtractionModule, Any] = {}
        self._init_extractors()

    def _init_extractors(self) -> None:
        """Initialize enabled extraction modules."""
        for module in self.enabled:
            match module:
                case ExtractionModule.ARTWORK:
                    self.extractors[ExtractionModule.ARTWORK] = ArtworkExtract()
                case ExtractionModule.LEARNED:
                    self.extractors[ExtractionModule.LEARNED] = LearnedExtract()
                case ExtractionModule.RESIDUAL:
                    self.extractors[ExtractionModule.RESIDUAL] = Residual(self.spec)
                case ExtractionModule.WAVELET:
                    self.extractors[ExtractionModule.WAVELET] = WaveletContext(self.spec, verbose=False)
                case ExtractionModule.VAE:
                    self.extractors[ExtractionModule.VAE] = VAEExtract(self.spec, verbose=False)
                case ExtractionModule.VIT:
                    self.extractors[ExtractionModule.VIT] = VITExtract(self.spec, verbose=False)

    def __call__(self, image: Image.Image | Tensor) -> dict[str, float]:
        """Extract features from a single image using enabled modules.\n
        :param image: Input PIL image or tensor.
        :returns: Dictionary with combined features from all enabled modules.
        """
        results: dict[str, float] = {}

        if ExtractionModule.ARTWORK in self.enabled:
            artwork_features = self.extractors[ExtractionModule.ARTWORK](image)
            results.update(artwork_features)

        if ExtractionModule.LEARNED in self.enabled:
            learned_features = self.extractors[ExtractionModule.LEARNED](image)
            results.update(learned_features)

        if ExtractionModule.RESIDUAL in self.enabled:
            residual_features = self.extractors[ExtractionModule.RESIDUAL](image)
            results.update({k: v for k, v in residual_features.items() if isinstance(v, (int, float))})

        if ExtractionModule.WAVELET in self.enabled:
            wavelet_features = self._extract_wavelet(image)
            results.update(wavelet_features)

        if ExtractionModule.VAE in self.enabled:
            vae_features = self._extract_vae(image)
            results.update(vae_features)

        if ExtractionModule.VIT in self.enabled:
            vit_features = self._extract_vit(image)
            results.update(vit_features)

        return results

    def extract_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        """Extract features from a batch of images.\n
        :param images: List of PIL images.
        :returns: List of feature dictionaries, one per image.
        """
        return [self(image) for image in images]

    def _to_numeric(self, image: Image.Image | Tensor) -> np.ndarray:
        """Convert image to numeric array for residual processing.\n
        :param image: Input image.
        :returns: Grayscale numeric array.
        """
        if isinstance(image, Tensor):
            numeric = image.cpu().numpy()
        else:
            numeric = np.asarray(image)

        while numeric.ndim > 3:
            numeric = numeric.squeeze(0)

        if numeric.ndim == 3 and numeric.shape[0] <= 4:
            numeric = np.moveaxis(numeric, 0, -1)

        from skimage.color import rgb2gray

        gray = rgb2gray(numeric)
        return gray.astype(np.float64)

    def _extract_wavelet(self, image: Image.Image) -> dict[str, float]:
        """Extract wavelet features using WaveletContext.\n
        :param image: Input PIL image.
        :returns: Dictionary of wavelet features.
        """
        wavelet_ctx = self.extractors[ExtractionModule.WAVELET]
        analyzer = WaveletAnalyze(wavelet_ctx)

        try:
            from datasets import Dataset

            dataset = Dataset.from_list([{"image": image}])
            result = analyzer(dataset)
            return result.get("results", [{}])[0] if result.get("results") else {}
        except Exception:
            return {}

    def _extract_vae(self, image: Image.Image) -> dict[str, float]:
        """Extract VAE features.\n
        :param image: Input PIL image.
        :returns: Dictionary of VAE features.
        """
        import torchvision.transforms as T

        vae_extractor = self.extractors[ExtractionModule.VAE]
        transform = T.Compose(
            [
                T.CenterCrop((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        try:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.spec.device, dtype=self.spec.dtype)
            vae_features = vae_extractor(tensor)

            results = {}
            if vae_features.get("features"):
                feat = vae_features["features"][0]
                if isinstance(feat, Tensor):
                    results["vae_latent_mean"] = float(feat.mean())
                    results["vae_latent_std"] = float(feat.std())
                else:
                    results["vae_latent_mean"] = float(np.mean(vae_features["features"]))
                    results["vae_latent_std"] = float(np.std(vae_features["features"]))

            return results
        except Exception:
            return {}

    def _extract_vit(self, image: Image.Image) -> dict[str, float]:
        """Extract VIT features.\n
        :param image: Input PIL image.
        :returns: Dictionary of VIT features.
        """
        try:
            vit_extractor = self.extractors[ExtractionModule.VIT]
            image_features = vit_extractor(image)

            results = {}
            if isinstance(image_features, list) and len(image_features) > 0:
                feat = image_features[0]
                if isinstance(feat, Tensor):
                    results["vit_features_mean"] = float(feat.mean())
                    results["vit_features_std"] = float(feat.std())
                else:
                    results["vit_features_mean"] = float(np.mean(feat))
                    results["vit_features_std"] = float(np.std(feat))

            return results
        except Exception:
            return {}

    def feature_names(self) -> list[str]:
        """Return ordered list of all possible feature names."""
        names = []

        if ExtractionModule.ARTWORK in self.enabled:
            dummy = Image.new("RGB", (255, 255), color="gray")
            names.extend(list(self.extractors[ExtractionModule.ARTWORK](dummy).keys()))

        if ExtractionModule.LEARNED in self.enabled:
            names.extend([f"cnxt_{i}" for i in range(768)])

        if ExtractionModule.RESIDUAL in self.enabled:
            names.extend(["image_mean_ff", "image_std"])

        if ExtractionModule.WAVELET in self.enabled:
            names.extend(["wavelet_error"])

        if ExtractionModule.VAE in self.enabled:
            names.extend(["vae_latent_mean", "vae_latent_std"])

        if ExtractionModule.VIT in self.enabled:
            names.extend(["vit_features_mean", "vit_features_std"])

        return names

    def cleanup(self) -> None:
        """Free resources from all extractors."""
        for name, extractor in self.extractors.items():
            if hasattr(extractor, "cleanup"):
                try:
                    extractor.cleanup()
                except Exception:
                    pass
            if hasattr(extractor, "__exit__"):
                extractor.__exit__(None, None, None)

        gc.collect()
        try:
            if self.spec.device.type != "cpu":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "UnifiedExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


class ExtractorPipeline:
    """Pipeline for running extractors in configurable order.

    This class allows users to define a custom extraction pipeline with
    specific extractors in specific order. Each extractor runs independently
    and results are merged at the end.

    Attributes:
        spec: Configuration specification containing device/dtype settings.
        pipeline: List of (module_name, extractor) tuples in execution order.
    """

    def __init__(self, spec: Spec, order: list[str] | None = None) -> None:
        """Initialize pipeline with specified order.\n
        :param spec: Specification container with model config and hardware settings.
        :param order: List of module names in execution order.
        """
        self.spec = spec
        self.order = order or list(DEFAULT_ENABLED_MODULES)
        self.pipeline: dict[str, Any] = {}
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the extraction pipeline based on order."""
        for module in self.order:
            match module:
                case ExtractionModule.ARTWORK:
                    self.pipeline[ExtractionModule.ARTWORK] = ArtworkExtract()
                case ExtractionModule.LEARNED:
                    self.pipeline[ExtractionModule.LEARNED] = LearnedExtract()
                case ExtractionModule.RESIDUAL:
                    self.pipeline[ExtractionModule.RESIDUAL] = Residual(self.spec)
                case ExtractionModule.WAVELET:
                    self.pipeline[ExtractionModule.WAVELET] = WaveletContext(self.spec, verbose=False)
                case ExtractionModule.VAE:
                    self.pipeline[ExtractionModule.VAE] = VAEExtract(self.spec, verbose=False)
                case ExtractionModule.VIT:
                    self.pipeline[ExtractionModule.VIT] = VITExtract(self.spec, verbose=False)

    def run(self, image: Image.Image | Tensor) -> dict[str, float]:
        """Run the pipeline on a single image.\n
        :param image: Input PIL image or tensor.
        :returns: Dictionary with combined features from pipeline.
        """
        results: dict[str, float] = {}

        for module in self.order:
            if module == ExtractionModule.ARTWORK:
                results.update(self.pipeline[ExtractionModule.ARTWORK](image))
            elif module == ExtractionModule.LEARNED:
                results.update(self.pipeline[ExtractionModule.LEARNED](image))
            elif module == ExtractionModule.RESIDUAL:
                from skimage.color import rgb2gray

                numeric = np.asarray(image)
                if numeric.ndim == 3:
                    numeric = np.moveaxis(numeric, 0, -1)
                gray = rgb2gray(numeric)
                res = self.pipeline[ExtractionModule.RESIDUAL](gray)
                results.update({k: v for k, v in res.items() if isinstance(v, (int, float))})
            elif module == ExtractionModule.WAVELET:
                pass
            elif module == ExtractionModule.VAE:
                pass
            elif module == ExtractionModule.VIT:
                results.update(self._run_vit(image))

        return results

    def _run_vit(self, image: Image.Image) -> dict[str, float]:
        """Run VIT extraction on image."""
        vit_extractor = self.pipeline[ExtractionModule.VIT]
        try:
            image_features = vit_extractor(image)
            if isinstance(image_features, list) and len(image_features) > 0:
                feat = image_features[0]
                if isinstance(feat, Tensor):
                    return {"vit_features_mean": float(feat.mean()), "vit_features_std": float(feat.std())}
        except Exception:
            pass
        return {}

    def cleanup(self) -> None:
        """Clean up all resources in pipeline."""
        for extractor in self.pipeline.values():
            if hasattr(extractor, "cleanup"):
                extractor.cleanup()
        gc.collect()


def create_extractor(spec: Spec, modules: list[str]) -> UnifiedExtractor:
    """Factory function to create a unified extractor with specified modules.\n
    :param spec: Specification container with model config and hardware settings.
    :param modules: List of module names to enable.
    :returns: UnifiedExtractor instance.
    """
    return UnifiedExtractor(spec, enable=modules)


def create_pipeline(spec: Spec, order: list[str]) -> ExtractorPipeline:
    """Factory function to create a pipeline with specified order.\n
    :param spec: Specification container with model config and hardware settings.
    :param order: List of module names in execution order.
    :returns: ExtractorPipeline instance.
    """
    return ExtractorPipeline(spec, order=order)
