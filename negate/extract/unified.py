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

    def __init__(self, spec: Spec, enable: Sequence[str] | None = None) -> None:
        """Initialize the unified extractor with selected modules.\n
        :param spec: Specification container with model config and hardware settings.
        :param enable: Sequence of module names to enable. If None, all modules are enabled.
        """
        self.spec = spec
        self.enabled = set(enable) if enable else {"artwork", "learned", "vae", "vit", "residual", "wavelet"}
        self.extractors: dict[str, Any] = {}
        self._init_extractors()

    def _init_extractors(self) -> None:
        """Initialize enabled extraction modules."""
        if "artwork" in self.enabled:
            self.extractors["artwork"] = ArtworkExtract()
        if "learned" in self.enabled:
            self.extractors["learned"] = LearnedExtract()
        if "residual" in self.enabled:
            self.extractors["residual"] = Residual(self.spec)
        if "wavelet" in self.enabled:
            self.extractors["wavelet"] = WaveletContext(self.spec, verbose=False)
        if "vae" in self.enabled:
            self.extractors["vae"] = VAEExtract(self.spec, verbose=False)
        if "vit" in self.enabled:
            self.extractors["vit"] = VITExtract(self.spec, verbose=False)

    def __call__(self, image: Image.Image | Tensor) -> dict[str, float]:
        """Extract features from a single image using enabled modules.\n
        :param image: Input PIL image or tensor.
        :returns: Dictionary with combined features from all enabled modules.
        """
        results: dict[str, float] = {}

        if "artwork" in self.enabled:
            artwork_features = self.extractors["artwork"](image)
            results.update(artwork_features)

        if "learned" in self.enabled:
            learned_features = self.extractors["learned"](image)
            results.update(learned_features)

        if "residual" in self.enabled:
            residual_features = self.extractors["residual"](self._to_numeric(image))
            results.update({k: v for k, v in residual_features.items() if isinstance(v, (int, float))})

        if "wavelet" in self.enabled:
            wavelet_features = self._extract_wavelet(image)
            results.update(wavelet_features)

        if "vae" in self.enabled:
            vae_features = self._extract_vae(image)
            results.update(vae_features)

        if "vit" in self.enabled:
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
        from torch import nn

        wavelet_ctx = self.extractors["wavelet"]
        analyzer = WaveletAnalyze(wavelet_ctx)

        try:
            from datasets import Dataset

            dataset = Dataset.from_iterable([{"image": image}])
            result = analyzer(dataset)
            return result.get("results", [{}])[0] if result.get("results") else {}
        except Exception as e:
            return {"wavelet_error": float(str(e)) if str(e) else 0.0}

    def _extract_vae(self, image: Image.Image) -> dict[str, float]:
        """Extract VAE features.\n
        :param image: Input PIL image.
        :returns: Dictionary of VAE features.
        """
        import torchvision.transforms as T

        vae_extractor = self.extractors["vae"]
        transform = T.Compose(
            [
                T.CenterCrop((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

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

    def _extract_vit(self, image: Image.Image) -> dict[str, float]:
        """Extract VIT features.\n
        :param image: Input PIL image.
        :returns: Dictionary of VIT features.
        """
        vit_extractor = self.extractors["vit"]
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

    def feature_names(self) -> list[str]:
        """Return ordered list of all possible feature names."""
        names = []

        if "artwork" in self.enabled:
            dummy = Image.new("RGB", (255, 255), color="gray")
            names.extend(list(self.extractors["artwork"](dummy).keys()))

        if "learned" in self.enabled:
            names.extend([f"cnxt_{i}" for i in range(768)])

        if "residual" in self.enabled:
            names.extend(["image_mean_ff", "image_std"])

        if "wavelet" in self.enabled:
            names.extend(["wavelet_error"])

        if "vae" in self.enabled:
            names.extend(["vae_latent_mean", "vae_latent_std"])

        if "vit" in self.enabled:
            names.extend(["vit_features_mean", "vit_features_std"])

        return names

    def cleanup(self) -> None:
        """Free resources from all extractors."""
        for name, extractor in self.extractors.items():
            if hasattr(extractor, "cleanup"):
                extractor.cleanup()
            if hasattr(extractor, "__exit__"):
                extractor.__exit__(None, None, None)

        gc.collect()
        if self.spec.device.type != "cpu":
            torch.cuda.empty_cache()

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
        self.order = order or ["artwork", "learned", "vae", "vit", "residual", "wavelet"]
        self.pipeline: dict[str, Any] = {}
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the extraction pipeline based on order."""
        for module in self.order:
            if module == "artwork":
                self.pipeline["artwork"] = ArtworkExtract()
            elif module == "learned":
                self.pipeline["learned"] = LearnedExtract()
            elif module == "residual":
                self.pipeline["residual"] = Residual(self.spec)
            elif module == "wavelet":
                self.pipeline["wavelet"] = WaveletContext(self.spec, verbose=False)
            elif module == "vae":
                self.pipeline["vae"] = VAEExtract(self.spec, verbose=False)
            elif module == "vit":
                self.pipeline["vit"] = VITExtract(self.spec, verbose=False)

    def run(self, image: Image.Image | Tensor) -> dict[str, float]:
        """Run the pipeline on a single image.\n
        :param image: Input PIL image or tensor.
        :returns: Dictionary with combined features from pipeline.
        """
        results: dict[str, float] = {}

        for module in self.order:
            if module == "artwork":
                results.update(self.pipeline["artwork"](image))
            elif module == "learned":
                results.update(self.pipeline["learned"](image))
            elif module == "residual":
                from skimage.color import rgb2gray

                numeric = np.asarray(image)
                if numeric.ndim == 3:
                    numeric = np.moveaxis(numeric, 0, -1)
                gray = rgb2gray(numeric)
                res = self.pipeline["residual"](gray)
                results.update({k: v for k, v in res.items() if isinstance(v, (int, float))})
            elif module == "wavelet":
                pass  # wavelet requires dataset, handled separately
            elif module == "vae":
                pass  # vae requires tensor, handled separately
            elif module == "vit":
                results.update(self._run_vit(image))

        return results

    def _run_vit(self, image: Image.Image) -> dict[str, float]:
        """Run VIT extraction on image."""
        vit_extractor = self.pipeline["vit"]
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
