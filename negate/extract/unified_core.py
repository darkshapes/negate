# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Core unified feature extraction components."""

from __future__ import annotations

import gc
from enum import Enum, auto
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

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
    """Unified feature extraction interface with interchangeable analyzers."""

    def __init__(self, spec: Spec, enable: Sequence[ExtractionModule | str] | None = None) -> None:
        """Initialize the unified extractor with selected modules.

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
        from negate.decompose.surface import SurfaceFeatures as ArtworkExtract
        from negate.decompose.complex import ComplexFeatures
        from negate.decompose.edge import EdgeFeatures
        from negate.decompose.enhanced import EnhancedFeatures
        from negate.decompose.hog import HOGFeatures
        from negate.decompose.linework import LineworkFeatures
        from negate.decompose.numeric import NumericImage
        from negate.decompose.patch import PatchFeatures
        from negate.decompose.wavelet import WaveletAnalyze, WaveletContext
        from .feature_conv import LearnedExtract
        from .feature_vae import VAEExtract
        from .feature_vit import VITExtract
        from negate.decompose.residuals import Residual

        for module in self.enabled:
            match module:
                case ExtractionModule.ARTWORK:
                    dummy_image = Image.new("RGB", (255, 255))
                    self.extractors[ExtractionModule.ARTWORK] = ArtworkExtract(NumericImage(dummy_image))
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
        """Extract features from a single image using enabled modules.

        :param image: Input PIL image or tensor.
        :returns: Dictionary with combined features from all enabled modules.
        """
        results: dict[str, float] = {}

        if ExtractionModule.ARTWORK in self.enabled:
            results.update(self.extractors[ExtractionModule.ARTWORK]())
        if ExtractionModule.LEARNED in self.enabled:
            results.update(self.extractors[ExtractionModule.LEARNED](image))
        if ExtractionModule.RESIDUAL in self.enabled:
            results.update({k: v for k, v in self.extractors[ExtractionModule.RESIDUAL](image).items() if isinstance(v, (int, float))})
        if ExtractionModule.WAVELET in self.enabled:
            results.update(self._extract_wavelet(image))
        if ExtractionModule.VAE in self.enabled:
            results.update(self._extract_vae(image))
        if ExtractionModule.VIT in self.enabled:
            results.update(self._extract_vit(image))

        return results

    def extract_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        """Extract features from a batch of images.

        :param images: List of PIL images.
        :returns: List of feature dictionaries, one per image.
        """
        return [self(image) for image in images]

    def _extract_wavelet(self, image: Image.Image) -> dict[str, float]:
        """Extract wavelet features using WaveletContext.

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
        except ImportError:
            return {}

    def _extract_vae(self, image: Image.Image) -> dict[str, float]:
        """Extract VAE features.

        :param image: Input PIL image.
        :returns: Dictionary of VAE features.
        """
        import torchvision.transforms as T

        vae_extractor = self.extractors[ExtractionModule.VAE]
        transform = T.Compose([T.CenterCrop((512, 512)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

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
        except RuntimeError:
            return {}

    def _extract_vit(self, image: Image.Image) -> dict[str, float]:
        """Extract VIT features.

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
        except RuntimeError:
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
                except RuntimeError:
                    pass
            if hasattr(extractor, "__exit__"):
                extractor.__exit__(None, None, None)

        gc.collect()
        try:
            if self.spec.device.type != "cpu":
                torch.cuda.empty_cache()
        except RuntimeError:
            pass

    def __enter__(self) -> "UnifiedExtractor":
        """Return self as context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit context and cleanup resources."""
        self.cleanup()
