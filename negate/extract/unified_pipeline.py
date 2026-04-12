# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Pipeline orchestration for unified extraction."""

from __future__ import annotations

import gc
from typing import Any

from PIL import Image
from torch import Tensor

from negate.extract.unified_core import DEFAULT_ENABLED_MODULES, ExtractionModule
from negate.io.spec import Spec


class ExtractorPipeline:
    """Pipeline for running extractors in configurable order."""

    def __init__(self, spec: Spec, order: list[str] | None = None) -> None:
        """Initialize pipeline with specified order.

        :param spec: Specification container with model config and hardware settings.
        :param order: List of module names in execution order.
        """
        self.spec = spec
        self.order = order or list(DEFAULT_ENABLED_MODULES)
        self.pipeline: dict[str, Any] = {}
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the extraction pipeline based on order."""
        from negate.decompose.surface import SurfaceFeatures as ArtworkExtract
        from negate.decompose.complex import ComplexFeatures
        from negate.decompose.edge import EdgeFeatures
        from negate.decompose.enhanced import EnhancedFeatures
        from negate.decompose.hog import HOGFeatures
        from negate.decompose.linework import LineworkFeatures
        from negate.decompose.numeric import NumericImage
        from negate.decompose.patch import PatchFeatures
        from negate.decompose.wavelet import WaveletAnalyze, WaveletContext

        for module in self.order:
            match module:
                case ExtractionModule.ARTWORK:
                    self.pipeline[ExtractionModule.ARTWORK] = ArtworkExtract(NumericImage(Image.new("RGB", (255, 255))))
                case ExtractionModule.LEARNED:
                    self.pipeline[ExtractionModule.LEARNED] = ComplexFeatures()
                case ExtractionModule.RESIDUAL:
                    self.pipeline[ExtractionModule.RESIDUAL] = EdgeFeatures()
                case ExtractionModule.WAVELET:
                    self.pipeline[ExtractionModule.WAVELET] = EnhancedFeatures()
                case ExtractionModule.VAE:
                    self.pipeline[ExtractionModule.VAE] = HOGFeatures()
                case ExtractionModule.VIT:
                    self.pipeline[ExtractionModule.VIT] = LineworkFeatures()

    def run(self, image: Image.Image | Tensor) -> dict[str, float]:
        """Run the pipeline on a single image.

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
        """Run VIT extraction on image.

        :param image: Input PIL image.
        :returns: Dictionary of VIT features.
        """
        vit_extractor = self.pipeline[ExtractionModule.VIT]
        try:
            image_features = vit_extractor(image)
            if isinstance(image_features, list) and len(image_features) > 0:
                feat = image_features[0]
                if isinstance(feat, Tensor):
                    return {"vit_features_mean": float(feat.mean()), "vit_features_std": float(feat.std())}
        except RuntimeError:
            pass
        return {}

    def cleanup(self) -> None:
        """Clean up all resources in pipeline."""
        for extractor in self.pipeline.values():
            if hasattr(extractor, "cleanup"):
                extractor.cleanup()
        gc.collect()


def create_extractor(spec: Spec, modules: list[str]) -> UnifiedExtractor:
    """Factory function to create a unified extractor with specified modules.

    :param spec: Specification container with model config and hardware settings.
    :param modules: List of module names to enable.
    :returns: UnifiedExtractor instance.
    """
    from negate.extract.unified_core import UnifiedExtractor

    return UnifiedExtractor(spec, enable=modules)


def create_pipeline(spec: Spec, order: list[str]) -> ExtractorPipeline:
    """Factory function to create a pipeline with specified order.

    :param spec: Specification container with model config and hardware settings.
    :param order: List of module names in execution order.
    :returns: ExtractorPipeline instance.
    """
    from negate.extract.unified_core import ExtractionModule, UnifiedExtractor

    return ExtractorPipeline(spec, order=order)
