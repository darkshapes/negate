# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Feature extraction modules."""

from .combination import run_all_combinations
from .unified_core import ExtractionModule, DEFAULT_ENABLED_MODULES, UnifiedExtractor
from .unified_pipeline import ExtractorPipeline, create_extractor, create_pipeline

from .feature_conv import LearnedExtract
from .feature_vae import VAEExtract
from .feature_vit import VITExtract

from negate.decompose.complex import ComplexFeatures
from negate.decompose.edge import EdgeFeatures
from negate.decompose.enhanced import EnhancedFeatures
from negate.decompose.hog import HOGFeatures
from negate.decompose.linework import LineworkFeatures
from negate.decompose.numeric import NumericImage
from negate.decompose.patch import PatchFeatures
from negate.decompose.surface import SurfaceFeatures
from negate.decompose.wavelet import WaveletAnalyze, WaveletContext

__all__ = [
    "ComplexFeatures",
    "EdgeFeatures",
    "EnhancedFeatures",
    "HOGFeatures",
    "LineworkFeatures",
    "NumericImage",
    "PatchFeatures",
    "SurfaceFeatures",
    "ExtractionModule",
    "ExtractorPipeline",
    "UnifiedExtractor",
    "create_extractor",
    "create_pipeline",
    "DEFAULT_ENABLED_MODULES",
    "run_all_combinations",
]
