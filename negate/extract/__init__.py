# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Extract package for feature extraction modules.

This package provides unified feature extraction with interchangeable analyzers.
Users can select which modules to run and in what order.

Modules:
    - UnifiedExtractor: Main interface for selecting and running extractors
    - ExtractorPipeline: Configurable pipeline for ordered extraction
    - ArtworkExtract: 158 handcrafted features for artwork detection
    - LearnedExtract: 768 ConvNeXt features for learned representation
    - VAEExtract: VAE latent features with drift analysis
    - VITExtract: Vision Transformer feature extraction
    - Residual: Residual image feature computations

Usage:
    >>> from negate.extract import UnifiedExtractor, create_extractor
    >>> from negate.io.spec import Spec
    >>> spec = Spec()
    >>> extractor = UnifiedExtractor(spec, enable=["artwork", "learned", "vae"])
    >>> features = extractor(image)
    >>> # or use factory function
    >>> extractor = create_extractor(spec, ["artwork", "learned", "vit"])
"""

from negate.extract.feature_artwork import ArtworkExtract
from negate.extract.feature_learned import LearnedExtract
from negate.extract.feature_vae import VAEExtract
from negate.extract.feature_vit import VITExtract
from negate.extract.unified import UnifiedExtractor, ExtractorPipeline, create_extractor, create_pipeline

__all__ = [
    "ArtworkExtract",
    "LearnedExtract",
    "VAEExtract",
    "VITExtract",
    "UnifiedExtractor",
    "ExtractorPipeline",
    "create_extractor",
    "create_pipeline",
]
