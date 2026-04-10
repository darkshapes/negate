# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Feature extraction modules."""

from .combination import run_all_combinations
from .unified import (
    ComplexFeatures,
    EdgeFeatures,
    EnhancedFeatures,
    HOGFeatures,
    LineworkFeatures,
    ExtractionModule,
    ExtractorPipeline,
    NumericImage,
    PatchFeatures,
    SurfaceFeatures,
    UnifiedExtractor,
    create_extractor,
    create_pipeline,
)

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
    "run_all_combinations",
]
