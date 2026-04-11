# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Feature extraction classes for AI-generated image detection."""

from negate.decompose.complex import ComplexFeatures
from negate.decompose.edge import EdgeFeatures
from negate.decompose.enhanced import EnhancedFeatures
from negate.decompose.hog import HOGFeatures
from negate.decompose.linework import LineworkFeatures
from negate.decompose.numeric import NumericImage
from negate.decompose.patch import PatchFeatures
from negate.decompose.surface import SurfaceFeatures
from negate.decompose.wavelet import WaveletAnalyze
from negate.decompose.wavelet import WaveletContext

__all__ = [
    "NumericImage",
    "ComplexFeatures",
    "EdgeFeatures",
    "EnhancedFeatures",
    "HOGFeatures",
    "LineworkFeatures",
    "NumericImage",
    "PatchFeatures",
    "SurfaceFeatures",
    "WaveletContext",
    "WaveletAnalyze",
]
