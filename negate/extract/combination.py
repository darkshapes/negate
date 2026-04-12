# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Run all possible combinations of extraction modules on an image."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from PIL import Image

from negate.extract.unified_core import ExtractionModule, UnifiedExtractor
from negate.io.spec import Spec


def run_all_combinations(image_path: Path | str) -> dict[str, Any]:
    """Run all possible combinations of extraction modules on an image."""
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    spec = Spec()
    all_modules = list(ExtractionModule)

    results: dict[str, Any] = {
        "single_modules": {},
        "module_pairs": {},
        "summary": {},
    }

    single_results: dict[str, int] = {}
    pair_results: dict[str, int] = {}

    all_extractors = []

    for module in all_modules:
        try:
            extractor = UnifiedExtractor(spec, enable=[module])
            all_extractors.append(extractor)
            features = extractor(image)
            results["single_modules"][module.name] = features
            single_results[module.name] = len(features)
        except RuntimeError:
            results["single_modules"][module.name] = {}
            single_results[module.name] = 0

    for mod1, mod2 in itertools.combinations(all_modules, 2):
        pair_name = f"{mod1.name}+{mod2.name}"
        try:
            extractor = UnifiedExtractor(spec, enable=[mod1, mod2])
            all_extractors.append(extractor)
            features = extractor(image)
            results["module_pairs"][pair_name] = features
            pair_results[pair_name] = len(features)
        except RuntimeError:
            results["module_pairs"][pair_name] = {}
            pair_results[pair_name] = 0

    for extractor in all_extractors:
        extractor.cleanup()

    results["summary"] = {
        "total_single_modules": len(single_results),
        "total_module_pairs": len(pair_results),
        "single_module_feature_counts": single_results,
        "pair_feature_counts": pair_results,
    }

    return results
