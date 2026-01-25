# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import asyncio
from pathlib import Path
from negate.extract import ResidualExtractor


def try_extract_synthetic():
    residual_extractor = ResidualExtractor(input=Path("assets/synthetic_v2"), verbose=True)

    async def async_main() -> None:
        residuals = await residual_extractor.process_residuals()
        return residuals

    residuals = asyncio.run(async_main())
    fractal_stats = residuals["fractal_complexity"]
    texture_stats = residuals["texture_complexity"]
    return fractal_stats, texture_stats


def try_extract_human():
    residual_extractor = ResidualExtractor(input=Path("assets/real"), verbose=True)

    async def async_main() -> None:
        residuals = await residual_extractor.process_residuals()
        return residuals

    residuals = asyncio.run(async_main())

    fractal_stats = residuals["fractal_complexity"]
    texture_stats = residuals["texture_complexity"]
    return fractal_stats, texture_stats


if __name__ == "__main__":
    synthetic_stats = try_extract_synthetic()
    human_origin_stats = try_extract_human()
    from negate.quantify import graph_comparison

    graph_comparison(synthetic_stats, human_origin_stats)
