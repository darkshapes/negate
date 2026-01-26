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
    return residual_extractor


def try_extract_human():
    residual_extractor = ResidualExtractor(input=Path("assets/real"), verbose=True)

    async def async_main() -> None:
        residuals = await residual_extractor.process_residuals()
        return residuals

    residuals = asyncio.run(async_main())

    return residual_extractor


if __name__ == "__main__":
    from negate.quantify import graph_comparison

    synthetic_stats = try_extract_synthetic()
    human_origin_stats = try_extract_human()

    graph_comparison(synthetic_stats.data_frame, human_origin_stats.data_frame)
