"""Tests for WaveletAnalyzer."""

import numpy as np
import pytest
from PIL import Image
from negate.wavelet import WaveletAnalyzer
from negate.datasets import generate_dataset

# def test_output_consistency():
#     """All output patches and crops must have identical dimensions."""
#     analyzer = WaveletAnalyzer(patch_dim=16, resize_percent=1.0)

#     # Create a 64x64 grayscale image
#     img = Image.new("L", (64, 64), 128)

#     result = analyzer._decompose_single(img)

#     shapes = {
#         "min_patch": np.array(result["min_patch"]).shape,
#         "max_patch": np.array(result["max_patch"]).shape,
#         "min_crop": np.array(result["min_crop"]).shape,
#         "max_crop": np.array(result["max_crop"]).shape,
#     }

#     unique_shapes = set(shapes.values())
#     assert len(unique_shapes) == 1, f"Inconsistent shapes: {shapes}"


# def test_output_consistency_odd_dimensions():
#     """Outputs must be consistent even with odd-dimension source images."""
#     analyzer = WaveletAnalyzer(patch_dim=16)

#     # Create a 63x65 image (both dimensions are odd)
#     img = Image.new("L", (63, 65), 200)

#     result = analyzer._decompose_single(img)

#     shapes = {
#         "min_patch": np.array(result["min_patch"]).shape,
#         "max_patch": np.array(result["max_patch"]).shape,
#         "min_crop": np.array(result["min_crop"]).shape,
#         "max_crop": np.array(result["max_crop"]).shape,
#     }

#     unique_shapes = set(shapes.values())
#     assert len(unique_shapes) == 1, f"Inconsistent shapes: {shapes}"


# def test_patch_dim_respected():
#     """Patches should match configured patch_dim when possible."""
#     analyzer = WaveletAnalyzer(patch_dim=16)

#     img = Image.new("L", (64, 64), 100)
#     from datasets import Dataset

#     dataset = Dataset.from_list([{"image": img}])
#     result = analyzer._decompose_single(dataset)

#     # Patches should be exactly patch_dim x patch_dim
#     assert np.array(result["min_patch"]).shape == (16, 16)
#     assert np.array(result["max_patch"]).shape == (16, 16)


def test_decompose_returns_consistent_features():
    """Full decompose method should produce consistent output shapes."""

    analyzer = WaveletAnalyzer(patch_dim=8)
    # Create a simple dataset
    data = [{"image": Image.new("L", (32, 32), i * 20)} for i in range(3)]
    dataset = generate_dataset(data, label=1)
    result_dataset = dataset.map(
        analyzer._process_batch,
        remove_columns=["image"],
        batched=True,
        batch_size=1,
        desc="Extracting wavelet features...",  # type: ignore
    )
    result_dataset = analyzer.decompose(dataset)
    result_dataset.set_format(type="numpy", columns=["features", "label"])
    for item in result_dataset:
        print(item)
        features = item["features"]
        shapes = {
            "min_patch": np.array(features["min_patch"]).shape,
            "max_patch": np.array(features["max_patch"]).shape,
            "min_crop": np.array(features["min_crop"]).shape,
            "max_crop": np.array(features["max_crop"]).shape,
        }
        unique_shapes = set(shapes.values())
        assert len(unique_shapes) == 1, f"Inconsistent shapes: {shapes}"
