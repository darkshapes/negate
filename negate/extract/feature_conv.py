# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Learned feature extraction via frozen ConvNeXt-Tiny.

Complements the 148 handcrafted features with 768 learned features from
a frozen ImageNet-pretrained ConvNeXt-Tiny model. The learned features
capture visual patterns that handcrafted features miss — particularly
artifacts from novel generator architectures.

Key properties:
    - 768-dimensional output (penultimate layer of ConvNeXt-Tiny)
    - Frozen weights — no fine-tuning, no GPU training needed
    - ~28 img/s on CPU (25x faster than handcrafted features)
    - NOT CLIP-based — no text encoder bias
    - NOT DINOv2 — ConvNeXt has different inductive biases (local + hierarchical)

Unlike CLIP (which we proved has generator bias), ConvNeXt-Tiny is purely
visual and pretrained on ImageNet classification — it has no special
relationship with any generator architecture.
"""

from __future__ import annotations

import numpy as np

import torch
from numpy.typing import NDArray
from PIL import Image


class LearnedExtract:
    """Extract 768 learned features from a frozen ConvNeXt-Tiny model.

    Usage:
        >>> extractor = LearnedExtract()
        >>> features = extractor(pil_image)  # returns dict of 768 floats
        >>> len(features)  # 768
    """

    def __init__(self):
        from timm import create_model
        from timm.data.transforms_factory import create_transform
        from timm.data.config import resolve_data_config

        self._model = create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0)
        self._model.eval()
        self._transform = create_transform(**resolve_data_config(self._model.pretrained_cfg))

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> dict[str, float]:
        """Extract 768 features from a PIL image."""
        image = image.convert("RGB")
        inp = self._transform(image).unsqueeze(0)
        feat = self._model(inp).squeeze(0).numpy()
        return {f"cnxt_{i}": float(feat[i]) for i in range(len(feat))}

    @torch.no_grad()
    def batch(self, images: list[Image.Image], batch_size: int = 32) -> NDArray:
        """Extract features from a batch of images. Returns (N, 768) array."""
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = []
            for img in batch_imgs:
                try:
                    tensors.append(self._transform(img.convert("RGB")))
                except ValueError:
                    tensors.append(torch.zeros(3, 224, 224))
            batch_tensor = torch.stack(tensors)
            feats = self._model(batch_tensor).numpy()
            all_feats.append(feats)
        return np.vstack(all_feats) if all_feats else np.empty((0, 768))

    @torch.no_grad()
    def perturb_compare(self, image: Image.Image, sigma: float = 5.0) -> dict[str, float]:
        """Compare ConvNeXt features of clean vs slightly noisy image.

        Real images change more under perturbation than AI images because
        AI images sit on the generator's learned manifold and are more
        stable to small noise. Inspired by RIGID (DINOv2 perturbation check).

        :param image: PIL Image.
        :param sigma: Gaussian noise standard deviation.
        :returns: Dictionary with perturbation comparison metrics.
        """
        image = image.convert("RGB")
        arr = np.array(image, dtype=np.float64)

        # Add small Gaussian noise
        noise = np.random.RandomState(42).normal(0, sigma, arr.shape)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_arr)

        # Extract features for both
        clean_inp = self._transform(image).unsqueeze(0)
        noisy_inp = self._transform(noisy_image).unsqueeze(0)

        clean_feat = self._model(clean_inp).squeeze(0).numpy()
        noisy_feat = self._model(noisy_inp).squeeze(0).numpy()

        # Cosine distance
        dot = np.dot(clean_feat, noisy_feat)
        norm_clean = np.linalg.norm(clean_feat)
        norm_noisy = np.linalg.norm(noisy_feat)
        cosine_sim = dot / (norm_clean * norm_noisy + 1e-10)

        # L2 distance
        l2_dist = float(np.linalg.norm(clean_feat - noisy_feat))

        # Per-dimension change statistics
        diff = np.abs(clean_feat - noisy_feat)

        return {
            "perturb_cosine_dist": float(1.0 - cosine_sim),
            "perturb_l2_dist": l2_dist,
            "perturb_max_change": float(diff.max()),
            "perturb_mean_change": float(diff.mean()),
        }

    def feature_names(self) -> list[str]:
        return [f"cnxt_{i}" for i in range(768)]

    def perturb_feature_names(self) -> list[str]:
        return ["perturb_cosine_dist", "perturb_l2_dist", "perturb_max_change", "perturb_mean_change"]
