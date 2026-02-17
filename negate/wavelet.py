# SPDX-License-Identifier: MPL-2.0 And LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Haar Wavelet processing adapted from sungikchoi/WaRPAD/ and mever-team/spai"""

from __future__ import annotations

import gc
from typing import ContextManager, TypedDict, Any

import numpy as np
import torch
from datasets import Dataset
from pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor
from torch.nn.functional import cosine_similarity

from negate.config import Spec
from negate.feature_vae import VAEExtract
from negate.feature_vit import VITExtract
from negate.residuals import Residual
from negate.scaling import patchify_image, tensor_rescale

"""Haar Wavelet processing"""


class WaveletContext:
    """Container for wavelet analysis dependencies."""

    spec: Spec
    dwt: DWTForward
    idwt: DWTInverse
    extract: VITExtract
    residual: Residual

    def __init__(
        self,
        spec: Spec,
        dwt: DWTForward | None = None,
        idwt: DWTInverse | None = None,
        extract: VITExtract | VAEExtract | None = None,
        residual: Residual | None = None,
    ):
        self.spec = spec
        self.dwt = dwt or DWTForward(J=2, wave="haar")
        self.idwt = idwt or DWTInverse(wave="haar")
        self.extract = extract or VITExtract(spec)  # type: ignore
        self.residual = residual or Residual(spec)

    def __enter__(self) -> WaveletContext:
        return self

    def __exit__(self, *args: object) -> None:
        pass  # Cleanup if needed.


class WaveletAnalyze(ContextManager):
    """Analyze images using wavelet transform."""

    context: WaveletContext

    def __init__(self, context: WaveletContext) -> None:
        """Extract wavelet energy features from images."""
        spec = context.spec
        self.batch_size = spec.opt.batch_size
        self.alpha = spec.opt.alpha
        dim_patch = spec.opt.dim_patch
        self.dim_patch = (dim_patch, dim_patch)
        self.dim_rescale = spec.opt.dim_factor * dim_patch
        self.device = spec.device
        self.np_dtype = spec.np_dtype
        self.cast_move: dict = spec.apply
        self.dwt = context.dwt.to(**self.cast_move)
        self.idwt = context.idwt.to(**self.cast_move)
        self.extract = context.extract
        self.residual = context.residual

    @torch.inference_mode()
    def __call__(self, dataset: Dataset, extrema=False) -> dict[str, dict[str, Any]]:
        """Forward passes any resolution images and exports their normal and perturbed feature similarity.\n
        The batch size of the tensors in the `x` list should be equal to 1, i.e. each
        tensor in the list should correspond to a single image.
        :param dataset: dataset with key "image", a `list` of 1 x C x H_i x W_i tensors, where i denotes the i-th image in the list
        :returns: A tuple containing"""

        results: dict[str, dict[str, float | list[float]]] = {}

        images = dataset["image"]
        rescaled = tensor_rescale(images, self.dim_rescale, **self.cast_move)

        for idx, img in enumerate(rescaled):
            patched: Tensor = patchify_image(img, patch_size=self.dim_patch, stride=self.dim_patch)  # b x L_i x C x H x W
            max_magnitudes: list = []
            discrepancy: dict[str, float | list[float]] = {}
            for patch in patched:
                discrepancy = self.residual.fourier_discrepancy(patch)
                max_magnitudes.append(discrepancy["max_magnitude"])

            if not max_magnitudes:
                continue

            max_idx = int(np.argmax(max_magnitudes))
            if extrema:
                selected = patched[[max_idx]]
            else:
                selected = patched

            low_residual, high_coefficient = self.dwt(selected)  # more or less verbatim from sungikchoi/WaRPAD
            perturbed_high_freq = self.idwt((torch.zeros_like(low_residual), high_coefficient))
            perturbed_selected = selected - self.alpha * perturbed_high_freq
            base_features: Tensor | list[Tensor] = self.extract(selected)
            warp_features: Tensor | list[Tensor] = self.extract(perturbed_selected)
            results[str(idx)] = (
                {
                    "selected_patch_idx": float(max_idx),
                    "max_fourier_magnitude": float(max_magnitudes[max_idx]),
                    "all_magnitudes": max_magnitudes,
                }
                | discrepancy
                | self.residual(selected)
                | self.shape_extrema(base_features, warp_features, selected.shape[0])
            )

        return {"results": results}

    @torch.inference_mode()
    def shape_extrema(self, base_features: Tensor | list[Tensor], warp_features: Tensor | list[Tensor], batch: int) -> dict[str, float]:
        """Compute minimum and maximum cosine similarity between base and warped features.\n
        :param base_features: Raw feature tensors from original patches.
        :param warp_features: Warped feature tensors after wavelet perturbation.
        :param batch: Number of images in current processing tensor.
        :returns: Dictionary with min/max similarity arrays."""

        min_warps = []
        max_warps = []
        min_base = []
        max_base = []

        for idx, tensor in enumerate(base_features):  # also from sungikchoi/WaRPAD/
            if idx >= len(warp_features):
                raise IndexError("Warped feature stack is shorter than base feature stack (should be 1:1)")
            similarity = cosine_similarity(tensor, warp_features[idx], dim=-1)
            reshaped_similarity = similarity.unsqueeze(1).reshape([batch, -1])

            similarity_min = torch.mean(reshaped_similarity, 1).view([batch])
            base_min = torch.argmin(reshaped_similarity, 1).view(batch)
            similarity_max = reshaped_similarity.view([-1])
            base_max = torch.argmax(reshaped_similarity, 1).view(batch)

            min_warps.append(np.atleast_2d(similarity_min.cpu().numpy()))
            max_warps.append(np.atleast_2d(similarity_max.cpu().numpy()))
            min_base.append(np.atleast_2d(base_min.cpu().numpy()))
            max_base.append(np.atleast_2d(base_max.cpu().numpy()))

        if not min_warps:
            return {"min_warp": 0.0, "max_warp": 0.0, "min_base": 0.0, "max_base": 0.0}

        min_warps_val = float(np.concatenate(min_warps, axis=None).flatten().mean())
        max_warps_val = float(np.concatenate(max_warps, axis=None).flatten().mean())
        min_base_val = float(np.concatenate(min_base, axis=None).flatten().mean())
        max_base_val = float(np.concatenate(max_base, axis=None).flatten().mean())

        return {
            "min_warp": min_warps_val,
            "max_warp": max_warps_val,
            "min_base": min_base_val,
            "max_base": max_base_val,
        }

    def cleanup(self) -> None:
        """Free resources once discarded."""

        device_name = self.device.type
        del self.device
        if device_name != "cpu":
            self.gpu = getattr(torch, device_name)
            self.gpu.empty_cache()  # type: ignore
        gc.collect()

    def __enter__(self) -> "WaveletAnalyze":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "extract"):
            self.cleanup()
