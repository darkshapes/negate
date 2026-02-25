# SPDX-License-Identifier: MPL-2.0 And LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Haar Wavelet processing adapted from sungikchoi/WaRPAD/ and mever-team/spai"""

from __future__ import annotations

import gc
from typing import Any, ContextManager

import numpy as np
import torch
from datasets import Dataset
from pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor
from torch.nn.functional import cosine_similarity

from negate.decompose.residuals import Residual
from negate.decompose.scaling import patchify_image, tensor_rescale
from negate.extract.feature_vae import VAEExtract
from negate.extract.feature_vit import VITExtract
from negate.io.config import Spec
from negate.io.save import save_features

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
        extract: VITExtract | None = None,
        vae: VAEExtract | None = None,
        residual: Residual | None = None,
    ):
        self.spec = spec
        self.dwt = dwt or DWTForward(J=2, wave="haar")
        self.idwt = idwt or DWTInverse(wave="haar")
        self.extract = extract or VITExtract(spec)  # type: ignore
        self.vae = vae or VAEExtract(spec)
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
        print("Initializing Analyzer...")
        spec = context.spec
        self.cast_move: dict = spec.apply
        self.dwt = context.dwt.to(**self.cast_move)
        self.idwt = context.idwt.to(**self.cast_move)
        self.extract = context.extract
        self.residual = context.residual
        self.vae = context.vae
        self.batch_size = spec.opt.batch_size
        self.alpha = spec.opt.alpha
        dim_patch = spec.opt.dim_patch
        self.dim_patch = (dim_patch, dim_patch)
        self.dim_factor = spec.opt.dim_factor
        print("Initializing device...")
        self.device = spec.device
        self.np_dtype = spec.np_dtype
        self.magnitude_sampling = spec.opt.magnitude_sampling
        self.top_k = spec.opt.top_k
        print("Please wait...")

    @torch.inference_mode()
    def __call__(self, dataset: Dataset) -> dict[str, Any]:
        """Forward passes any resolution images and exports their normal and perturbed feature similarity.\n
        The batch size of the tensors in the `x` list should be equal to 1, i.e. each
        tensor in the list should correspond to a single image.
        :param dataset: dataset with key "image", a `list` of 1 x C x H_i x W_i tensors, where i denotes the i-th image in the list
        :returns: A dict of processed fourier residual, wavelet and rrc data"""

        images = dataset["image"]
        results: list[dict[str, Any]] = []

        scale = self.dim_factor * self.dim_patch[0]
        rescaled = tensor_rescale(images, scale, **self.cast_move)

        for img in rescaled:
            vae_results = {}
            patched: Tensor = patchify_image(img, patch_size=self.dim_patch, stride=self.dim_patch)  # b x L_i x C x H x W
            selected, fourier_max = self.select_patch(patched)

            low_residual, high_coefficient = self.dwt(selected)  # more or less verbatim from sungikchoi/WaRPAD
            perturbed_high_freq = self.idwt((torch.zeros_like(low_residual), high_coefficient))
            perturbed_selected = selected - self.alpha * perturbed_high_freq
            base_features: Tensor | list[Tensor] = self.extract(selected)
            warp_features: Tensor | list[Tensor] = self.extract(perturbed_selected)
            sim_extrema = self.sim_extrema(base_features, warp_features, selected.shape[0])

            residuals = self.residual(selected)

            if self.vae.model is not None:
                latent_drift = self.vae(selected)
                latent_drift = self.vae(selected)
                perturbed_drift = {f"perturbed_{k}": v for k, v in self.vae.latent_drift(perturbed_selected).items()}
                perturbed_drift = {f"perturbed_{k}": v for k, v in self.vae.latent_drift(perturbed_selected).items()}
                vae_results = latent_drift | perturbed_drift

            results.append({"scale": scale} | fourier_max | sim_extrema | residuals | vae_results)

        return {"results": results}

    @torch.inference_mode()
    def select_patch(self, img: Tensor, selected_only: bool = False) -> tuple[Tensor, dict[str, float | Tensor | list[dict[str, list[float]]]]]:
        patched: Tensor = patchify_image(img, patch_size=self.dim_patch, stride=self.dim_patch)  # b x L_i x C x H x

        max_magnitudes: list = []
        metrics: list = []
        discrepancy: dict[str, float | list[float]] = {}
        chosen: list[dict[str, list[float]]] = []

        for patch in patched:  # TODO: add top_k logic here
            discrepancy = self.residual.fourier_discrepancy(patch)
            max_magnitudes.append(discrepancy["max_magnitude"])
            metrics.extend(discrepancy["mag_spectrum"])
        metrics.sort(key=lambda x: x[0], reverse=True)
        chosen.extend([p for _, p in metrics[: self.top_k]])
        chosen.extend([p for _, p in metrics[-self.top_k :]])

        max_idx = int(np.argmax(max_magnitudes))
        selected: Tensor = patched[[max_idx]]
        return selected, {
            "selected_patch_idx": float(max_idx),
            "max_fourier_magnitude": float(max_magnitudes[max_idx]),
            "mag_spectrum": chosen,
        }

    @torch.inference_mode()
    def sim_extrema(self, base_features: Tensor | list[Tensor], warp_features: Tensor | list[Tensor], batch: int) -> dict[str, float]:
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


def wavelet_preprocessing(dataset: Dataset, spec: Spec) -> Dataset:
    """Apply wavelet analysis transformations to dataset.\n
    :param dataset: HuggingFace Dataset with 'image' column.
    :param spec: Specification container with analysis configuration.
    :return: Transformed dataset with 'features' column."""
    print("Beginning preprocessing.")
    kwargs = {}
    if spec.opt.batch_size > 0:
        kwargs["batched"] = True
        kwargs["batch_size"] = spec.opt.batch_size

    context = WaveletContext(spec)
    with WaveletAnalyze(context) as analyzer:  # type: ignore
        dataset = dataset.map(
            analyzer,
            remove_columns=["image"],
            desc="Computing wavelets...",
            **kwargs,
        )
    save_features(dataset)
    return dataset
