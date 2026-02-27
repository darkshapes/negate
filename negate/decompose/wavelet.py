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
from negate.decompose.scaling import condense_tensors, patchify_image, tensor_rescale
from negate.extract.feature_vae import VAEExtract
from negate.extract.feature_vit import VITExtract
from negate.io.spec import Spec

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
        inference: bool,
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
        if inference:
            self.vae.next_model(1)
        self.residual = residual or Residual(spec)
        self.inference = inference

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
        self.context = context
        self.cast_move: dict = self.context.spec.apply
        self.context.dwt = self.context.dwt.to(**self.cast_move)
        self.context.idwt = self.context.idwt.to(**self.cast_move)
        self.dim_patch = (self.context.spec.opt.dim_patch, self.context.spec.opt.dim_patch)
        print("Initializing device...")
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

        scale = self.context.spec.opt.dim_factor * self.dim_patch[0]
        rescaled = tensor_rescale(images, scale, **self.cast_move)

        for img in rescaled:
            patched: Tensor = patchify_image(img, patch_size=self.dim_patch, stride=self.dim_patch)  # b x L_i x C x H x W
            selected, fourier_max, patch_spectrum = self.select_patch(patched)

            decomposed_feat = {}

            vae_feat = self.context.vae(patch_spectrum)
            condensed_feat = {"features_dc": condense_tensors(vae_feat["features"], self.context.spec.opt.condense_factor, self.context.spec.opt.top_k)}

            decomposed_feat: dict[str, float | tuple[int, int]] = self.ensemble_decompose(selected)

            results.append(decomposed_feat | condensed_feat | fourier_max)

        return {"results": results}

    @torch.inference_mode()
    def ensemble_decompose(self, tensor: Tensor) -> dict[str, float | tuple[int, int]]:
        """Process tensors using multiple fourier decomposition and analysis methods (Haar, Laplace, Sobel, Spectral, Residual processing,etc )
        :param selected: Patched tensor to analyze
        :returns: A dictionary of measurements"""
        low_residual, high_coefficient = self.context.dwt(tensor)  # more or less verbatim from sungikchoi/WaRPAD
        perturbed_high_freq = self.context.idwt((torch.zeros_like(low_residual), high_coefficient))
        perturbed_selected = tensor - self.context.spec.opt.alpha * perturbed_high_freq
        base_features: Tensor | list[Tensor] = self.context.extract(tensor)
        warp_features: Tensor | list[Tensor] = self.context.extract(perturbed_selected)

        sim_extrema = self.sim_extrema(base_features, warp_features, tensor.shape[0])
        residuals = self.context.residual(tensor)
        latent_drift = self.context.vae.latent_drift(tensor)
        perturbed_drift = {f"perturbed_{k}": v for k, v in self.context.vae.latent_drift(perturbed_selected).items()}
        return sim_extrema | residuals | latent_drift | perturbed_drift

    @torch.inference_mode()
    def select_patch(self, img: Tensor) -> tuple[Tensor, dict[str, float | int | Tensor | list[float]], list[Tensor]]:
        """Select highest Fourier magnitude patches from image.\n
        :param img: Input tensor image to patchify.
        :returns: Tuple of (selected patch tensor, metadata dict, spectrum patches).
        """
        patched: Tensor = patchify_image(img, patch_size=self.dim_patch, stride=self.dim_patch)

        max_magnitudes: list[float] = []  # fixed type hint
        discrepancy: dict[str, float] = {}

        for patch in patched:
            discrepancy = self.context.residual.fourier_discrepancy(patch)
            max_magnitudes.append(discrepancy["max_magnitude"])

        mag_array = np.array(max_magnitudes)
        k = min(self.context.spec.opt.top_k, len(mag_array))
        if k == 0:
            raise RuntimeError("No patches found for Fourier analysis.")
        assert self.context.spec.opt.top_k >= 1, ValueError("top_k must be ≥ 1 for Fourier patch selection.")
        top_k_idx = np.argpartition(mag_array, -k)[-k:]

        max_mag_idx = int(top_k_idx[np.argmax(mag_array[top_k_idx])])
        selected: Tensor = patched[[max_mag_idx]]
        max_fourier = float(max_magnitudes[max_mag_idx])

        patch_spectrum = [patched[i] for i in top_k_idx if i != max_mag_idx]
        if not patch_spectrum:
            print("Empty fourier magnitude spectrum: falling back to max magnitude patch.")
            patch_spectrum = [selected]

        return (
            selected,
            {
                "selected_patch_idx": max_mag_idx,
                "max_fourier_magnitude": max_fourier,
            },
            patch_spectrum,
        )

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

        device_name = self.context.spec.device.type
        del self.context.spec.device
        if device_name != "cpu":
            self.gpu = getattr(torch, device_name)
            self.gpu.empty_cache()  # type: ignore
        gc.collect()

    def __enter__(self) -> "WaveletAnalyze":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "extract"):
            self.cleanup()
