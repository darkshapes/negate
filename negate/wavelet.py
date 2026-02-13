# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Haar Wavelet processing"""

import gc

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torchvision.transforms.functional import five_crop

from negate.config import Spec
from negate.feature_vit import VITExtract
from pytorch_wavelets import DWTForward, DWTInverse

from negate.scaling import patchify_image, tensor_rescale

"""Haar Wavelet processing"""


class WaveletAnalyze:
    """Extract wavelet energy features from images."""

    def __init__(self, spec: Spec, dwt: DWTForward, idwt: DWTInverse, extract: VITExtract) -> None:
        """Initialize analyzer with configuration."""

        self.batch_size = spec.opt.batch_size
        self.alpha = spec.opt.alpha
        dim_patch = spec.opt.dim_patch
        self.dim_patch = (dim_patch, dim_patch)
        self.dim_rescale = spec.opt.dim_factor * dim_patch
        self.device = spec.device
        self.np_dtype = spec.np_dtype
        self.cast_move: dict = spec.apply
        self.dwt = dwt.to(**self.cast_move)
        self.idwt = idwt.to(**self.cast_move)
        self.extract = extract

    @torch.inference_mode()
    def __call__(self, dataset: Dataset) -> dict[str, list[dict[str, np.ndarray]]]:
        """Forward passes any resolution images and exports their normal and perturbed feature similarity.\n
        The batch size of the tensors in the `x` list should be equal to 1, i.e. each
        tensor in the list should correspond to a single image.
        :param dataset: dataset with key "image", a `list` of 1 x C x H_i x W_i tensors, where i denotes the i-th image in the list
        :returns: A tuple containing"""

        minimum_patches = 0
        cos_sim: list[dict[str, np.ndarray]] = []

        rescaled = tensor_rescale(dataset["image"], self.dim_rescale, **self.cast_move)

        for img in rescaled:
            patched: torch.Tensor = patchify_image(img, patch_size=self.dim_patch, stride=self.dim_patch)  # 1 x L_i x C x H x W
            batch, _, _, _tb = patched.shape

            # Perturb each patch
            low_residual, high_coefficient = self.dwt(patched)
            perturbed_high_freq = self.idwt((torch.zeros_like(low_residual), high_coefficient))
            perturbed_patches = patched - self.alpha * perturbed_high_freq

            base_features: Tensor | list[Tensor] = self.extract(patched)
            warp_features: Tensor | list[Tensor] = self.extract(perturbed_patches)
            cos_sim.append(self.shape_extrema(base_features, warp_features, batch))

        return {"results": cos_sim}

    @torch.inference_mode()
    def shape_extrema(self, base_features: Tensor | list[Tensor], warp_features: Tensor | list[Tensor], batch: int) -> dict[str, np.ndarray]:
        """Compute minimum and maximum cosine similarity between base and warped features.\n
        Calculates per-batch cosine similarities across feature maps, identifying both the\n
        extreme values (min/max) and their corresponding indices within each batch.\n
        :param base_features: Raw feature tensors from original patches\n
        :param warp_features: Warped feature tensors after wavelet perturbation\n
        :param batch: Number of images in current processing batch\n
        :returns: Tuple of ndarrays containing (min_similarities, max_similarities, min_indices, max_indices)
        """
        min_warps = []
        max_warps = []
        min_base = []
        max_base = []

        for idx, tensor in enumerate(base_features):
            similarity = cosine_similarity(tensor, warp_features[idx], dim=-1)
            reshaped_similarity = similarity.unsqueeze(1).reshape([batch, -1])

            similarity_min = torch.mean(reshaped_similarity, 1).view([batch])
            base_min = torch.argmin(reshaped_similarity, 1).view(batch)
            similarity_max = reshaped_similarity.view([-1])
            base_max = torch.argmax(reshaped_similarity, 1).view(batch)

            min_warps.append(similarity_min.cpu().numpy())
            max_warps.append(similarity_max.cpu().numpy())
            min_base.append(base_min.cpu().numpy())
            max_base.append(base_max.cpu().numpy())

        return {
            "min_warp": np.concatenate(min_warps, dtype=self.np_dtype).flatten(),
            "max_warp": np.concatenate(max_warps, dtype=self.np_dtype).flatten(),
            "min_base": np.concatenate(min_base, dtype=self.np_dtype).flatten(),
            "max_base": np.concatenate(max_base, dtype=self.np_dtype).flatten(),
        }

    def cleanup(self) -> None:
        """Free the VAE and GPU memory."""

        device_name = self.device.type
        del self.device
        if device_name != "cpu":
            self.gpu = getattr(torch, device_name)
            self.gpu.empty_cache()  # type: ignore
        del self.dwt
        del self.idwt
        del self.extract
        gc.collect()

    def __enter__(self) -> "WaveletAnalyze":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "extract"):
            self.cleanup()
