# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clase-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Haar Wavelet processing"""

import torch
from numpy.typing import NDArray
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor

from negate import chip, negate_opt


class WaveletAnalyzer:
    """Extract wavelet energy features from images.

    Attributes:
        patch_dim: Size of square cells for analysis.
        batch_size: Batch size for dataset processing (0 = no batching).
        alpha: Perturbation weight for HF(x) subtraction (0 < α < 1).

    Example:

    """

    def __init__(self) -> None:
        """Initialize analyzer with configuration.\n
        :param dim_patch: Dimension of square cells (default 224).
        :param resize_percent: Resize factor before celling (default 1.0, no resize).
        :param batch_size: Batch size for processing (0 disables batching).
        :param alpha: Perturbation weight (0 < α < 1) for HF(x) subtraction.
        """

        self.batch_size = negate_opt.batch_size
        self.alpha = negate_opt.alpha
        dim_patch = negate_opt.dim_patch
        self.dim_patch = (dim_patch, dim_patch)
        self.dim_rescale = negate_opt.dim_factor * dim_patch
        self.dwt = DWTForward(J=2, wave="haar").to(**self.cast_move)
        self.idwt = DWTInverse(wave="haar").to(**self.cast_move)
        self.cast_move = {"device": chip.device, "dtype": chip.dtype}

    @torch.inference_mode()
    def _find_extrema(self, images: Image.Image | list[Image.Image]) -> tuple[list[NDArray], list[NDArray]]:
        """Find min/max energy cells.\n
        :param orig: Original numeric image (unused but kept for signature compatibility).
        :param cells: Dict of center->cell mappings.\n
        :returns: Tuple of (min_cells list, max_cells list) with centers and data.
        """

        from negate.scaling import patchify_image, tensor_rescale

        tensor_rescale(images, self.dim_rescale, **self.cast_move)

        min_perts = []
        max_perts = []
        tensor_patches: list[Tensor] = []
        for image in tensor_patches:
            tensor_patches.append(patchify_image(img=image, patch_size=self.dim_patch, stride=self.dim_patch))

            yl, yh = self.dwt(tensor_patches)
            pert_hf = self.idwt((torch.zeros_like(yl), yh))
            perturbed_patches = tensor_patches - self.alpha * pert_hf

            min_perts.append(min(perturbed_patches))
            max_perts.append(perturbed_patches)

        return min_perts, max_perts
