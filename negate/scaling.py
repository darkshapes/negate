# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/mever-team/spai

"""Scaling utilities for image tensors."""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor


def tensor_rescale(images: Image.Image | list[Image.Image], dim_rescale: int, device: torch.device, dtype: torch.dtype):
    """Apply ImageNet normalization to images after rescaling.\n
    images: Single PIL.Image or list of PIL.Images for processing.
    dim_rescale: Target height and width for rescaling (square output).
    device: Torch device for tensor placement (e.g., 'cpu', torch.device('cuda')).
    dtype: Torch data type for tensors (e.g., torch.float32).\n
    :returns: List of normalized tensors with shape [C, dim_rescale, dim_rescale].
    """

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform: T.Compose = T.Compose([T.Resize((dim_rescale, dim_rescale), interpolation=Image.BICUBIC), T.ToTensor(), normalize])  # type: ignore

    if isinstance(images, list):
        tensor_rescale: list[Tensor] = [transform(img).to(device=device, dtype=dtype) for img in images]  # type: ignore
    else:
        tensor_rescale: list[Tensor] = [transform(images).unsqueeze(0).to(device=device, dtype=dtype)]  # type: ignore
    return tensor_rescale


def patchify_image(img: torch.Tensor, patch_size: tuple[int, int], stride: tuple[int, int]) -> torch.Tensor:
    """Splits an input image into patches\n
    :param img: Input image of size (B, C, H, W).
    :param patch_size: (height, width) of patches.
    :param stride: Stride on (height, width) dimensions.\n
    :returns: Patchified image of size (B, L, C, patch_height, patch_width).
    """

    kh, kw = patch_size
    dh, dw = stride
    img = img.unfold(2, kh, dh).unfold(3, kw, dw)
    img = img.permute(0, 2, 3, 1, 4, 5)
    img = img.contiguous()
    img = img.view(img.size(0), -1, img.size(3), kh, kw)
    batch, l_, channel, height, width = img.shape
    img = img.view(batch * l_, channel, height, width)
    return img


def split_array(form: np.ndarray, limit: int = 2**31 - 1) -> list[np.ndarray]:
    """Yield sub-arrays of ``form`` with length ≤ ``limit``."""
    if form.size <= limit:
        return [form]
    items: list[np.ndarray] = []
    for index in range(0, form.size, limit):
        items.append(form[index : index + limit])
    return items
