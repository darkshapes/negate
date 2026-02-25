# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/mever-team/spai

"""Scaling utilities for image tensors."""

import torch
import torchvision.transforms as T
from PIL import Image
from PIL.Image import fromarray
from torch import Tensor
import numpy as np


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


def crop_select(image: Image.Image, size: int = 512, top_k: int = 5, mask_radius: int = 50, dtype: np.typing.DTypeLike = np.float64) -> list[Image.Image]:
    """Crop image into patches, compute freq-divergence, return most extreme patches.\n
    :param image: PIL image to process.\n
    :param size: Patch dimension.\n
    :param top_k: Number of extreme patches to return.\n
    :param mask_radius: Radius used in masked_spectral logic.\n
    :return: List of selected patch images."""
    gray = image.convert("L")
    arr = np.array(gray, dtype=dtype)

    h, w = arr.shape
    nx = (w + size - 1) // size
    ny = (h + size - 1) // size

    metrics: list[tuple[float, Image.Image]] = []

    for image_y in range(ny):
        for image_x in range(nx):
            x0 = image_x * size
            y0 = image_y * size
            patch_arr = arr[y0 : y0 + size, x0 : x0 + size]
            if patch_arr.shape != (size, size):
                pad = np.zeros((size, size), dtype=dtype)
                pad[: patch_arr.shape[0], : patch_arr.shape[1]] = patch_arr
                patch_arr = pad

            f = np.fft.fft2(patch_arr)
            f_shift = np.fft.fftshift(f)

            rows, cols = size, size
            y, x = np.ogrid[:rows, :cols]
            c = (rows // 2, cols // 2)
            distribution = np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2)
            mask = distribution < mask_radius

            low_mask = ~mask
            high_mask = mask

            # Magnitudes
            low_mag = np.abs(f_shift[low_mask])
            high_mag = np.abs(f_shift[high_mask])

            diverge_metric = abs(np.mean(high_mag) - np.mean(low_mag))

            patch_img = fromarray(np.uint8(patch_arr), mode="L").convert("RGB")
            metrics.append((diverge_metric, patch_img))

    metrics.sort(key=lambda x: x[0], reverse=True)

    chosen: list[Image.Image] = []
    chosen.extend([p for _, p in metrics[:top_k]])  # high diverges
    chosen.extend([p for _, p in metrics[-top_k:]])  # low diverges

    return chosen
