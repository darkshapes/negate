# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/mever-team/spai

#   FrequencyLoss	filter_image_frequencies
#   Direction	    Image → Freq → Loss	    Image ↔ Freq ↔ Image
#   Frequency role	Comparison metric	    Data to be selected/removed
#   Mask behavior	Soft weighting (0-1)    Hard binary filter
#               measure freq-domain dist    extract/remove specific frequencies
#               backpropagation gradients   analysis
#
# FFT-based image processing utilities that share foundational FFT operations but diverge in purpos
# one measures frequency-domain distance for optimization, the other extracts or removes specific frequencies for analysis.

from typing import Optional

import torch
import torch.nn as nn
from torch import fft, linalg

__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


def filter_image_frequencies(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Filters the frequencies of an image according to the provided mask.

    Dimensionalities:
        - B is the batch size.
        - C is the number of image channels.
        - H is the height of the image.
        - W is the width of the image.

    :param image: The images to filter. Dimensionality: [[B] x C] x H x W
    :param mask: The mask that indicates the frequencies in the 2D DFT spectrum to be
        filtered out. Values of 1 in the mask indicate that the corresponding frequency will
        be allowed, while values of 0 indicate that the corresponding frequency will be
        filtered out. The mask is multiplied with the center-shifted spectrum of the image.
        Thus, it should follow this format. Dimensionality: [[B] x C] x H x W.

    :return: A tuple that includes:
        - The filtered image. Dimensionality: [[B] x C] x H x W
        - The residual of the filtered image (filtered with the inverse mask).
            Dimensionality: [[B] x C] x H x W
    """
    # Compute FFT.
    image = fft.fft2(image)
    image = fft.fftshift(image)

    # Filter image. NOTE: Binary
    filtered_image: torch.Tensor = image * mask
    residual_filtered_image: torch.Tensor = image * (1 - mask)

    # Compute IFFT.
    filtered_image = fft.ifftshift(filtered_image)
    filtered_image = fft.ifft2(filtered_image).real
    residual_filtered_image = fft.ifftshift(residual_filtered_image)
    residual_filtered_image = fft.ifft2(residual_filtered_image).real  # Back to pixel space

    return filtered_image, residual_filtered_image


def generate_circular_mask(input_size: int, mask_radius_start: int, mask_radius_stop: Optional[int] = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    coordinates: torch.Tensor = generate_centered_2d_coordinates_grid(input_size, device)
    radius: torch.Tensor = linalg.vector_norm(coordinates, dim=-1)
    mask: torch.Tensor = torch.where(radius < mask_radius_start, 1, 0)
    if mask_radius_stop is not None:
        mask = torch.where(radius > mask_radius_stop, 1, mask)
    return mask


def generate_centered_2d_coordinates_grid(size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    assert size % 2 == 0, "Input size must be even."
    coords_values: torch.Tensor = torch.arange(0, size // 2, dtype=torch.float, device=device)
    coords_values = torch.cat([coords_values.flip(dims=(0,)), coords_values], dim=0)
    coordinates_x: torch.Tensor = coords_values.unsqueeze(dim=0).expand(size, -1)
    coordinates_y: torch.Tensor = torch.t(coordinates_x)
    coordinates: torch.Tensor = torch.stack([coordinates_x, coordinates_y], dim=2)
    return coordinates


class FrequencyLoss(nn.Module):
    """Frequency loss.

    Modified from:
    `<https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py>`_.

    Args:
        loss_gamma (float): the exponent to control the sharpness of the frequency distance. Defaults to 1.
        matrix_gamma (float): the scaling factor of the spectrum weight matrix for flexibility. Defaults to 1.
        patch_factor (int): the factor to crop image patches for patch-based frequency loss. Defaults to 1.
        ave_spectrum (bool): whether to use minibatch average spectrum. Defaults to False.
        with_matrix (bool): whether to use the spectrum weight matrix. Defaults to False.
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Defaults to False.
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Defaults to False.
    """

    def __init__(self, loss_gamma=1.0, matrix_gamma=1.0, patch_factor=1, ave_spectrum=False, with_matrix=False, log_matrix=False, batch_matrix=False):
        super(FrequencyLoss, self).__init__()
        self.loss_gamma = loss_gamma
        self.matrix_gamma = matrix_gamma
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.with_matrix = with_matrix
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, "Patch factor should be divisible by image height and width"
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1).float()  # NxPxCxHxW

        # perform 2D FFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm="ortho")
        # shift low frequency to the center
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        # stack the real and imaginary parts along the last dimension
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        loss = torch.sqrt(tmp[..., 0] + tmp[..., 1] + 1e-12) ** self.loss_gamma
        if self.with_matrix:
            # spectrum weight matrix
            if matrix is not None:
                # if the matrix is predefined
                weight_matrix = matrix.detach()
            else:
                # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
                matrix_tmp = (recon_freq - real_freq) ** 2
                matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.matrix_gamma

                # whether to adjust the spectrum weight matrix by logarithm
                if self.log_matrix:
                    matrix_tmp = torch.log(matrix_tmp + 1.0)

                # whether to calculate the spectrum weight matrix using batch-based statistics
                if self.batch_matrix:
                    matrix_tmp = matrix_tmp / matrix_tmp.max()
                else:
                    matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()

            assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                "The values of spectrum weight matrix should be in the range [0, 1], but got Min: %.10f Max: %.10f" % (weight_matrix.min().item(), weight_matrix.max().item())
            )
            # dynamic spectrum weighting (Hadamard product)
            loss = weight_matrix * loss
        return loss

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor with shape (N, C, H, W).
            target (torch.Tensor): Target tensor with shape (N, C, H, W).
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Defaults to None.
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix)
