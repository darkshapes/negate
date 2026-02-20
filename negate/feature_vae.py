# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

import gc
import os

import torch
from torch import Tensor
from torch.nn import KLDivLoss, BCELoss, MSELoss, L1Loss
from torch.nn import functional as F
from negate.config import Spec


class VAEExtract:
    def __init__(self, spec: Spec) -> None:
        """Set up the extractor with a VAE model.\n
        :param vae_type: VAEModel ID of the VAE.
        :param device: Target device.
        :param dtype: Data type for tensors."""
        print("Initializing VAE...")

        self.spec = spec

        self.device = spec.device
        self.dtype = spec.dtype
        self.model, self.library = spec.model_config.auto_vae
        if not hasattr(self, "vae") and self.model != "None":
            self.create_vae()
        self.kl_div = KLDivLoss(log_target=True)
        self.bce_loss = BCELoss()
        self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()

    def create_vae(self):
        """Download and load the VAE from the model repo."""

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            from diffusers.models import autoencoders  # type: ignore
        except (ImportError, ModuleNotFoundError, Exception):
            raise RuntimeError("missing dependency 'diffusers'. Please install it using 'negate[extractor]'")

        if getattr(self.spec.opt, "vae_tiling", False):
            self.vae.enable_tiling()
        if getattr(self.spec.opt, "vae_slicing", False):
            self.vae.enable_slicing()

        autoencoder_cls = getattr(autoencoders, self.library.split(".")[-1], None)  # type: ignore
        try:
            vae_model = autoencoder_cls.from_pretrained(self.model.enum.value, torch_dtype=self.dtype, local_files_only=True).to(self.device)  # type: ignore
        except (LocalEntryNotFoundError, OSError, AttributeError):
            print("Downloading model...")
        vae_path: str = snapshot_download(self.model, allow_patterns=["vae/*"])  # type: ignore
        vae_path = os.path.join(vae_path, "vae")
        vae_model = autoencoder_cls.from_pretrained(vae_path, torch_dtype=self.dtype, local_files_only=True).to(self.device)  # type: ignore

        vae_model.eval()
        self.vae = vae_model

    def _extract_special(self, batch):
        """Handle SANA and AuraEqui models.\n
        :param batch: Tensor of image + patches.
        :return: NumPy mean latent."""

        try:
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # type: ignore
        except (ImportError, ModuleNotFoundError, Exception):
            raise RuntimeError("missing dependency 'diffusers'. Please install it using 'negate[extractor]'")
        import torch

        latent: Tensor = self.vae.encode(batch).latent  # type: ignore
        mean = torch.mean(latent, dim=0).cpu().float()  # type: ignore

        logvar = torch.zeros_like(mean).cpu().float()
        params = torch.cat([mean, logvar], dim=1)
        dist = DiagonalGaussianDistribution(params)
        sample = dist.sample()
        return sample  # .mean(dim=0).cpu().float().numpy()

    @torch.inference_mode()
    def __call__(self, tensor: Tensor) -> dict[str, Tensor | list[Tensor]]:
        """Extract VAE features from a batch of images then use spectral contrast as divergence metric
        :param tensor: 4D image tensor
        :return: Dictionary with 'features' list."""
        import torch

        features_list = []

        features_latent = torch.stack([tensor]).to(self.device, self.dtype)

        with torch.no_grad():
            if "AutoencoderDC" in self.library:
                features = self._extract_special(features_latent)
            else:
                features = self.vae.encode(features_latent).latent_dist.sample()

            features_list.append(features)

        return {"features": features_list}

    @torch.inference_mode()
    def latent_drift(self, tensors: Tensor) -> dict[str, float]:
        """Compute L1/MSE/KL/BCE loss between input and VAE reconstruction.\n
        :param tensor: 4D image tensor
        """

        if "AutoencoderDC" in self.library:
            latents = self.vae.encode(tensors).latent
        else:
            latents = self.vae.encode(tensors).latent_dist.sample()  # type: ignore
        reconstructed = self.vae.decode(latents).sample  # depends on API

        l1_mean = self.l1_loss(reconstructed, tensors)
        mse_mean = self.mse_loss(reconstructed, tensors)
        bce_mean = self.bce_loss(reconstructed, tensors)

        log_input = F.log_softmax(reconstructed, dim=1)  # batch, features
        log_target = F.log_softmax(tensors, dim=1)
        kl_mean = self.kl_div(log_input, log_target)

        return {"bce_loss": float(bce_mean), "l1_mean": float(l1_mean), "mse_mean": float(mse_mean), "kl_loss": float(kl_mean)}

    def cleanup(self) -> None:
        """Free the VAE and GPU memory."""

        if self.device != "cpu":
            gpu: torch.device = self.device
            gpu.empty_cache()  # type: ignore
            del gpu
        del self.vae
        del self.device
        gc.collect()

    def __enter__(self) -> VAEExtract:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "vae"):
            self.cleanup()
