# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

import gc
import os

import torch
from torch import Tensor
from torch.nn import KLDivLoss, MSELoss, L1Loss
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

        latent: Tensor = self.vae.encode(batch)  # type: ignore
        mean = torch.mean(latent.latent, dim=0).cpu().float()  # type: ignore

        logvar = torch.zeros_like(mean).cpu().float()
        params = torch.cat([mean, logvar], dim=1)
        dist = DiagonalGaussianDistribution(params)
        sample = dist.sample()
        return sample  # .mean(dim=0).cpu().float().numpy()

    @torch.inference_mode()
    def __call__(self, tensor: Tensor) -> dict[str, Tensor | list[Tensor]]:
        """Extract VAE features from a batch of images then use spectral contrast as divergence metric
        :param dataset: HuggingFace Dataset with 'image' column.
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
        """Compute L1/MSE loss between input and VAE reconstruction.\n"""

        latents = self.vae.encode(tensors).latent_dist.sample()  # type: ignore
        reconstructed = self.vae.decode(latents).sample  # depends on API

        l1_mean = self.l1_loss(reconstructed, tensors)
        mse_mean = self.mse_loss(reconstructed, tensors)

        return {"mse_loss": float(mse_mean), "l1_loss": float(l1_mean)}

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
