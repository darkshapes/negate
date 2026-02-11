# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from dataclasses import dataclass
from enum import Enum

from datasets import Dataset


class VAEModel(str, Enum):
    """Choose the name and size of the VAE model to use for extraction."""

    SANA_FP32 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"
    SANA_FP16 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"  # dc_ae 'accuracy': 0.8235294117647058,
    FLUX2_FP32 = "black-forest-labs/FLUX.2-dev"  # f2 dev 'accuracy': 0.9313725490196079,
    FLUX2_FP16 = "black-forest-labs/FLUX.2-klein-4B"  # f2 klein 'accuracy': 0.9215686274509803,
    FLUX1_FP32 = "Tongyi-MAI/Z-Image"  # zimage 'accuracy': 0.9411764705882353,
    FLUX1_FP16 = "Freepik/F-Lite-Texture"  # flite 'accuracy': 0.9509803921568627,
    MITSUA_FP16 = "exdysa/mitsua-vae-SAFETENSORS"  # mitsua 'accuracy': 0.9509803921568627,
    NO_VAE = "None"


@dataclass
class VAEInfo:
    enum: VAEModel
    module: str  # e.g. "autoencoders.AutoencoderKLFlux2"


MODEL_MAP = {
    VAEModel.MITSUA_FP16: VAEInfo(VAEModel.MITSUA_FP16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX1_FP32: VAEInfo(VAEModel.FLUX1_FP32, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX1_FP16: VAEInfo(VAEModel.FLUX1_FP16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX2_FP32: VAEInfo(VAEModel.FLUX2_FP32, "autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2"),
    VAEModel.FLUX2_FP16: VAEInfo(VAEModel.FLUX2_FP16, "autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2"),
    VAEModel.SANA_FP16: VAEInfo(VAEModel.SANA_FP16, "autoencoders.autoencoder_dc.AutoencoderDC"),
    VAEModel.SANA_FP32: VAEInfo(VAEModel.SANA_FP32, "autoencoders.autoencoder_dc.AutoencoderDC"),
    VAEModel.NO_VAE: VAEInfo(VAEModel.NO_VAE, "None"),
}


class VAEExtractor:
    def __init__(self, vae_type: VAEModel) -> None:
        """Set up the extractor with a VAE model.\n
        :param vae_type: VAEModel ID of the VAE.
        :param device: Target device.
        :param dtype: Data type for tensors."""

        from negate import Residual, hyper_param as hp_config, negate_opt, chip  # `B̴̨̒e̷w̷͇̃ȁ̵͈r̸͔͛ę̵͂ ̷̫̚t̵̻̐h̶̜͒ȩ̸̋ ̵̪̄ő̷̦ù̵̥r̷͇̂o̷̫͑b̷̲͒ò̷̫r̴̢͒ô̵͍s̵̩̈́` #type: ignore

        self.residual_transform = Residual(patch_size=negate_opt.dim_patch, top_k=hp_config.top_k)
        self.device = chip.device
        self.dtype = chip.dtype
        self.model: VAEInfo = MODEL_MAP[vae_type]
        if not hasattr(self, "vae") and vae_type != VAEModel.NO_VAE:
            self.create_vae()

    def create_vae(self):
        """Download and load the VAE from the model repo."""
        from negate import negate_opt

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            from diffusers.models import autoencoders  # type: ignore
        except (ImportError, ModuleNotFoundError, Exception):
            raise RuntimeError("missing dependency 'diffusers'. Please install it using 'negate[extractor]'")

        if getattr(negate_opt, "vae_tiling", False):
            self.vae.enable_tiling()
        if getattr(negate_opt, "vae_slicing", False):
            self.vae.enable_slicing()

        autoencoder_cls = getattr(autoencoders, self.model.module.split(".")[-1], None)
        try:
            vae_model = autoencoder_cls.from_pretrained(self.model.enum.value, torch_dtype=self.dtype, local_files_only=True).to(self.device)  # type: ignore
        except (LocalEntryNotFoundError, OSError, AttributeError):
            print("Downloading model...")
        vae_path: str = snapshot_download(self.model.enum.value, allow_patterns=["vae/*"])  # type: ignore
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
        from torch import Tensor

        latent: Tensor = self.vae.encode(batch)  # type: ignore
        mean = torch.mean(latent.latent, dim=0).cpu().float()  # type: ignore

        logvar = torch.zeros_like(mean).cpu().float()
        params = torch.cat([mean, logvar], dim=1)
        dist = DiagonalGaussianDistribution(params)
        sample = dist.sample()
        return sample  # .mean(dim=0).cpu().float().numpy()

    def batch_extract(self, dataset: Dataset):
        """Extract VAE features from a batch of images then use spectral contrast as divergence metric
        :param dataset: HuggingFace Dataset with 'image' column.
        :return: Dictionary with 'features' list."""
        import torch

        features_list = []

        features_latent = torch.stack(dataset["image"]).to(self.device, self.dtype)

        with torch.no_grad():
            if self.model.enum == VAEModel.SANA_FP32 or self.model.enum == VAEModel.SANA_FP16:
                features = self._extract_special(features_latent)
            else:
                features = self.vae.encode(features_latent).latent_dist.sample()

            features_list.append(features)

        return {"features": features_list}

    def cleanup(self) -> None:
        """Free the VAE and GPU memory."""

        import gc

        import torch

        if self.device != "cpu":
            gpu: torch.device = self.device
            gpu.empty_cache()  # type: ignore
            del gpu
        del self.vae
        del self.device
        gc.collect()

    def __enter__(self) -> "VAEExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "vae"):
            self.cleanup()
