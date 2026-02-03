# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum
from dataclasses import dataclass
from datasets import Dataset
from PIL.Image import Image


class VAEModel(str, Enum):
    """Choose the name and size of the VAE model to use for extraction."""

    SANA_FP32 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"
    SANA_FP16 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"
    AURAEQUI_BF16 = "exdysa/AuraEquiVAE-SAFETENSORS"
    GLM_BF16 = "zai-org/GLM-Image"
    FLUX2_FP32 = "black-forest-labs/FLUX.2-dev"
    FLUX2_FP16 = "black-forest-labs/FLUX.2-klein-4B"
    FLUX1_FP32 = "Tongyi-MAI/Z-Image"
    FLUX1_FP16 = "Freepik/F-Lite-Texture"
    MITSUA_FP16 = "exdysa/mitsua-vae-SAFETENSORS"


@dataclass
class VAEInfo:
    enum: VAEModel
    module: str  # e.g. "autoencoders.AutoencoderKLFlux2"


MODEL_MAP = {
    VAEModel.MITSUA_FP16: VAEInfo(VAEModel.MITSUA_FP16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.AURAEQUI_BF16: VAEInfo(VAEModel.AURAEQUI_BF16, "ae.VAE"),
    VAEModel.GLM_BF16: VAEInfo(VAEModel.GLM_BF16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX1_FP32: VAEInfo(VAEModel.FLUX1_FP32, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX1_FP16: VAEInfo(VAEModel.FLUX1_FP16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX2_FP32: VAEInfo(VAEModel.FLUX2_FP32, "autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2"),
    VAEModel.FLUX2_FP16: VAEInfo(VAEModel.FLUX2_FP16, "autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2"),
    VAEModel.SANA_FP16: VAEInfo(VAEModel.SANA_FP16, "autoencoders.autoencoder_dc.AutoencoderDC"),
    VAEModel.SANA_FP32: VAEInfo(VAEModel.SANA_FP32, "autoencoders.autoencoder_dc.AutoencoderDC"),
}


class DeviceName(str, Enum):
    """Graphics processors usable by the VAE pipeline.\n"""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class FeatureExtractor:
    import torch
    import torchvision.transforms as T

    transform: T.Compose = T.Compose(
        [
            T.CenterCrop((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(self, vae_type: VAEModel, device: DeviceName, dtype: torch.dtype) -> None:
        """Set up the extractor with a VAE model.\n
        :param vae_type: VAEModel ID of the VAE.
        :param device: Target device.
        :param dtype: Data type for tensors."""

        from diffusers.models.autoencoders.vae import AutoencoderMixin
        from negate import Residual, ae  # `B̴̨̒e̷w̷͇̃ȁ̵͈r̸͔͛ę̵͂ ̷̫̚t̵̻̐h̶̜͒ȩ̸̋ ̵̪̄ő̷̦ù̵̥r̷͇̂o̷̫͑b̷̲͒ò̷̫r̴̢͒ô̵͍s̵̩̈́` #type: ignore

        self.device = device.value
        self.dtype = dtype
        self.model: VAEInfo = MODEL_MAP[vae_type]
        self.vae: AutoencoderMixin | ae.VAE | None = None
        self.residual_transform = Residual()
        if self.vae is None:
            self.create_vae()

    def aura_equi_vae(self, vae_path: str):
        """Processing specifically for AuraEquiVAE
        :param vae_path: The path to the VAE model directory"""
        import os
        from negate.ae import VAE
        from safetensors.torch import load_file

        vae = VAE(resolution=256, in_channels=3, ch=256, out_ch=3, ch_mult=[1, 2, 4, 4], num_res_blocks=2, z_channels=16).to(self.device).bfloat16()
        vae_file = os.path.join(vae_path, "vae_epoch_3_step_49501_bf16.safetensors")
        state_dict = load_file(vae_file)
        vae.load_state_dict(state_dict)
        return vae

    def create_vae(self):
        """Download and load the VAE from the model repo."""

        import os

        from diffusers.models import autoencoders
        from huggingface_hub.errors import LocalEntryNotFoundError
        from huggingface_hub import snapshot_download

        autoencoder_cls = getattr(autoencoders, self.model.module.split(".")[-1], None)
        try:
            vae_model = autoencoder_cls.from_pretrained(self.model.enum.value, torch_dtype=self.dtype, local_files_only=True).to(self.device)
        except (LocalEntryNotFoundError, OSError, AttributeError):
            print("Downloading model...")
        vae_path: str = snapshot_download(self.model.enum.value, allow_patterns=["vae/*"])  # type: ignore
        vae_path = os.path.join(vae_path, "vae")
        if self.model.enum == VAEModel.AURAEQUI_BF16:
            vae_model = self.aura_equi_vae(vae_path)
        else:
            vae_model = autoencoder_cls.from_pretrained(vae_path, torch_dtype=self.dtype, local_files_only=True).to(self.device)

        vae_model.eval()
        self.vae = vae_model

    def _extract_generic(self, batch: "torch.Tensor"):
        """Encode with standard Diffusers VAE and return mean latent.\n
        :param batch: Tensor of image + patches.
        :return: NumPy mean latent."""

        latent = self.vae.encode(batch).latent_dist.sample()  # type: ignore
        return latent.mean(dim=0).cpu().float().numpy()

    def _extract_special(self, batch: "torch.Tensor", image: Image):
        """Handle SANA and AuraEqui models.\n
        :param batch: Tensor of image + patches.
        :param img: Original PIL image.
        :return: NumPy mean latent."""

        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        import torch
        from torch import Tensor

        if self.model.enum == VAEModel.AURAEQUI_BF16:
            latent: Tensor = self.vae.encoder(batch)
            latent = latent.clamp(-8.0, 8.0)
            mean = torch.mean(latent, dim=0).cpu().float()
        else:
            latent: Tensor = self.vae.encode(batch)  # type: ignore
            mean = torch.mean(latent.latent, dim=0).cpu().float()  # type: ignore

        logvar = torch.zeros_like(mean).cpu().float()
        params = torch.cat([mean, logvar], dim=1)
        dist = DiagonalGaussianDistribution(params)
        sample = dist.sample()
        return sample.mean(dim=0).cpu().float().numpy()

    def batch_extract(self, dataset: Dataset):
        """Extract VAE features from a batch of images.
        :param dataset: HuggingFace Dataset with 'image' column.
        :return: Dictionary with 'features' list."""

        import torch

        assert self.vae is not None
        features_list = []
        patch_stack = []

        for image in dataset["image"]:
            rgb = image.convert("RGB")
            col = self.transform(rgb)
            for patches in self.residual_transform.crop_select(image, size=768, top_k=1):
                patch_stack.append(self.transform(patches.convert("RGB")))

            batch = torch.stack([col, *patch_stack]).to(self.device, self.dtype)
            with torch.no_grad():
                match self.model.enum:
                    case VAEModel.SANA_FP32 | VAEModel.SANA_FP16 | VAEModel.AURAEQUI_BF16:
                        mean_latent = self._extract_special(batch, image)
                    case _:
                        mean_latent = self._extract_generic(batch)

            features_list.append(mean_latent.flatten())

        return {"features": features_list}

    def cleanup(self) -> None:
        """Free the VAE and GPU memory."""

        import gc

        import torch

        if self.device != "cpu":
            gpu = getattr(torch, self.device)
            gpu.empty_cache()
            del gpu
        del self.vae
        del self.device
        gc.collect()

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "vae"):
            self.cleanup()


def features(dataset: Dataset, vae_type: VAEModel) -> Dataset:
    """Generate a feature dataset from images.\n
    :param dataset: Dataset containing images.
    :param vae_type: VAE model type for feature extraction.
    :return: Dataset with feature vectors."""

    import torch

    device = DeviceName.CUDA if torch.cuda.is_available() else DeviceName.MPS if torch.mps.is_available() else DeviceName.CPU
    dtype = torch.bfloat16

    # <chud> was here

    with FeatureExtractor(vae_type, device, dtype) as extractor:
        features_dataset = dataset.map(
            extractor.batch_extract,
            batched=True,
            batch_size=4,
            remove_columns=["image"],
            desc="Extracting features...",
        )
    features_dataset.set_format(type="numpy", columns=["features", "label"])

    return features_dataset
