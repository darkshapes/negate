# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum
from dataclasses import dataclass
from datasets import Dataset


class VAEModel(str, Enum):
    """Choose the name and size of the VAE model to use for extraction."""

    SANA_FP32 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"
    SANA_FP16 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"
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
    VAEModel.GLM_BF16: VAEInfo(VAEModel.GLM_BF16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX1_FP32: VAEInfo(VAEModel.FLUX1_FP32, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX1_FP16: VAEInfo(VAEModel.FLUX1_FP16, "autoencoders.autoencoder_kl.AutoencoderKL"),
    VAEModel.FLUX2_FP32: VAEInfo(VAEModel.FLUX2_FP32, "autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2"),
    VAEModel.FLUX2_FP16: VAEInfo(VAEModel.FLUX1_FP16, "autoencoders.autoencoder_kl_flux2.AutoencoderKLFlux2"),
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
        :param model: Repository ID of the VAE.
        :param device: Target device.
        :param dtype: Data type for tensors."""
        from diffusers.models.autoencoders.vae import AutoencoderMixin
        from negate import Residual  # `B̴̨̒e̷w̷͇̃ȁ̵͈r̸͔͛ę̵͂ ̷̫̚t̵̻̐h̶̜͒ȩ̸̋ ̵̪̄ő̷̦ù̵̥r̷͇̂o̷̫͑b̷̲͒ò̷̫r̴̢͒ô̵͍s̵̩̈́` #type: ignore

        self.device = device
        self.dtype = dtype
        self.model: VAEInfo = MODEL_MAP[vae_type]
        self.vae: AutoencoderMixin | None = None
        self.residual_transform = Residual()
        if self.vae is None:
            self.create_vae()

    def create_vae(self):
        """Download and load the VAE from the model repo."""
        import os

        from diffusers.models import autoencoders
        from huggingface_hub.errors import LocalEntryNotFoundError
        from huggingface_hub import snapshot_download

        autoencoder_cls = getattr(autoencoders, self.model.module.split(".")[-1])
        try:
            vae_model = autoencoder_cls.from_pretrained(self.model.enum.value, torch_dtype=self.dtype, local_files_only=True).to(self.device.value)
        except (LocalEntryNotFoundError, OSError):
            print("Downloading model...")
            vae_path: str = snapshot_download(self.model.enum.value, allow_patterns=["vae/*"])  # type: ignore
            vae_path = os.path.join(vae_path, "vae")
            vae_model = autoencoder_cls.from_pretrained(vae_path, torch_dtype=self.dtype, local_files_only=True).to(self.device.value)

        vae_model.eval()
        self.vae = vae_model

    def cleanup(self) -> None:
        """Free the VAE and GPU memory."""
        import gc

        import torch

        device = self.device
        if device != "cpu":
            gpu = getattr(torch, device)
            gpu.empty_cache()
        del self.vae
        gc.collect()

    def batch_extract(self, dataset: Dataset):
        """Extract VAE features from a batch of images.
        :param dataset: HuggingFace Dataset with 'image' column.
        :return: Dictionary with 'features' list."""
        import torch

        assert self.vae is not None
        features_list = []
        patch_stack = []

        for image in dataset["image"]:
            color_image = image.convert("RGB")
            color_tensor = self.transform(color_image)
            patches = self.residual_transform.crop_select(image, size=768, top_k=1)
            for patch in patches:
                patch_image = patch.convert("RGB")
                patch_tensor = self.transform(patch_image)
                patch_stack.append(patch_tensor)

            batch_tensor = torch.stack([color_tensor, *patch_stack]).to(self.device, dtype=self.dtype)
            with torch.no_grad():
                if self.model.enum != VAEModel.SANA_FP32:  # type: ignore can't access encode
                    latents_2_dim_h_w = self.vae.encode(batch_tensor).latent_dist.sample()  # type: ignore can't access encode
                    mean_latent = latents_2_dim_h_w.mean(dim=0).cpu().float().numpy()
                else:
                    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

                    latent = self.vae.encode(batch_tensor)  # # type: ignore can't access encode
                    mean_latent = torch.mean(latent.latent, dim=0).cpu().float()  # distribution with mean
                    logvar_latent = torch.zeros_like(mean_latent).cpu().float()  # & logvar
                    params = torch.cat([mean_latent, logvar_latent], dim=1)
                    distribution = DiagonalGaussianDistribution(params)
                    sample = distribution.sample()
                    mean_latent = sample.mean(dim=0).cpu().float().numpy()
                feature_vec = mean_latent.flatten()

            features_list.append(feature_vec)

        return {"features": features_list}

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "vae"):
            self.cleanup()


def features(dataset: Dataset, vae_type: VAEModel) -> Dataset:
    """Generate a feature dataset from images.\n
    :param dataset: Dataset containing images.
    :return: Dataset with feature vectors."""
    import torch

    device = DeviceName.CUDA if torch.cuda.is_available() else DeviceName.MPS if torch.mps.is_available() else DeviceName.CPU
    dtype = torch.bfloat16

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
