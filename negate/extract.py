# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum

from datasets import Dataset


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

    def __init__(self, model: str, device: DeviceName, dtype: torch.dtype) -> None:
        """Set up the extractor with a VAE model.\n
        :param model: Repository ID of the VAE.
        :param device: Target device.
        :param dtype: Data type for tensors."""
        from diffusers.models.autoencoders.vae import AutoencoderMixin
        from negate import Residual  # `B̴̨̒e̷w̷͇̃ȁ̵͈r̸͔͛ę̵͂ ̷̫̚t̵̻̐h̶̜͒ȩ̸̋ ̵̪̄ő̷̦ù̵̥r̷͇̂o̷̫͑b̷̲͒ò̷̫r̴̢͒ô̵͍s̵̩̈́` #type: ignore

        self.device = device
        self.dtype = dtype
        self.model = model
        self.vae: AutoencoderMixin | None = None
        self.residual_transform = Residual()
        if self.vae is None:
            self.create_vae()

    def create_vae(self):
        """Download and load the VAE from the model repo."""
        import os

        from diffusers.models import autoencoders
        from huggingface_hub import snapshot_download

        autoencoder_map = {  # dev will be fp32, klein will be fp16. results are better for flux 2 models by ~5%
            "black-forest-labs/FLUX.1-dev": autoencoders.AutoencoderKL,  # type: ignore not exported from
            "Tongyi-MAI/Z-Image": autoencoders.AutoencoderKL,  # type: ignore not exported from
            "black-forest-labs/FLUX.2-dev": autoencoders.AutoencoderKLFlux2,  # type: ignore not exported from
            "black-forest-labs/FLUX.2-klein-9B": autoencoders.AutoencoderKLFlux2,  # type: ignore not exported from
            "black-forest-labs/FLUX.2-klein-4B": autoencoders.AutoencoderKLFlux2,  # type: ignore not exported from
        }

        vae_path = snapshot_download(self.model, allow_patterns=["vae/*"])  # type: ignore
        vae_path = os.path.join(vae_path, "vae")
        vae_model = autoencoder_map[self.model].from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device.value)
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

        for image in dataset["image"]:
            color_image = image.convert("RGB")
            color_tensor = self.transform(color_image)
            batch_tensor = torch.stack([color_tensor]).to(self.device, dtype=self.dtype)  # type: ignore residual tensor

            with torch.no_grad():
                latents_2_dim_h_w = self.vae.encode(batch_tensor).latent_dist.sample()  # type: ignore latent_dist
                mean_latent = latents_2_dim_h_w.mean(dim=0).cpu().float().numpy()
                feature_vec = mean_latent.flatten()

            features_list.append(feature_vec)

        return {"features": features_list}

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "vae"):
            self.cleanup()


def features(dataset: Dataset) -> Dataset:
    """Generate a feature dataset from images.\n
    :param dataset: Dataset containing images.
    :return: Dataset with feature vectors."""
    import torch

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    match device_type:
        case "cuda":
            device = DeviceName.CUDA
            dtype = torch.float32
        case "mps":
            device = DeviceName.MPS
            dtype = torch.float32
        case _:
            device = DeviceName.CPU
            dtype = torch.float32

    model = "Tongyi-MAI/Z-Image"

    with FeatureExtractor(model, device, dtype) as extractor:
        features_dataset = dataset.map(
            extractor.batch_extract,
            batched=True,
            batch_size=4,
            remove_columns=["image"],
            desc="Extracting features...",
        )
    features_dataset.set_format(type="numpy", columns=["features", "label"])

    return features_dataset
