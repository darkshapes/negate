# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum

import numpy as np
import torch
import torchvision.transforms as T
from datasets import Dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import snapshot_download
from PIL.Image import Image, fromarray
from skimage.filters import laplace


class DeviceName(str, Enum):
    """Graphics processors usable by the VAE pipeline.\n"""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Residual:
    def __init__(self, dtype: np.typing.DTypeLike = np.float32):
        """Initialize Residual.\n
        :param dtype: dtype for internal numpy conversion.
        return: None."""
        self.dtype = dtype

    def __call__(self, image: Image) -> Image:
        """Create a 3‑channel residual from a grayscale image.\n
        :param image: PIL image to process.
        :return: Residual image in RGB mode."""

        greyscale = image.convert("L")
        numeric_image = np.array(greyscale, dtype=self.dtype)
        residual = laplace(numeric_image, ksize=3).astype(self.dtype)
        residual_image: Image = fromarray(np.uint8(residual), mode="L").convert("RGB")
        return residual_image


class FeatureExtractor:
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
        :param dtype: Data type for tensors.
        :return: None."""

        self.device = device
        self.dtype = dtype
        self.model = model
        self.vae: AutoencoderKL | None = None
        self.residual_transform = Residual()
        if self.vae is None:
            self.create_vae()

    def create_vae(self):
        """Download and load the VAE from the model repo."""
        import os

        vae_path = snapshot_download(self.model, allow_patterns=["vae/*"])  # type: ignore
        vae_path = os.path.join(vae_path, "vae")
        vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device.value)  # type: ignore DeviceLike
        vae_model.eval()

        self.vae = vae_model

    def cleanup(self) -> None:  # type:ignore
        """Free the VAE and GPU memory."""

        import gc

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

        assert self.vae is not None
        features_list = []

        for image in dataset["image"]:
            color_image = image.convert("RGB")
            gray_image = image.convert("L")
            residual_image = self.residual_transform(gray_image)

            color_tensor = self.transform(color_image)
            residual_tensor = self.transform(residual_image)

            batch_tensor = torch.stack([color_tensor, residual_tensor]).to(self.device, dtype=self.dtype)

            with torch.no_grad():
                latents_2_dim_h_w = self.vae.encode(batch_tensor).latent_dist.sample()
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

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    match device_type:
        case "cuda":
            device = DeviceName.CUDA
            dtype = torch.bfloat16
        case "mps":
            device = DeviceName.MPS
            dtype = torch.bfloat16
        case _:
            device = DeviceName.CPU
            dtype = torch.float32

    model = "black-forest-labs/FLUX.1-dev"

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
