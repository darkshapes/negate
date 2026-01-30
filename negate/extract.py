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
    """Graphics processors usable by the VAE pipeline."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Residual:
    def __init__(self, dtype: np.typing.DTypeLike = np.float32):
        """Initialize the residual class"""

        self.dtype = dtype
        self.dtype_name = repr(dtype).split(".")[-1]

    def __call__(self, image: Image) -> Image:
        """Convert the input image to grayscale, apply a Laplace filter,
        and replicate the result across three channels."""

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
        self.device = device
        self.dtype = dtype
        self.model = model
        self.vae: AutoencoderKL | None = None
        self.residual_transform = Residual()
        if self.vae is None:
            self.create_vae()

    def create_vae(self):
        import os

        vae_path = snapshot_download(self.model, allow_patterns=["vae/*"])  # type: ignore
        vae_path = os.path.join(vae_path, "vae")
        vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device.value)  # type: ignore DeviceLike
        vae_model.eval()

        self.vae = vae_model

    def cleanup(self) -> None:  # type:ignore
        """Cleans up the model and frees GPU memory
        :param model: The model instance used for feature extraction"""

        import gc

        device = self.device
        if device != "cpu":
            gpu = getattr(torch, device)
            gpu.empty_cache()
        del self.vae
        gc.collect()


class BatchFeatures:
    def __call__(self, feature_extractor: FeatureExtractor, dataset: Dataset):
        """Extract VAE features from a batch of images."""

        assert hasattr(feature_extractor, "vae")
        features_list = []

        for image in dataset["image"]:
            color_image = image.convert("RGB")
            gray_image = image.convert("L")
            residual_image = feature_extractor.residual_transform(gray_image)

            # Convert to tensors
            color_tensor = feature_extractor.transform(color_image)
            residual_tensor = feature_extractor.transform(residual_image)

            batch_tensor = torch.stack([color_tensor, residual_tensor]).to(feature_extractor.device, dtype=feature_extractor.dtype)

            with torch.no_grad():
                latents_2_dim_h_w = feature_extractor.vae.encode(batch_tensor).latent_dist.sample()
                mean_latent = latents_2_dim_h_w.mean(dim=0).cpu().float().numpy()
                feature_vec = mean_latent.flatten()

            features_list.append(feature_vec)

        return {"features": features_list}


def features(dataset: Dataset) -> Dataset:
    if torch.cuda.is_available():
        device = DeviceName.CUDA
        dtype = torch.bfloat16
    elif torch.mps.is_available():
        device = DeviceName.MPS
        dtype = torch.bfloat16
    else:
        device = DeviceName.CPU
        dtype = torch.float32

    model = "black-forest-labs/FLUX.1-dev"
    extractor = FeatureExtractor(model, device=device, dtype=dtype)
    extractor.create_vae()

    batch_extract = lambda batch: BatchFeatures()(extractor, batch)  # Use lambda to capture extractor

    features_dataset = dataset.map(
        batch_extract,
        batched=True,
        batch_size=4,
        remove_columns=["image"],
        desc="Extracting features...",
    )
    features_dataset.set_format(type="numpy", columns=["features", "label"])
    extractor.cleanup()

    return features_dataset


if __name__ == "__main__":
    features()
