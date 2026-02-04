# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from datasets import Dataset
from PIL.Image import Image

from negate.config import negate_options as negate_opt


class VAEModel(str, Enum):
    """Choose the name and size of the VAE model to use for extraction."""

    SANA_FP32 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"
    SANA_FP16 = "exdysa/dc-ae-f32c32-sana-1.1-diffusers"  # dc_ae 'accuracy': 0.8235294117647058,
    FLUX2_FP32 = "black-forest-labs/FLUX.2-dev"  # f2 dev 'accuracy': 0.9313725490196079,
    FLUX2_FP16 = "black-forest-labs/FLUX.2-klein-4B"  # f2 klein 'accuracy': 0.9215686274509803,
    FLUX1_FP32 = "Tongyi-MAI/Z-Image"  # zimage 'accuracy': 0.9411764705882353,
    FLUX1_FP16 = "Freepik/F-Lite-Texture"  # flite 'accuracy': 0.9509803921568627,
    MITSUA_FP16 = "exdysa/mitsua-vae-SAFETENSORS"  # mitsua 'accuracy': 0.9509803921568627,


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

        from negate import Residual  # `B̴̨̒e̷w̷͇̃ȁ̵͈r̸͔͛ę̵͂ ̷̫̚t̵̻̐h̶̜͒ȩ̸̋ ̵̪̄ő̷̦ù̵̥r̷͇̂o̷̫͑b̷̲͒ò̷̫r̴̢͒ô̵͍s̵̩̈́` #type: ignore

        self.residual_transform = Residual(patch_size=negate_opt.patch_size, top_k=negate_opt.top_k)
        self.device = device.value
        self.dtype = dtype
        self.model: VAEInfo = MODEL_MAP[vae_type]
        if not hasattr(self, "vae"):
            self.create_vae()

    def create_vae(self):
        """Download and load the VAE from the model repo."""

        import os

        from diffusers.models import autoencoders
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        if negate_opt.vae_tiling:
            self.vae.enable_tiling()
        if negate_opt.vae_slicing:
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

        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from torch import Tensor

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
            for patches in self.residual_transform.crop_select(image):
                patch_stack.append(self.transform(patches.convert("RGB")))

            batch = torch.stack([col, *patch_stack]).to(self.device, self.dtype)
            with torch.no_grad():
                match self.model.enum:
                    case VAEModel.SANA_FP32 | VAEModel.SANA_FP16:
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


def feature_cache(dataset: Dataset, vae_type: VAEModel) -> Path:
    """Generate cache filename based on dataset fingerprint and VAE model.\n
    Dataset fingerprint automatically changes when data changes.

    :param dataset: The incoming dataset to be processed.
    :param vae_type: The VAE model typeselectedfor feature extraction.
    :returns: Location to cache results of feature extraction"""

    import hashlib

    cache_dir = Path(".cache/features")  # <chud> was here
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = dataset._fingerprint if hasattr(dataset, "_fingerprint") else hashlib.md5(str(dataset).encode()).hexdigest()[:8]
    vae_name = vae_type.value.split("/")[-1].replace("-", "_")
    cache_file = cache_dir / f"features_{vae_name}_{dataset_hash}.arrow"

    if cache_file.exists():
        print(f"Using cached features from {cache_file}")
    else:
        print(f"Extracting features (will cache to {cache_file})")
    return cache_file


def features(dataset: Dataset, vae_type: VAEModel) -> Dataset:
    """Generate a feature dataset from images.\n
    :param dataset: Dataset containing images.
    :param vae_type: VAE model type for feature extraction.
    :return: Dataset with feature vectors."""

    import torch

    device = DeviceName.CUDA if torch.cuda.is_available() else DeviceName.MPS if torch.mps.is_available() else DeviceName.CPU

    dtype = getattr(torch, negate_opt.dtype)
    kwargs = {}
    match negate_opt:
        case opt if opt.batch_size > 0:
            kwargs.setdefault("batch_size", opt.batch_size)
        case opt if opt.cache_features:
            cache_file_name = str(feature_cache(dataset, vae_type))
            kwargs.setdefault("cache_file_name", cache_file_name)

    with FeatureExtractor(vae_type, device, dtype) as extractor:
        features_dataset = dataset.map(
            extractor.batch_extract,
            batched=negate_opt.batch_size > 0,
            remove_columns=["image"],
            desc="Extracting features...",
            **kwargs,
        )
    features_dataset.set_format(type="numpy", columns=["features", "label"])

    return features_dataset
