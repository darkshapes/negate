# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clase-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Wavelet-based image feature extraction."""

import numpy as np
import torch
import torchvision.transforms as T
from datasets import Dataset, IterableDataset
from numpy.typing import NDArray
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
from torch import Tensor

from negate import negate_opt, model_config


class WaveletAnalyzer:
    """Extract wavelet energy features from images.

    Attributes:
        patch_dim: Size of square cells for analysis.
        batch_size: Batch size for dataset processing (0 = no batching).
        alpha: Perturbation weight for HF(x) subtraction (0 < α < 1).

    Example:
        >>> import datasets as Datasets
        >>> from negate.datasets import generate_dataset
        >>> dataset = generate_dataset("path/to/images",label=0)
        >>> analyzer = WaveletAnalyzer("timm/vit_base_patch16_dinov3.lvd1689m")
        >>> features: list[dict[str,NDArray]] = analyzer.decompose(dataset)
        >>> list(features["sensitivity])
    """

    def __init__(self, model_name: str) -> None:
        """Initialize analyzer with configuration.\n
        :param dim_patch: Dimension of square cells (default 224).
        :param resize_percent: Resize factor before celling (default 1.0, no resize).
        :param batch_size: Batch size for processing (0 disables batching).
        :param alpha: Perturbation weight (0 < α < 1) for HF(x) subtraction.
        """
        self.model_name = model_name

        self.batch_size = negate_opt.batch_size
        self.alpha = negate_opt.alpha
        self.dim_patch = negate_opt.dim_patch
        self.dim_rescale = negate_opt.dim_rescale * self.dim_patch
        self.magnitude_sampling = negate_opt.magnitude_sampling
        self.np_dtype = getattr(np, negate_opt.dtype)
        self.model_dtype: torch.dtype = getattr(torch, negate_opt.dtype)
        self._set_models()

    @torch.inference_mode()
    def _set_models(self):
        self.library = model_config.library_for_model(self.model_name)
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.cast_move = {"device": self.device, "dtype": self.model_dtype}

        match self.library:
            case "timm":
                import timm

                self.model = timm.create_model(self.model_name, pretrained=True, features_only=True).to(**self.cast_move)
                data_config = timm.data.resolve_model_data_config(self.model)  # type: ignore
                self.transforms = timm.data.create_transform(**data_config, is_training=False)  # type: ignore
            case "openclip":
                import open_clip

                precision = "f32" if self.model_dtype is torch.float32 else "bf16" if self.model_dtype is torch.bfloat16 else "f16"

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    device=self.device,
                    precision=precision,
                )
            case "transformers":
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    dtype=self.model_dtype,
                    trust_remote_code=True,
                )
            case _:
                error = f"{self.library} : Unsupported library"
                raise NotImplementedError(error)

        self.dwt = DWTForward(J=2, wave="haar").to(**self.cast_move)
        self.idwt = DWTInverse(wave="haar").to(**self.cast_move)

        self.model = self.model.eval().to(**self.cast_move)

    @torch.inference_mode()
    def _prepare_patches(self, tensor_patch: Tensor) -> Tensor:
        """Prepare image patches for wavelet analysis.\n
        Handles device placement, MPS optimizations, and patch extraction from input tensor.\n
        :param tensor_patch: Input tensor of shape (B, C, H, W).
        :returns: Tensor containing flattened image patches.
        """
        if tensor_patch.device != self.device:
            tensor_patch = tensor_patch.to(**self.cast_move)

        if self.device.type == "mps" and tensor_patch.shape[0] > 4:
            tensor_patch = tensor_patch[:4]

        b, c = tensor_patch.shape[:2]
        dim_p = self.dim_patch

        flat_patches = tensor_patch.unfold(2, dim_p, dim_p).unfold(3, dim_p, dim_p).reshape([b, c, -1, dim_p, dim_p]).transpose(1, 2).flatten(0, 1)

        if self.device.type == "mps" and flat_patches.numel() > 10_000_000:  # ~40MB for float32
            mps_reduction_factor = max(2, int(flat_patches.shape[0] / 8))
            flat_patches = flat_patches[::mps_reduction_factor]

        return flat_patches

    @torch.inference_mode()
    def process_patches(self, tensor_patch: Tensor) -> tuple[Tensor, Tensor]:
        """Compute sensitivity metrics for image patches.\n
        :param tensor_patch: Input tensor of shape (B, C, H, W).
        :returns: Tuple of (min_similarity, max_similarity, min_index, max_index).
        """

        flat_patches = self._prepare_patches(tensor_patch)

        if flat_patches.shape[0] == 0:
            empty = torch.tensor([], **self.cast_move)
            return (empty[:1].fill_(0), empty[:1].fill_(0))

        yl, yh = self.dwt(flat_patches)
        pert_hf = self.idwt((torch.zeros_like(yl), yh))

        perturbed_patches = flat_patches - self.alpha * pert_hf

        feat_out: Tensor | NDArray = self.extract_features(flat_patches)
        pert_feat_output: Tensor | NDArray = self.extract_features(perturbed_patches)

        if isinstance(feat_out, list):
            outputs: Tensor = torch.cat(feat_out) if all(isinstance(x, Tensor) for x in feat_out) else feat_out[0]
        else:
            outputs = feat_out  # type: ignore
        if isinstance(pert_feat_output, list):
            perturbed_outputs: Tensor = torch.cat(pert_feat_output) if all(isinstance(x, Tensor) for x in pert_feat_output) else pert_feat_output[0]
        else:
            perturbed_outputs = pert_feat_output  # type: ignore

        return outputs, perturbed_outputs

    @torch.inference_mode()
    def extract_features(self, image: Tensor) -> NDArray:
        """Run vision model on images to extract deep features.\n
        :param images: Single PIL Image or list of PIL Images.
        :returns: Numpy array of extracted feature vector(s).
        """

        match self.library:
            case "timm":
                try:
                    image_features = self.model(self.transforms(image))
                except (RuntimeError, Exception) as _exec_info:
                    image_features = self.model(self.transforms(image).unsqueeze(0))

            case "openclip":
                image_features = self.model.encode_image(image)  # type: ignore
                image_features /= image_features.norm(dim=-1, keepdim=True)

            case "transformers":
                try:
                    image_features = self.model(image).pixel_values
                    image_features = image_features.pooler_output
                except AttributeError as _error:
                    _, image_features = self.model(image)

            case _:
                raise NotImplementedError("Unsupported model configuration")

        return image_features

    @torch.inference_mode()
    def compute_sensitivity(self, original_images: Image.Image | list[Image.Image]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Compute WaRPAD(x) sensitivity score following Choi12025 methodology.\n
        :param original_images: Single PIL Image or list of PIL Images.
        :returns: Tuple of (sim_min, sim_max, idx_min, idx_max) per image.
        """

        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transform: T.Compose = T.Compose([T.Resize((self.dim_rescale, self.dim_rescale), interpolation=Image.BICUBIC), T.ToTensor(), normalize])  # type: ignore

        if isinstance(original_images, list):
            batch_size = len(original_images)
            tensor_patches: Tensor = torch.stack([transform(image) for image in original_images]).to(**self.cast_move)  # type: ignore
        else:
            batch_size = 1
            tensor_patches = transform(original_images).unsqueeze(0).to(self.model_dtype).to(**self.cast_move)  # type: ignore

        outputs, perturbed_outputs = self.process_patches(tensor_patches)

        if negate_opt.euclidean:
            similarity = 1 / (1 + torch.nn.functional.pairwise_distance(outputs, perturbed_outputs))
        else:
            similarity: Tensor = torch.nn.functional.cosine_similarity(outputs, perturbed_outputs, dim=-1)

        sim_min_val, idx_min = similarity.min(dim=0)
        sim_max_val, idx_max = similarity.max(dim=0)

        sim_min = sim_min_val.unsqueeze(0)
        sim_max = sim_max_val.unsqueeze(0)
        idx_min = idx_min.view(-1)
        ids_max = idx_max.view(-1)

        sim_min = np.full(batch_size, float(sim_min.item()) if sim_min.numel() == 1 else sim_min.cpu().numpy().flatten()[0])
        sim_max = np.full(batch_size, float(sim_max.item()) if sim_max.numel() == 1 else sim_max.cpu().numpy().flatten()[0])
        idx_min = np.full(batch_size, int(idx_min.item()) if idx_min.numel() == 1 else idx_min.cpu().numpy().flatten()[0])
        idx_max = np.full(batch_size, int(ids_max.item()) if ids_max.numel() == 1 else ids_max.cpu().numpy().flatten()[0])

        return sim_min, sim_max, idx_min, idx_max

    def decompose(self, dataset: Dataset | IterableDataset) -> Dataset:
        """Generate feature dataset from wavelet sensitivity scores.\n
        :param dataset: Input dataset with image column.
        :returns: Dataset containing similarity scores from HFwav(x).
        """
        kwargs = {}
        if self.batch_size > 0:
            kwargs["batched"] = True
            kwargs["batch_size"] = self.batch_size

        def process_batch(batch):
            images = batch["image"]
            sense = self.compute_sensitivity(images)

            result = {
                "sim_min": np.array(sense[0]),
                "sim_max": np.array(sense[1]),
                "idx_min": np.array(sense[2], dtype=int),
                "idx_max": np.array(sense[3], dtype=int),
            }
            return result

        return dataset.map(
            process_batch,
            remove_columns=["image"],
            desc="Computing HF sensitivity...",  # type: ignore
            **kwargs,
        )
