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

from negate import negate_opt


class WaveletAnalyzer:
    """Extract wavelet energy features from images.

    Attributes:
        cell_dim: Size of square cells for analysis.
        resize_percent: Percent to resize images before celling.
        batch_size: Batch size for dataset processing (0 = no batching).
        alpha: Perturbation weight for HF(x) subtraction (0 < α < 1).

    Example:
        >>> import datasets as Datasets
        >>> from negate.datasets import generate_dataset
        >>> dataset = generate_dataset("path/to/images",label=0)
        >>> analyzer = WaveletAnalyzer(cell_dim=16, resize_percent=1.0, batch_size=20, alpha:0.7)
        >>> features: list[dict[str,NDArray]] = analyzer.decompose(dataset)
        >>> list(features["sensitivity])
    """

    def __init__(
        self,
        cell_dim: int = negate_opt.cell_dim,
        resize_percent: float = negate_opt.resize_pct,
        batch_size: int = negate_opt.batch_size,
        dim_rescale: int = negate_opt.dim_rescale,
        alpha: float = negate_opt.alpha,
    ) -> None:
        """Initialize analyzer with configuration.\n
        :param cell_dim: Dimension of square cells (default 224).
        :param resize_percent: Resize factor before celling (default 1.0, no resize).
        :param batch_size: Batch size for processing (0 disables batching).
        :param alpha: Perturbation weight (0 < α < 1) for HF(x) subtraction.
        """

        self.resize_percent = resize_percent
        self.batch_size = batch_size
        self.alpha = alpha

        self.device = torch.device("cpu")
        self.np_dtype = getattr(np, negate_opt.numpy_dtype, "float32")
        self.model_dtype = getattr(torch, negate_opt.model_dtype, "float32")
        self.magnitude_sampling = negate_opt.magnitude_sampling
        self.dim_rescale = dim_rescale
        self.cell_dim = cell_dim
        self._set_models()

    @torch.inference_mode()
    def _set_models(self):
        """Create the model definitions for the class"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")

        match negate_opt.library:
            case "openclip":
                import open_clip

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(f"hf-hub:{negate_opt.model}", device=self.device)
                self.model = self.model.eval()
            case "timm":
                import timm

                model = timm.create_model(
                    negate_opt.model,
                    pretrained=True,
                    features_only=True,
                ).to(self.device)
                model = model.eval()
            case "transformers":
                from transformers import AutoModel, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(negate_opt.model)
                self.model = AutoModel.from_pretrained(
                    negate_opt.model,
                    device_map="auto",
                    dtype=self.model_dtype,
                ).to(self.device)

    @torch.inference_mode()
    def metric(self, tensor_patch: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute sensitivity metrics for image patches.\n
        :param tensor_patch: Input tensor of shape (C, H, W).
        :returns: Tuple of (min_similarity, max_similarity, min_index, max_index).
        """
        dwt = DWTForward(J=2, wave="haar").to(self.device)
        idwt = DWTInverse(wave="haar").to(self.device)
        if tensor_patch.device != self.device:
            tensor_patch = tensor_patch.to(self.device)

        b, c, h, w = tensor_patch.shape

        zero_tensor = tensor_patch.unfold(2, self.cell_dim, self.cell_dim).unfold(3, self.cell_dim, self.cell_dim)
        reshaped = zero_tensor.reshape([b, c, -1, self.cell_dim, self.cell_dim]).transpose(1, 2)
        patches = reshaped.reshape([-1, c, self.cell_dim, self.cell_dim])

        if patches.shape[0] == 0:
            empty = torch.tensor([], dtype=torch.float32, device=self.device)
            return (empty[:1].fill_(0), empty[:1].fill_(0), empty[:1].long(), empty[:1].long())

        yl, yh = dwt(patches.to(self.device))
        yl_zeros = torch.zeros_like(yl)
        pert_hf = idwt((yl_zeros, yh))
        perturbed_patches = patches.to(self.model_dtype) - (self.alpha * pert_hf).to(self.model_dtype)

        outputs = self.extract_features(patches.to(self.device))
        perturbed_outputs = self.extract_features(perturbed_patches.to(self.device))

        similarity: Tensor = torch.nn.functional.cosine_similarity(outputs, perturbed_outputs, dim=-1)  # type: ignore
        # similarity = similarity.unsqueeze(1)

        similarity_min = torch.mean(similarity, dim=0, keepdim=True)
        index_min = torch.argmin(similarity).view(-1)
        similarity_max = torch.max(similarity, dim=0, keepdim=True).values
        index_max = torch.argmax(similarity).view(-1)

        return similarity_min, similarity_max, index_min, index_max

    @torch.inference_mode()
    def extract_features(self, image: Tensor) -> NDArray:
        """Run vision model on images to extract deep features.\n
        :param images: Single PIL Image or list of PIL Images.
        :returns: Numpy array of extracted feature vector(s).
        """

        match negate_opt.library:
            case "timm":
                import timm

                data_config = timm.data.resolve_model_data_config(self.model)  # type: ignore
                transforms = timm.data.create_transform(**data_config, is_training=False)  # type: ignore
                image_features = self.model(transforms(image).unsqueeze(0))

            case "openclip":
                image_features = self.model.encode_image(image)  # type: ignore
                image_features /= image_features.norm(dim=-1, keepdim=True)

            case "transformers":
                image_features = self.model(image)
                image_features.pooler_output

            case _:
                raise NotImplementedError(f"{negate_opt.library} : Unsupported library")

        return image_features

    @torch.inference_mode()
    def compute_sensitivity(self, original_images: Image.Image | list[Image.Image]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Compute WaRPAD(x) sensitivity score following Choi12025- arXiv:2511.14030v1 methodology.\n
        ```
        1. Load and rescale to d_rescale (1344)
        2. Partition into K×K patches of size patch_size
        3. Extract features from original patch (Eq. 1 numerator)
        4. Perturbed patch: x - α·HF(x)
        5. Extract features from perturbed patch
        6. Cosine similarity for this patch (Eq. 1)
        7. Average of cosine similarity across all patches (Eq. 3)
        = HFwav(x) from Eq 1: similarity between original and HF-perturbed version
        ```
        :param original_images: Single PIL Image or list of PIL Images.
        :returns: Tuple of (sim_min, sim_max, idx_min, idx_max) arrays per image.
        """

        rescale = negate_opt.dim_rescale
        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transform: T.Compose = T.Compose([T.Resize((rescale, rescale), interpolation=Image.BICUBIC), T.ToTensor(), normalize])  # type: ignore

        if isinstance(original_images, list):
            tensor_patches: Tensor = torch.stack([transform(image) for image in original_images]).to(self.device)
        else:
            tensor_patches = transform(original_images).unsqueeze(0).to(self.device)

        patch_smin, patch_smax, patch_imin, patch_imax = self.metric(tensor_patches)
        return (
            patch_smin.cpu().numpy(),
            patch_smax.cpu().numpy(),
            patch_imin.cpu().numpy(),
            patch_imax.cpu().numpy(),
        )

    def decompose(self, dataset: Dataset | IterableDataset) -> Dataset:
        """Generate feature dataset from wavelet sensitivity scores.\n
        :param dataset: Input dataset with image column.
        :returns: Dataset containing similarity scores from HFwav(x).
        """
        kwargs = {}
        if self.batch_size > 0:
            kwargs["batched"] = True
            kwargs["batch_size"] = self.batch_size

        def squeeze_or_item(val):
            """Convert (1,) to scalar, leave arrays as-is."""
            if hasattr(val, "__len__") and len(val) == 1:
                return val[0]
            return val

        def process_batch(batch):
            images = batch["image"]
            sense = self.compute_sensitivity(images)
            # Each element is shape (1,) - squeeze to scalar per image
            sim_min = np.array([squeeze_or_item(x) for x in sense[0].tololist()])
            sim_max = np.array([squeeze_or_item(x) for x in sense[1].tololist()])

            result = {
                "sim_min": sim_min,
                "sim_max": sim_max,
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
