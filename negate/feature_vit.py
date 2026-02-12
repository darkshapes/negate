# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clase-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from __future__ import annotations

import torch
from torch import Tensor

from negate.config import Spec


class VITExtract:
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

    def __init__(self, spec: Spec) -> None:
        """Initialize analyzer with configuration.\n"""

        self.model_name = spec.model
        self.dtype = spec.dtype
        self.device = spec.device
        self.cast_move = spec.apply
        self.library = spec.model_config.library_for_model(self.model_name)
        self._set_models()

    @torch.inference_mode()
    def _set_models(self):
        match self.library:
            case "timm":
                import timm

                self.model = timm.create_model(self.model_name, pretrained=True, features_only=True).to(**self.cast_move)  # type: ignore
                data_config = timm.data.resolve_model_data_config(self.model)  # type: ignore
                self.transforms = timm.data.create_transform(**data_config, is_training=False)  # type: ignore
            case "openclip":
                import open_clip

                precision = "f32" if self.dtype is torch.float32 else "bf16" if self.dtype is torch.bfloat16 else "f16"

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    device=self.device,
                    precision=precision,
                )
            case "transformers":
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    trust_remote_code=True,
                )
            case _:
                error = f"{self.library} : Unsupported library"
                raise NotImplementedError(error)

        self.model = self.model.eval().to(**self.cast_move)

    @torch.inference_mode()
    def __call__(self, image: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
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

    def cleanup(self) -> None:
        """Free the VAE and GPU memory."""

        import gc

        if self.device.type != "cpu":
            gpu: torch.device = self.device
            gpu.empty_cache()  # type: ignore
            del gpu
        del self.model
        del self.device
        gc.collect()

    def __enter__(self) -> VITExtract:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "model"):
            self.cleanup()
