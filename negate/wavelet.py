# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clase-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Wavelet-based image feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset, IterableDataset
from numpy.typing import NDArray
from PIL import Image

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

        self.np_dtype = getattr(np, negate_opt.numpy_dtype, "float32")
        self.model_dtype = getattr(torch, negate_opt.model_dtype, "float32")

        # WaRPAD paper constants: 1344 → 6×6 patches of 224x224
        self.dim_rescale = dim_rescale
        self.cell_dim = cell_dim
        match negate_opt.library:
            case "openclip":
                import open_clip

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(f"hf-hub:{negate_opt.model}", device="mps")
                self.model = self.model.eval()
            case "timm":
                import timm

                model = timm.create_model(
                    negate_opt.model,
                    pretrained=True,
                    features_only=True,
                )
                model = model.eval()
            case "transformers":
                from transformers import AutoProcessor, AutoModel

                self.processor = AutoProcessor.from_pretrained(negate_opt.model)
                self.model = AutoModel.from_pretrained(
                    negate_opt.model,
                    device_map="auto",
                    dtype=self.model_dtype,
                )

    def _haarlet_decompose(self, img: NDArray) -> tuple[NDArray, NDArray]:
        """Haar wavelet decomposition into low and high frequency components.\n
        :param img: 2D array of pixel values.
        :returns: Tuple of (low_freq, high_freq) arrays same size as input.
        """
        h, w = img.shape

        # Make dimensions even for Haar decomposition
        h_even = h - h % 2
        w_even = w - w % 2
        img = img[:h_even, :w_even]

        low_row = (img[::2] + img[1::2]) / 2  # Low freq rows
        high_row = (img[::2] - img[1::2]) / 2  # High freq rows

        low_freq = (low_row[:, ::2] + low_row[:, 1::2]) / 2  # LL band
        horizontal = (low_row[:, ::2] - low_row[:, 1::2]) / 2  # HL band
        vertical = (high_row[:, ::2] + high_row[:, 1::2]) / 2  # LH band
        diagonal = (high_row[:, ::2] - high_row[:, 1::2]) / 2  # HH band

        # Reconstruct HF component at original resolution using zero-padding upsampling
        hf_upsample = np.zeros((h_even, w_even), dtype=self.np_dtype)

        # Place detail coefficients into their proper positions (standard Haar upsampling pattern)
        hf_upsample[::2, ::2] += 0  # LL: no contribution to HF
        hf_upsample[::2, 1::2] = horizontal[:, :]  # HL → row even, col odd
        hf_upsample[1::2, ::2] = vertical[:]  # LH → row odd, col even
        hf_upsample[1::2, 1::2] = diagonal  # HH → both odd

        return low_freq, hf_upsample

    def _haarlet_high_freq(self, img: NDArray) -> NDArray:
        """Extract high-frequency component HF(x).\n
        :param img: 2D array of pixel values.
        :returns: HF component at same resolution as input (zero-padded for odd dims).
        """
        h, w = img.shape
        _, hf = self._haarlet_decompose(img)

        if hf.shape[0] != h or hf.shape[1] != w:
            hf_padded = np.zeros((h, w), dtype=self.np_dtype)
            hf_padded[: hf.shape[0], : hf.shape[1]] = hf
            return hf_padded

        return hf

    def _segment(self, img: NDArray) -> dict[tuple[int, int], NDArray]:
        """Split image into cells keyed by center coordinates.\n
        :param img: 2D array of pixel values.
        :returns: Dict mapping center (x,y) to cell arrays.
        """
        height, width = img.shape
        cells = {}
        for y_start in range(0, height, self.cell_dim):
            for x_start in range(0, width, self.cell_dim):
                cell = img[
                    y_start : y_start + self.cell_dim,
                    x_start : x_start + self.cell_dim,
                ]
                center_x = x_start + self.cell_dim // 2
                center_y = y_start + self.cell_dim // 2
                cells[(center_x, center_y)] = cell
        return cells

    def _compute_energy(self, cell: NDArray) -> float:
        """Measure high-frequency energy using Haar wavelet.\n
        :param cell: 2D array of pixel values.
        :returns: Energy as a float.
        """
        _, hf = self._haarlet_decompose(cell)
        return float(np.sum(hf**2))

    def _find_extrema(self, cells: dict[tuple[int], NDArray]) -> list[NDArray]:
        """Find min/max energy cells.\n
        :param orig: Original numeric image (unused but kept for signature compatibility).
        :param cells: Dict of center->cell mappings.
        :returns: Tuple of (min_cells list, max_cells list) with centers and data.
        """
        energies = {center: self._compute_energy(cell) for center, cell in cells.items()}

        min_center = min(energies, key=energies.get)  # type: ignore
        max_center = max(energies, key=energies.get)  # type: ignore

        return [cells[min_center], cells[max_center]]

    def _apply_perturbation(self, img: NDArray) -> Image.Image:
        """Apply HF(x) perturbation to image.\n
        :param img: Original numeric image.
        :returns: Perturbed image as PIL Image.
        """
        hf = self._haarlet_high_freq(img)
        perturbed = img - self.alpha * hf

        # Clip to valid pixel range and convert back to 0-255 for uint8
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        return Image.fromarray(perturbed)

    def _make_image(self, arr: NDArray) -> Image.Image:
        """Convert numeric array to PIL Image.\n
        :param arr: Numeric image array.
        :returns: PIL Image in grayscale mode.
        """
        clipped = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(clipped)

    def _cosine_sim(self, a: NDArray, b: NDArray) -> float:
        """Cosine similarity between two 1D vectors.\n
        :param a: First vector.
        :param b: Second vector.
        :returns: Similarity in [-1, 1].
        """
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    def _cosine_sim_t(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two torch tensors.\n
        :param a: First tensor.
        :param b: Second tensor.
        :returns: Similarity in [-1, 1].
        """
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = torch.linalg.norm(a_flat)
        norm_b = torch.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return torch.dot(a_flat, b_flat) / (norm_a * norm_b).cpu().float().numpy()

    def extract_features(self, image: Image.Image | list[Image.Image]) -> NDArray:
        """Run vision model on images to extract deep features.\n
        :param images: Single PIL Image or list of PIL Images.
        :returns: Numpy array of extracted feature vector(s).
        """

        match negate_opt.library:
            case "timm":
                import timm

                image = self.preprocess(image).unsqueeze(0).to(self.model.device)  # type: ignore
                data_config = timm.data.resolve_model_data_config(self.model)  # type: ignore
                transforms = timm.data.create_transform(**data_config, is_training=False)  # type: ignore

                image_features = self.model(transforms(image).unsqueeze(0))

            case "openclip":
                image = self.preprocess(image).unsqueeze(0).to("mps")  # type: ignore

                with torch.no_grad(), torch.inference_mode():
                    image_features = self.model.encode_image(image)  # type: ignore
                    image_features /= image_features.norm(dim=-1, keepdim=True)

            case "transformers":
                inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
                with torch.inference_mode():
                    image_features = self.model(**inputs)
                    image_features = self.model(**inputs)
                    image_features.pooler_output

            case _:
                raise NotImplementedError(f"{negate_opt.library} : Unsupported library")

        return image_features.cpu().float().numpy()

    def _rescale_image(self, pil_img: Image.Image) -> NDArray:
        """Rescale image to d_rescale x d_rescale.\n
        :param pil_img: Input PIL Image.
        :returns: Rescaled numpy array (0-255 float).
        """
        resized = pil_img.resize((self.dim_rescale, self.dim_rescale), Image.Resampling.BICUBIC)
        return np.array(resized).astype(self.np_dtype)

    def _partition_patches(self, img: NDArray) -> list[NDArray]:
        """Partition image into non-overlapping patches of patch_size.\n
        :param img: 2D array to partition.
        :returns: List of patch arrays.
        """
        patches = []
        for y in range(0, self.dim_rescale, self.cell_dim):
            for x in range(0, self.dim_rescale, self.cell_dim):
                patches.append(img[y : y + self.cell_dim, x : x + self.cell_dim])
        return patches

    def compute_sensitivity(self, original_image: Image.Image, magnitude_sampling: bool = False) -> float:
        """Compute WaRPAD sensitivity score following paper methodology.\n
        :param image_path: Path to input image file.
        :returns: Average cosine similarity (Eq. 3 from paper).
        """
        # 1. Load and rescale to d_rescale (1344)
        img_arr = self._rescale_image(original_image.convert("L"))

        # 2. Partition into K×K patches of size patch_size
        patches = self._partition_patches(img_arr)  # K=6, so 36 patches
        if magnitude_sampling:
            patches = self._find_extrema(patches)  # type: ignore
            # patches = min_cells.extend(max_cells)  # type: ignore

        scores: list[float] = []
        for patch in patches:
            # 3. Extract features from original patch (Eq. 1 numerator)
            orig_feat_np = self.extract_features(self._make_image(patch))
            assert orig_feat_np.ndim == 2 and orig_feat_np.shape[0] == 1
            orig_vec = torch.from_numpy(orig_feat_np[0])  # type: ignore

            # 4. Perturbed patch: x - α·HF(x)
            hf = self._haarlet_high_freq(patch)
            pert_patch = patch - self.alpha * hf

            # 5. Extract features from perturbed patch
            pert_feat_np = self.extract_features(self._make_image(pert_patch))
            assert pert_feat_np.ndim == 2 and pert_feat_np.shape[0] == 1
            pert_vec = torch.from_numpy(pert_feat_np[0])  # type: ignore

            # 6. Cosine similarity for this patch (Eq. 1)
            score = self._cosine_sim_t(orig_vec, pert_vec)  # Hfwav(p_i)
            scores.append(score)

        # 7. Average across all patches (Eq. 3)
        return float(np.mean(scores))  # WaRPAD(x)

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
            result = {"sensitivity": []}
            for img in batch["image"]:
                # HFwav(x) from Eq 1: similarity between original and HF-perturbed version
                sens = self.compute_sensitivity(img, magnitude_sampling=negate_opt.magnitude_sampling)
                result["sensitivity"].append(sens)
            return result

        return dataset.map(
            process_batch,
            remove_columns=["image"],
            desc="Computing HF sensitivity...",  # type: ignore
            **kwargs,
        )
