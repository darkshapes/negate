from typing import NamedTuple


class NegateConfig(NamedTuple):
    """YAML config values.\n
    :param patch_size: Patch width for residuals.\n
    :param top_k: Number of patches.\n
    :param vae_tiling: Enable tiling.\n
    :param vae_slicing: Enable slicing.\n
    :param use_onnx: Use ONNX for inference.\n
    :param use_gpu: Use GPU if available.\n
    :return: Config instance."""  # noqa: D401

    batch_size: int
    cache_features: bool
    default_vae: str
    dtype: str
    n_components: float
    num_boost_round: int
    patch_size: int
    top_k: int
    use_onnx: bool
    vae_slicing: bool
    vae_tiling: bool
    early_stopping_rounds: int
    colsample_bytree: float
    eval_metric: list
    learning_rate: float
    max_depth: int
    objective: str
    subsample: float
    scale_pos_weight: float | None
    seed: int


def load_config_options() -> NegateConfig:
    """Load YAML configuration options.\n
    :return: Config dict."""

    from pathlib import Path

    import yaml

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as config_file:
        data = yaml.safe_load(config_file)
    train_cfg = data.pop("train", {})
    data.update(train_cfg)
    return NegateConfig(**data)


negate_options = load_config_options()
