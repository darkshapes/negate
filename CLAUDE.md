# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CRITICAL RULES

- Scan the existing codebase and reuse existing functions wherever possible.
- Keep all imports within functions unless they must be mocked in a test.
- If an import is small, performative, and significantly reduces needs for new code, use the library.
- Generally code should stay under 100 lines.
- A code file should never exceed 200 lines.
- Write short Sphinx docstrings as a single line description, a single line for each parameter, and no empty lines.
- On first line of docstrings use \n instead of line break.
- Variable names must be `snake_case` sequence of descriptive words <=5 letters long
- Keep labels consistent across the entire project.
- In commit messages: use `+` for code adds, `-` for code subtractions, `~` for refactors/fixes.
- Write full variable names at all times. No abbreviations.
- Use descriptive variable names instead of comments.
- No inline comments.
- No emoji.
- No global variables.
- No semantic commit messages.

## Commands

```bash
uv sync --dev          # Install all dependencies (uses uv.lock)
pytest -v              # Run all tests
pytest tests/test_chip.py -v   # Run a single test file
ruff check             # Lint
pyright                # Type check (checks negate/ directory)
negate infer image.png # Run inference on an image
negate train           # Train a new model
negate pretrain        # Extract features and generate visualizations
```

CI runs `pytest -v` on Python 3.13 via GitHub Actions on push/PR to main.

## Architecture

**Data flow:** CLI args → `CmdContext` → preprocessing (wavelet + feature extraction) → PCA → XGBoost → `ModelOutput`

**Training path:** `build_datasets()` → `pretrain()` (wavelet decomposition + VIT/VAE feature extraction) → `train_model()` (PCA + XGBoost) → `save_model()` (`.ubj`, `.pkl`, `.onnx`)

**Inference path:** `generate_dataset()` → `preprocessing()` → `predict_gne_or_syn()` (XGBoost/ONNX) → `ModelOutput` (probability + `OriginLabel`)

### Key modules

- `negate/__main__.py` — CLI entry point with three commands: `pretrain`, `train`, `infer`
- `negate/train.py` — PCA + XGBoost training, returns `TrainResult`
- `negate/inference.py` — Prediction via XGBoost native or ONNX, heuristic weighting
- `negate/decompose/` — Haar wavelet (pytorch_wavelets), Fourier residuals, image scaling
- `negate/extract/` — VIT features (timm/openclip/transformers), VAE reconstruction loss, artwork features (49 CPU-only features)
- `negate/io/spec.py` — `Spec` container that aggregates all config objects; `load_spec()` resolves configs from datestamped result folders
- `negate/io/config.py` — `Chip` singleton for hardware detection (CUDA/MPS/XPU/CPU), TOML config loading, all `Named/Tuple` config containers
- `negate/metrics/heuristics.py` — `compute_weighted_certainty()` combines multi-model results

### Key patterns

- **Chip singleton:** `Chip()` in `config.py` auto-detects GPU hardware and manages dtype globally. Access via `spec.device`, `spec.dtype`.
- **Lazy imports:** `negate/__init__.py` uses `__getattr__` — modules load only when accessed.
- **Spec container:** `Spec` bundles `NegateConfig`, `NegateHyperParam`, `NegateDataPaths`, `NegateModelConfig`, `Chip`, `NegateTrainRounds`. Created from `config/config.toml`.
- **Datestamped folders:** Models saved to `models/YYYYMMDD_HHMMSS/`, results to `results/YYYYMMDD_HHMMSS/`. `load_spec()` can reconstruct a Spec from any datestamped result folder's `config.toml`.
- **OriginLabel enum:** `GNE=0` (genuine/human), `SYN=1` (synthetic/AI). `ModelOutput.from_probability()` converts float → label.

## Configuration

`config/config.toml` is the central config file. It contains dataset repos, model names (VIT/VAE), XGBoost hyperparameters, and training round settings. Tests use `tests/test_config.toml` with overridden values.

Models are exported to both XGBoost native (`.ubj`) and ONNX (`.onnx`) formats, with PCA stored as `.pkl`. Metadata (scale_pos_weight, feature count) goes in `.npz`.

## Linting

Ruff is configured with max line length 140. Pyright checks the `negate/` directory only.
