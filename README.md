<br>
<br>

# negate <br>

entrypoint synthetic image classifier

[![negate pytest](https://github.com/darkshapes/negate/actions/workflows/negate.yml/badge.svg?branch=main)](https://github.com/darkshapes/negate/actions/workflows/negate.yml)

## About

A command-line tool and Python library for processing and analyzing images, providing methods for feature extraction, laplacian and spectral residual processing, and other comparative analysis methods to discriminate between synthetic images and real ones.

## Overview

We use a modern VAE to extract features from images generated from Diffusers, ComfyUI, Darkshapes tools (Zodiac/singularity) and Google Nano-Banana.

![Bar and grid graph comparing variance of the synthetic and real images](results/score_explained_variance.png)
![Graph comparing before and after pca transform operation of dataset](results/pca_transform_map.png)
![Graph comparing confusion matrix of the synthetic and real images](results/score_confusion_matrix.png)

## Requirements

- A dataset of images made by human artists with width and height dimensions larger than 512 pixels.
- A [huggingface](https://hf.co) account that will be used to download models and synthetic datasets. Create an API Key at their website, then sign in with `hf auth login`.
- It is recommended to run on a GPU to ensure efficient processing and reduce training time.

> [!NOTE]
>
> Our training results and visualizations were created with data provided consensually by generous artists at https://purelyhuman.xyz. We don't have permission to share that dataset here.

## Install

> [!IMPORTANT]
>
> Requires [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/darkshapes/negate.git
cd negate
uv sync
```

<sub>macos/linux</sub>

```bash
source .venv/bin/activate
```

<sub>windows</sub>

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; .venv\Scripts\Activate.ps1
```

## CLI:

Add human-origin assets to `assets/` folder

```sh
usage: negate [-h] {train,check} ...

Negate CLI

positional arguments:
  {train,check}
    train        Train model on the dataset in the provided path or `assets/`. The resulting model will be saved to disk.
    check        Check whether an image at the provided path is synthetic or original.

options:
  -h, --help     show this help message and exit
```
