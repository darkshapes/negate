<br>
<br>

# negate <br>

entrypoint synthetic image classifier

[![negate pytest](https://github.com/darkshapes/negate/actions/workflows/negate.yml/badge.svg?branch=main)](https://github.com/darkshapes/negate/actions/workflows/negate.yml)

## About

A command-line tool and Python library for processing and analyzing images, extracting Laplacian residuals to measure fractal and texture complexity, and other comparative analysis methods to discriminate synthetic images from real ones.

> [!NOTE]
>
> Demonstration of the provided test results and visualizations on our synthetic [darkshapes/a_slice dataset](https://huggingface.co/darkshapes/a_slice) and private works of human origin provided by consent from the generous artists at https://purelyhuman.xyz.

## Results Overview

![Bar and grid graph comparing variance of the synthetic and real images](results/variance_20260130_154140.png)
![Graph comparing before and after pca transform operation of dataset](results/pca_transform_20260130_154144.png)
![Graph comparing confusion matrix of the synthetic and real images](results/confusion_matrix_20260130_154142.png)

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
