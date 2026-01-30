<br>
<br>

# negate <br>

entrypoint synthetic image classifier

[![negate pytest](https://github.com/darkshapes/negate/actions/workflows/negate.yml/badge.svg?branch=main)](https://github.com/darkshapes/negate/actions/workflows/negate.yml)

## About

A command-line tool and Python library for processing and analyzing images, extracting Laplacian residuals to measure fractal and texture complexity, and other comparative analysis methods to discriminate synthetic images from real ones.

## Test Results

Demonstration of the provided test results and visualizations on our synthetic [darkshapes/a_slice dataset](https://huggingface.co/darkshapes/a_slice) and private works of human origin provided by consent from the generous artists at https://purelyhuman.xyz.

![Bar and grid graph comparing variance of the synthetic images](results/variance_0129261712.png)

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

## Train Model:

Add human-origin assets to assets/real

```sh
negate
```
