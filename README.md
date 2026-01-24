<br>
<br>

# negate <br>

entrypoint synthetic image classifier

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

## Test:

```sh
python -m tests.test_extract -v
```

## Scan A Folder or File

```bash
usage: negate [-h] [-i INPUT] [-o OUTPUT] [-v]
```

```
Extract Laplacian residuals from images.

options:
  -h, --help           show this help message and exit
  -i, --input INPUT    Input folder containing images or individual image.
  -o, --output OUTPUT  Output folder for residuals.
  -v, --verbose        Enable verbose output.

```

## Call from another application

```py
import asyncio

from negate import ResidualExtractor

residual_extractor = ResidualExtractor(image_path, output_folder, verbose=verbose)

async def async_main() -> tuple:
    fractal, texture = await residual_extractor.process_residuals()
    return (fractal, texture)

asyncio.run(async_main())
```
