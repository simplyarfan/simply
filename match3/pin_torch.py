# Writes pins for whatever torch the base image shipped, e.g. torch==2.13.0+rocm10.0.0.
# pip then refuses to "upgrade" torch to a CUDA build while installing other stuff;
# it fails loudly at build time instead of breaking silently at run time.
from importlib.metadata import PackageNotFoundError, version

for pkg in ("torch", "torchvision", "torchaudio", "pytorch-triton-rocm", "triton"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        pass
