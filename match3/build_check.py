# Runs at the end of the pip step in the Dockerfile. If anything here fails,
# the build fails, which is a lot better than finding out at evaluation time.
import sys

import torch

print("torch", torch.__version__)
if "rocm" not in torch.__version__:
    sys.exit(f"torch is no longer the ROCm build ({torch.__version__}); a pip install replaced it")

try:
    import torchvision
except ImportError:
    print("torchvision not installed (fine, transformers falls back to PIL)")
else:
    print("torchvision", torchvision.__version__)
    if "+cu" in torchvision.__version__:
        sys.exit(f"torchvision is a CUDA build ({torchvision.__version__})")

import docx  # noqa: E402,F401
import openpyxl  # noqa: E402,F401
import pypdf  # noqa: E402,F401
import transformers  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402,F401

print("transformers", transformers.__version__, "- all imports OK")
