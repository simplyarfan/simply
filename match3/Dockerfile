# Mandated base image. Checked by layer identity, so don't swap it.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# If the base's Python turns out to be the OS one, pip refuses to install
# (PEP 668). Nothing else in this container uses that Python, so allow it.
# No effect when the base uses a venv.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 1. Python packages, with torch locked to the ROCm build that came with the base.
#    The brief's trap #2: pip can swap ROCm torch for a CUDA build. The brief
#    suggests --no-deps; we pin instead (pin_torch.py writes torch==<the ROCm
#    version>), so pip still installs what transformers needs but errors out
#    rather than touching torch. Then we check, and the build fails if torch,
#    torchvision or any import is wrong. Nothing gets to runtime broken.
COPY pin_torch.py build_check.py requirements.txt /tmp/build/
RUN python3 /tmp/build/pin_torch.py > /tmp/build/torch-pins.txt \
 && cat /tmp/build/torch-pins.txt \
 && python3 -m pip install --no-cache-dir -c /tmp/build/torch-pins.txt -r /tmp/build/requirements.txt \
 && python3 /tmp/build/build_check.py

# 2. Model weights, fetched from Hugging Face during the build. There's no
#    network at evaluation time, so the weights have to live in the image.
#    One shard per step keeps every layer under 5 GB, which registries like
#    better. Fetched before the code so editing code doesn't re-fetch 17 GB.
ARG MODEL_REPO=Qwen/Qwen3-VL-8B-Instruct
ARG MODEL_REV=main
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
COPY fetch_model.py /tmp/build/
RUN python3 /tmp/build/fetch_model.py $MODEL_REPO $MODEL_REV /models/qwen3-vl-8b "*.json" "*.txt"
RUN python3 /tmp/build/fetch_model.py $MODEL_REPO $MODEL_REV /models/qwen3-vl-8b model-00001-of-00004.safetensors
RUN python3 /tmp/build/fetch_model.py $MODEL_REPO $MODEL_REV /models/qwen3-vl-8b model-00002-of-00004.safetensors
RUN python3 /tmp/build/fetch_model.py $MODEL_REPO $MODEL_REV /models/qwen3-vl-8b model-00003-of-00004.safetensors
RUN python3 /tmp/build/fetch_model.py $MODEL_REPO $MODEL_REV /models/qwen3-vl-8b model-00004-of-00004.safetensors

# 3. Our code.
COPY app/ /app/
RUN chmod +x /app/start.sh && mkdir -p /app/corpus /app/output /app/index /app/logs

# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL turns on the fast attention kernel
# on Radeon (gfx1100) cards too; MI300X has it either way. Without it a long
# prompt falls back to plain attention, which is slower and needs more memory.
ENV MODEL_DIR=/models/qwen3-vl-8b \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["/app/start.sh"]
