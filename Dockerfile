ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    HF_HOME=/data/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        curl \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Official Misaki's English module imports torch even with trf=False. Install
# its CPU wheel so it cannot pull a second CUDA stack into this ONNX image.
# The project declares CPU ONNX Runtime for normal installs, then replaces it
# with the CUDA 12 build. Never keep both ONNX Runtime packages installed.
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu "torch>=2.5.1,<3"
# pyopenjtalk's isolated build environment can be very slow or stall while
# resolving its compiler dependencies. Install them explicitly so the wheel is
# reproducible and cached before the rest of Misaki is resolved.
RUN python3 -m pip install --no-cache-dir \
        "numpy<2.3" "Cython<4" "cmake<5" "setuptools_scm>=8" \
    && python3 -m pip install --no-cache-dir --no-build-isolation "pyopenjtalk==0.4.1"
RUN python3 -m pip install --no-cache-dir \
        "numpy>=1.23,<2.3" \
        "onnxruntime>=1.17.0" \
        "tokenizers>=0.20.0" \
        "huggingface_hub>=0.23.0" \
        "soundfile>=0.12.1" \
        "scipy>=1.10.0" \
        "sounddevice>=0.5.2" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "kokoro-onnx>=0.6.1,<0.7" \
        "misaki[en,ja,zh]>=0.9.4,<0.10"
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m pip uninstall -y onnxruntime \
    && python3 -m pip install --no-cache-dir "onnxruntime-gpu>=1.20,<2"

# Keep third-party layers reusable when only application code changes.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "zerotts_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
