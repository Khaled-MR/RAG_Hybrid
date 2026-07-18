# RAG backend (FastAPI) — GPU-enabled (BGE-M3 embedder + reranker on CUDA).
# Build from the repo root:  docker compose build backend
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=8 \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# No apt packages needed — all Python deps ship prebuilt wheels for linux x86_64.
# Install the CUDA 12.8 torch FIRST (Ada/Blackwell; the cu128 wheel bundles the
# CUDA runtime, so no CUDA base image is needed). Doing it first means
# FlagEmbedding won't also pull a redundant CPU torch — halving the big
# downloads and the chances of a flaky-network failure.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
COPY requirements.txt .
RUN pip install -r requirements.txt "transformers>=4.44.2,<5.0"

COPY backend/ /app/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
