FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git \
    git+https://github.com/zai-org/glm-ocr.git \
    runpod requests pillow accelerate

# Pre-download model weights so cold starts don't hit HuggingFace
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"
ENV HF_HUB_OFFLINE=1

ENV MAX_IMAGE_SIDE=768
ENV MAX_NEW_TOKENS=100
ENV SMART_CROP=1

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
