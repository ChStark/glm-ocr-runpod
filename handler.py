"""
RunPod Serverless handler for GLM-OCR — optimized for TCG card scanning.

Optimizations over baseline:
- bfloat16 + explicit CUDA placement (no device_map discovery overhead)
- torch.inference_mode (faster than no_grad)
- 768px max image side (fewer visual patches — ~85% reduction from 2000px)
- BILINEAR resampling (faster than LANCZOS on CPU)
- max_new_tokens=100 (card metadata is short)
- Greedy decoding (do_sample=False, use_cache=True)
- torch.compile for fused CUDA kernels
"""

import os
import time
import base64
import logging
from io import BytesIO
from urllib.parse import urlparse

import requests
import runpod
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

MODEL_PATH = "zai-org/GLM-OCR"
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "768"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))

processor = None
model = None


def load_model():
    """Load model in bfloat16 directly on CUDA, then torch.compile."""
    global processor, model
    log.info("Loading GLM-OCR model (bfloat16, cuda)...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        log.info("torch.compile applied (reduce-overhead mode)")
    except Exception as e:
        log.warning("torch.compile failed, continuing without: %s", e)

    model.eval()
    log.info("GLM-OCR loaded in %.1fs", time.time() - t0)


def read_image(source):
    """Read image from data URI, URL, or file path."""
    if source.startswith("data:"):
        _, data = source.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(data)))

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    path = parsed.path if parsed.scheme == "file" else source
    return Image.open(path)


def resize_image(img):
    """Resize to MAX_IMAGE_SIDE using fast BILINEAR resampling."""
    if MAX_IMAGE_SIDE <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_SIDE:
        return img
    ratio = MAX_IMAGE_SIDE / float(longest)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return img.resize(new_size, Image.Resampling.BILINEAR)


def extract_image_and_prompt(job_input):
    """Extract image source and prompt text from various input formats."""
    if "image" in job_input or "url" in job_input:
        image_src = job_input.get("image") or job_input.get("url")
        prompt = job_input.get("prompt", "Text Recognition:")
        return image_src, prompt

    messages = job_input.get("messages", [])
    image_src = None
    prompt = "Text Recognition:"

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            prompt = content
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                iu = part.get("image_url", {})
                image_src = iu.get("url") if isinstance(iu, dict) else iu
            elif part.get("type") == "text":
                prompt = part.get("text", prompt)

    return image_src, prompt


def handler(job):
    """Process an OCR job with optimized inference."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input")

    if not isinstance(job_input, dict):
        return {"error": "Input must be a JSON object with 'image'/'url' or 'messages'."}

    t0 = time.time()

    image_src, prompt = extract_image_and_prompt(job_input)
    if not image_src:
        return {"error": "No image found in input. Provide 'image', 'url', or 'messages' with image_url."}

    max_tokens = job_input.get("max_tokens", MAX_NEW_TOKENS)

    try:
        img = read_image(image_src)
        img = resize_image(img)
    except Exception as e:
        log.error("Job %s: failed to load image: %s", job_id, e)
        return {"error": f"Failed to load image: {e}"}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    output_len = len(generated_ids[0]) - input_len

    output_text = processor.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=False,
    )

    elapsed = time.time() - t0
    log.info("Job %s: done in %.2fs (in=%d out=%d tokens)", job_id, elapsed, input_len, output_len)

    return {
        "text": output_text,
        "prompt": prompt,
        "usage": {
            "input_tokens": int(input_len),
            "output_tokens": int(output_len),
        },
    }


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
