"""
RunPod Serverless handler for GLM-OCR via Transformers.

Loads the model once at startup, then processes incoming OCR jobs.
"""

import os
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
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2000"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "8192"))

processor = None
model = None


def load_model():
    """Load model and processor once at startup."""
    global processor, model
    log.info("Loading GLM-OCR model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    log.info("GLM-OCR model loaded on %s", model.device)


def read_image(source):
    """Read image from URL, data URI, or file path."""
    if source.startswith("data:"):
        header, data = source.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(data)))

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    path = parsed.path if parsed.scheme == "file" else source
    return Image.open(path)


def resize_image(img):
    """Resize if either side exceeds MAX_IMAGE_SIDE."""
    if MAX_IMAGE_SIDE <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_SIDE:
        return img
    ratio = MAX_IMAGE_SIDE / float(longest)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    log.info("Resized image from %sx%s to %sx%s", w, h, *new_size)
    return img.resize(new_size, Image.Resampling.LANCZOS)


def extract_image_and_prompt(job_input):
    """Extract image source and prompt text from various input formats."""
    # Simple format: {"image": "url", "prompt": "Text Recognition:"}
    if "image" in job_input or "url" in job_input:
        image_src = job_input.get("image") or job_input.get("url")
        prompt = job_input.get("prompt", "Text Recognition:")
        return image_src, prompt

    # OpenAI-compatible format with messages
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
    """Process an OCR job."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input")

    if not isinstance(job_input, dict):
        return {"error": "Input must be a JSON object with 'image'/'url' or 'messages'."}

    log.info("Job %s: received", job_id)

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
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    log.info("Job %s: completed (%d output tokens)", job_id, len(generated_ids[0]) - inputs["input_ids"].shape[1])

    return {
        "text": output_text,
        "prompt": prompt,
        "usage": {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(len(generated_ids[0]) - inputs["input_ids"].shape[1]),
        },
    }


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
