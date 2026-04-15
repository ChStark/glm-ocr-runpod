# GLM-OCR RunPod Serverless

[![Runpod](https://api.runpod.io/badge/ChStark/glm-ocr-runpod)](https://console.runpod.io/hub/ChStark/glm-ocr-runpod)

GLM-OCR (0.9B) serverless endpoint on RunPod. Powered by vLLM with model weights baked into the image for fast cold starts.

## Usage

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "zai-org/GLM-OCR",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/document.png"}},
            {"type": "text", "text": "Text Recognition:"}
          ]
        }
      ],
      "max_tokens": 4096,
      "temperature": 0.0
    }
  }'
```

## Supported prompts

- `Text Recognition:` — extract text
- `Formula Recognition:` — extract formulas (LaTeX)
- `Table Recognition:` — extract tables (Markdown/HTML)
