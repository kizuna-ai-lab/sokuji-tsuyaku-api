# Translation Service Guide

This guide explains how to use the real-time translation features in Speaches.

## Overview

Speaches has been transformed into a real-time translation service that provides:

- **Text-to-Text Translation**: Translate text between 1000+ language pairs using MarianMT models
- **Real-time Audio Translation**: STT → Translation → TTS workflow for live audio translation
- **WebSocket/WebRTC Support**: Low-latency real-time translation over WebSocket and WebRTC

## Quick Start

### 1. Install Dependencies

Make sure you have `transformers` installed:

```bash
# Update uv
uv self update

# Sync dependencies
uv sync
```

### 2. Start the Server

```bash
uvicorn speaches.main:create_app --factory --host 0.0.0.0 --port 8000
```

### 3. Test Translation Endpoint

```bash
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -F "text=Hello, how are you?" \
  -F "model=Helsinki-NLP/opus-mt-en-zh"
```

Response:
```json
{
  "text": "你好，你好吗？",
  "model": "Helsinki-NLP/opus-mt-en-zh",
  "source_language": "en",
  "target_language": "zh"
}
```

## API Endpoints

### `/v1/audio/translations` (POST)

Translate text from one language to another.

**Parameters:**
- `text` (required): The text to translate
- `model` (required): The MarianMT model ID (e.g., `Helsinki-NLP/opus-mt-en-zh`)
- `source_language` (optional): Source language code
- `target_language` (optional): Target language code

**Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/translations",
    data={
        "text": "Hello, world!",
        "model": "Helsinki-NLP/opus-mt-en-es",
    }
)

print(response.json())
# {'text': '¡Hola, mundo!', 'model': 'Helsinki-NLP/opus-mt-en-es', ...}
```

## Using the OpenAI Client

You can use the OpenAI Python client to access the translation API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # or your API key if configured
)

# Translate text
translation = client.audio.translations.create(
    file=("text.txt", "Hello, how are you?".encode()),
    model="Helsinki-NLP/opus-mt-en-zh"
)

print(translation.text)  # 你好，你好吗？
```

## Real-time Translation (WebSocket)

Connect to the WebSocket endpoint for real-time audio translation:

```python
import asyncio
import websockets
import json

async def realtime_translation():
    uri = "ws://localhost:8000/v1/realtime?model=Helsinki-NLP/opus-mt-en-zh"

    async with websockets.connect(uri) as websocket:
        # Wait for session.created event
        event = json.loads(await websocket.recv())
        print(f"Session created: {event}")

        # Send audio data for transcription
        # ... (see Realtime API documentation)

asyncio.run(realtime_translation())
```

## Supported Language Pairs

MarianMT supports 1000+ language pairs from the Helsinki-NLP project. Common models:

| Model ID | Source → Target | Example |
|----------|----------------|---------|
| `Helsinki-NLP/opus-mt-en-zh` | English → Chinese | Hello → 你好 |
| `Helsinki-NLP/opus-mt-en-es` | English → Spanish | Hello → Hola |
| `Helsinki-NLP/opus-mt-en-fr` | English → French | Hello → Bonjour |
| `Helsinki-NLP/opus-mt-zh-en` | Chinese → English | 你好 → Hello |
| `Helsinki-NLP/opus-mt-es-en` | Spanish → English | Hola → Hello |
| `Helsinki-NLP/opus-mt-fr-en` | French → English | Bonjour → Hello |

Find more models on HuggingFace: https://huggingface.co/Helsinki-NLP

## Model Management

### Model Download and Conversion

Models are automatically downloaded and converted on first use:

1. **Download**: Model is fetched from HuggingFace Hub
2. **Conversion**: Automatically converted to CTranslate2 format (int8 quantization)
3. **Caching**: Converted models are cached locally

### Model TTL (Time-to-Live)

Models are automatically unloaded after inactivity to save memory:

```bash
# Configure model TTL (in seconds)
export MARIAN__MODEL_TTL=300  # Default: 300 seconds (5 minutes)

# Keep models loaded forever
export MARIAN__MODEL_TTL=-1

# Unload immediately after use
export MARIAN__MODEL_TTL=0
```

### Model Configuration

Configure MarianMT settings via environment variables:

```bash
# Device selection (auto, cpu, cuda)
export MARIAN__INFERENCE_DEVICE=auto

# Device index for GPU
export MARIAN__DEVICE_INDEX=0

# Quantization type (int8, float16, float32)
export MARIAN__COMPUTE_TYPE=int8
```

## Performance Optimization

### Latency Targets

- **Text Translation**: < 100ms for short texts
- **Real-time Audio**: < 400ms end-to-end (STT + Translation + TTS)

### Tips

1. **Pre-load Models**: Set `MARIAN__MODEL_TTL=-1` to keep models in memory
2. **Use GPU**: Set `MARIAN__INFERENCE_DEVICE=cuda` for faster inference
3. **Batch Requests**: Group translations when possible
4. **Choose Appropriate Models**: Smaller models are faster but may be less accurate

## Troubleshooting

### Model Download Issues

If model download fails:

```bash
# Manually download using huggingface-cli
huggingface-cli download Helsinki-NLP/opus-mt-en-zh

# Check HuggingFace cache
ls ~/.cache/huggingface/hub/
```

### Memory Issues

If you run out of memory:

```bash
# Reduce model TTL to unload models faster
export MARIAN__MODEL_TTL=60

# Use CPU instead of GPU
export MARIAN__INFERENCE_DEVICE=cpu
```

### Translation Quality

For better translation quality:

1. Use larger models (e.g., `opus-mt-tc-big` variants)
2. Ensure proper text preprocessing
3. Consider post-processing for better output

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  /v1/audio/translations (REST API)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Translation Client (ASGI)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     MarianMT Executor & Manager     │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│ CTranslate2 │ │ Transformers│
│  Translator │ │  Tokenizer  │
└─────────────┘ └─────────────┘
```

## Configuration Reference

All MarianMT settings can be configured via environment variables with the `MARIAN__` prefix:

```bash
# Model TTL
MARIAN__MODEL_TTL=300

# Device configuration
MARIAN__INFERENCE_DEVICE=auto
MARIAN__DEVICE_INDEX=0

# Quantization
MARIAN__COMPUTE_TYPE=int8
```

For more details, see `src/speaches/config.py`.

## Additional Resources

- [HuggingFace MarianMT Models](https://huggingface.co/Helsinki-NLP)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)
- [Speaches API Documentation](https://speaches.ai/api)

## Removed Features

The following features have been removed in this translation-focused version:

- ❌ `/v1/chat/completions` endpoint (LLM chat)
- ❌ External LLM dependencies (OpenAI/Ollama)
- ❌ Gradio Audio Chat UI

Retained features:

- ✅ `/v1/audio/transcriptions` (STT)
- ✅ `/v1/audio/speech` (TTS)
- ✅ Realtime API (WebSocket/WebRTC)
- ✅ Gradio UI (STT + TTS tabs)
