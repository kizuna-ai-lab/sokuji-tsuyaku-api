# Speaches

`speaches` is an OpenAI API-compatible server for **real-time translation** with support for streaming transcription and speech generation. Translation is powered by [MarianMT](https://huggingface.co/Helsinki-NLP), speech-to-text by [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and text-to-speech by [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M). This project aims to be Ollama, but for TTS/STT/Translation models.

**📢 Version 0.2.0** - **BREAKING CHANGES**: Chat completion features have been removed. This version focuses on translation services. See [CHANGELOG.md](CHANGELOG.md) and [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) for details.

See the documentation for installation instructions and usage: [speaches.ai](https://speaches.ai/)

## Features

- **OpenAI API compatible**: All tools and SDKs that work with OpenAI's API should work with `speaches`
- **Real-time Translation**:
  - Text-to-text translation using MarianMT (1000+ language pairs)
  - Real-time audio translation: STT → Translation → TTS
  - WebSocket and WebRTC support for low-latency translation
- **Translation API** (`/v1/audio/translations`):
  - Translate text between any supported language pair
  - Automatic model download and conversion
  - TTL-based model caching
- **Speech-to-Text**: Powered by faster-whisper with streaming support
- **Text-to-Speech**: Using `kokoro` (Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models
- **Dynamic Model Loading**: Specify which model you want to use in the request and it will be loaded automatically, then unloaded after inactivity
- **GPU and CPU Support**: Configurable inference device
- **Docker Deployment**: [Deployable via Docker Compose / Docker](https://speaches.ai/installation/)
- **[Realtime API](https://speaches.ai/usage/realtime-api)**: WebSocket and WebRTC for real-time audio translation
- **[Highly Configurable](https://speaches.ai/configuration/)**: Environment-based configuration

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/speaches-ai/speaches.git
cd speaches

# Update uv and install dependencies
uv self update
uv sync

# Start the server
uvicorn speaches.main:create_app --factory --host 0.0.0.0 --port 8000
```

### Basic Translation Example

```bash
# Translate English to Chinese
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -F "text=Hello, how are you?" \
  -F "model=Helsinki-NLP/opus-mt-en-zh"

# Response: {"text": "你好，你好吗？", "model": "Helsinki-NLP/opus-mt-en-zh", ...}
```

### Using Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Translate text
translation = client.audio.translations.create(
    file=("text.txt", "Hello, world!".encode()),
    model="Helsinki-NLP/opus-mt-en-es"
)

print(translation.text)  # ¡Hola, mundo!
```

## Supported Language Pairs

MarianMT supports 1000+ language pairs. Common examples:

| Model | Languages | Example |
|-------|-----------|---------|
| `Helsinki-NLP/opus-mt-en-zh` | English → Chinese | Hello → 你好 |
| `Helsinki-NLP/opus-mt-en-es` | English → Spanish | Hello → Hola |
| `Helsinki-NLP/opus-mt-en-fr` | English → French | Hello → Bonjour |
| `Helsinki-NLP/opus-mt-zh-en` | Chinese → English | 你好 → Hello |
| `Helsinki-NLP/opus-mt-es-en` | Spanish → English | Hola → Hello |

Find more models on [HuggingFace](https://huggingface.co/Helsinki-NLP).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/translations` | POST | Translate text between languages |
| `/v1/audio/transcriptions` | POST | Speech-to-text transcription |
| `/v1/audio/speech` | POST | Text-to-speech synthesis |
| `/v1/realtime` | WebSocket | Real-time audio translation (WebSocket) |
| `/v1/realtime` | POST | Real-time audio translation (WebRTC) |
| `/v1/models` | GET | List available models |

## Documentation

- **[Translation Guide](TRANSLATION_GUIDE.md)**: Complete guide to using translation features
- **[Migration Guide](MIGRATION_SUMMARY.md)**: Migrating from version 0.1.0 to 0.2.0
- **[Changelog](CHANGELOG.md)**: Version history and breaking changes
- **[Examples](examples/translation_example.py)**: Python usage examples
- **[Full Documentation](https://speaches.ai/)**: Installation, configuration, and usage

## Configuration

Configure MarianMT via environment variables:

```bash
# Model time-to-live (seconds)
export MARIAN__MODEL_TTL=300  # Default: 300 (5 minutes)

# Device selection
export MARIAN__INFERENCE_DEVICE=auto  # auto, cpu, cuda

# Quantization type
export MARIAN__COMPUTE_TYPE=int8  # int8, float16, float32
```

See [TRANSLATION_GUIDE.md](TRANSLATION_GUIDE.md) for complete configuration options.

## Performance

- **Text Translation**: < 100ms for short texts (after model load)
- **Model Load**: 2-5 seconds (first time, includes download/conversion)
- **Real-time Audio**: < 400ms end-to-end (STT + Translation + TTS)
- **Model Size**: ~200MB per model (int8 quantization)

## What's New in 0.2.0

### ✨ Added
- **MarianMT Translation**: 1000+ language pairs
- **Translation API**: `/v1/audio/translations` endpoint
- **Real-time Translation**: Updated Realtime API for audio translation

### ❌ Removed (Breaking Changes)
- **Chat Completions**: `/v1/chat/completions` endpoint removed
- **Audio Chat UI**: Gradio audio chat tab removed
- **External LLM**: No longer requires OpenAI/Ollama

See [CHANGELOG.md](CHANGELOG.md) for complete release notes.

## Examples

### Synchronous Translation
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

translation = client.audio.translations.create(
    file=("text.txt", "Good morning!".encode()),
    model="Helsinki-NLP/opus-mt-en-fr"
)

print(translation.text)  # Bonjour!
```

### Batch Translation
```python
import asyncio
from openai import AsyncOpenAI

async def translate_batch():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    texts = ["Hello", "Thank you", "Goodbye"]
    tasks = [
        client.audio.translations.create(
            file=("text.txt", text.encode()),
            model="Helsinki-NLP/opus-mt-en-es"
        )
        for text in texts
    ]

    translations = await asyncio.gather(*tasks)
    for orig, trans in zip(texts, translations):
        print(f"{orig} → {trans.text}")

asyncio.run(translate_batch())
```

See [examples/translation_example.py](examples/translation_example.py) for more examples.

## Contributing

Please create an issue if you find a bug, have a question, or a feature suggestion.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [MarianMT](https://huggingface.co/Helsinki-NLP) for translation models
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and [Piper](https://github.com/rhasspy/piper) for text-to-speech
- [CTranslate2](https://opennmt.net/CTranslate2/) for optimized inference
