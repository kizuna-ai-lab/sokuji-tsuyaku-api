# Sokuji Tsuyaku API

Real-time speech translation API server for Sokuji (即時通訳APIサーバー)

## About

Sokuji Tsuyaku API is a specialized real-time speech translation service forked from [speaches-ai/speaches](https://github.com/speaches-ai/speaches) (a TTS/STT server, upstream is now v0.8.3) and extensively modified to focus on real-time translation. We've transformed the original audio processing capabilities into a streamlined STT → Translation → TTS pipeline optimized for low-latency interpretation scenarios.

This project focuses on providing instant translation capabilities through WebSocket and WebRTC protocols, enabling seamless cross-language communication in real-time.

## Key Features

- **Real-time Translation Pipeline**: Streamlined STT → Translation → TTS workflow
- **OpenAI-compatible API**: Works with existing OpenAI client libraries and tools
- **WebSocket & WebRTC Support**: Low-latency real-time communication protocols
- **Multi-language Support**: Powered by state-of-the-art models
  - **STT**: Whisper variants through faster-whisper
  - **Translation**: MarianMT models (1000+ language pairs)
  - **TTS**: Kokoro and Piper models
- **Dynamic Model Loading**: Models loaded on-demand and cached with TTL
- **Voice Activity Detection**: Silero VAD v5 for accurate speech detection
- **GPU Acceleration**: Optional CUDA support for improved performance
- **Modular Architecture**: Easily swap models for different language pairs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kizuna-ai-lab/sokuji-tsuyaku-api.git
cd sokuji-tsuyaku-api

# Install dependencies (using pip)
pip install -e .

# Or using uv (if you prefer)
uv sync

# Start the server
speaches serve --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn speaches.main:create_app --factory --host 0.0.0.0 --port 8000
```

### Basic Translation Example

```bash
# Translate English to Chinese
curl -X POST "http://localhost:8000/v1/translations" \
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

| Endpoint                    | Method    | Description                              |
|-----------------------------|-----------|------------------------------------------|
| `/v1/translations`          | POST      | Translate text between languages         |
| `/v1/audio/transcriptions`  | POST      | Speech-to-text transcription             |
| `/v1/audio/speech`          | POST      | Text-to-speech synthesis                 |
| `/v1/realtime`              | WebSocket | Real-time audio translation (WebSocket)  |
| `/v1/realtime`              | POST      | Real-time audio translation (WebRTC)     |
| `/v1/models`                | GET       | List available models                    |
| `/health`                   | GET       | Health check endpoint                    |
| `/api/ps`                   | GET       | List loaded models                       |

### WebSocket Realtime API

Connect to the realtime endpoint with model configuration:
```
ws://localhost:8000/v1/realtime?model={stt_model}|{tts_model}|{translation_model}
```

Example:
```javascript
const ws = new WebSocket(
  'ws://localhost:8000/v1/realtime?model=Systran/faster-distil-whisper-small.en|speaches-ai/Kokoro-82M-v1.0-ONNX|Helsinki-NLP/opus-mt-en-zh'
);
```

## Documentation

- **[Translation Guide](TRANSLATION_GUIDE.md)**: Complete guide to using translation features
- **[Changelog](CHANGELOG.md)**: Version history and changes
- **[Examples](examples/)**: Usage examples and sample code

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

## Differences from Upstream

This fork specializes in real-time translation scenarios with the following modifications:

- **Simplified Pipeline**: Optimized STT → Translation → TTS flow for minimal latency
- **Translation-First Design**: API and internals optimized specifically for translation use cases
- **Enhanced Model Support**: Focus on translation-specific model optimization
- **Streamlined Dependencies**: Removed unnecessary components for cleaner deployment

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

## Support

For issues and questions:
- Create an issue in this repository
- Contact the Sokuji team

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on [speaches-ai/speaches](https://github.com/speaches-ai/speaches), an excellent open-source speech processing framework. We are grateful to the speaches team for their foundational work.

### Key Technologies Used
- [MarianMT](https://huggingface.co/Helsinki-NLP) for translation models
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and [Piper](https://github.com/rhasspy/piper) for text-to-speech
- [CTranslate2](https://opennmt.net/CTranslate2/) for optimized inference
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection

---

**Sokuji** - Instant Translation for Real-time Communication
