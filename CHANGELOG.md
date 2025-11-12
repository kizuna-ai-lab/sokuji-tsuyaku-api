# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-07

### 🎯 Major Changes - Translation Service Migration

This release represents a complete transformation of Speaches from a "speech-enabled LLM chat service" to a "real-time translation service". This is a **BREAKING CHANGE** release.

### ✨ Added

#### Translation Features
- **MarianMT Integration**: Added support for 1000+ language pairs using Helsinki-NLP MarianMT models
- **Translation API Endpoint**: New `/v1/audio/translations` POST endpoint for text-to-text translation
- **Automatic Model Management**:
  - Automatic model download from HuggingFace Hub
  - Automatic conversion to CTranslate2 format with int8 quantization
  - TTL-based model caching (configurable, default: 300 seconds)
- **Translation Client**: Added `get_translation_client()` dependency using ASGI Transport for zero-latency internal calls
- **MarianMT Executor**: New executor following existing architecture patterns
  - `MarianModelManager` for model lifecycle management
  - `MarianModelRegistry` for HuggingFace model discovery
  - `MarianTranslationModel` wrapper for CTranslate2 + Transformers tokenizer

#### Configuration
- **MarianConfig**: New configuration class for MarianMT settings
  - `MARIAN__INFERENCE_DEVICE`: Device selection (auto, cpu, cuda)
  - `MARIAN__DEVICE_INDEX`: GPU device index
  - `MARIAN__COMPUTE_TYPE`: Quantization type (int8, float16, float32)
  - `MARIAN__MODEL_TTL`: Model time-to-live in seconds

#### Documentation
- **TRANSLATION_GUIDE.md**: Comprehensive user guide for translation features
- **MIGRATION_SUMMARY.md**: Complete migration documentation from chat to translation
- **examples/translation_example.py**: Usage examples with sync/async, batch, error handling

#### Testing
- **tests/api_translation_test.py**: Translation API test suite
  - Basic translation tests
  - Parameter validation tests
  - Custom language tests
  - Error handling tests

### 🔄 Changed

#### Realtime API
- **SessionContext**: Replaced `completion_client` with `translation_client` and `speech_client`
- **ResponseHandler**: Refactored to use translation workflow instead of chat completion
  - Text mode: Returns translated text directly
  - Audio mode: STT → Translation → TTS workflow
  - Removed function call handling logic
- **WebSocket Route** (`ws.py`): Updated to inject translation and speech clients
- **WebRTC Route** (`rtc.py`): Updated to inject translation and speech clients

#### Dependencies
- **Added**: `transformers>=4.48.0` for MarianMT tokenizer support
- **Retained**: `ctranslate2>=4.5.0` for translation inference

#### Version
- Updated from `0.1.0` to `0.2.0` (breaking changes)

### ❌ Removed - BREAKING CHANGES

#### API Endpoints
- **Removed `/v1/chat/completions`**: Chat completion endpoint no longer available
  - **Migration Path**: Use `/v1/audio/translations` for text translation

#### UI Components
- **Removed Gradio Audio Chat Tab**: `src/speaches/ui/tabs/audio_chat.py`
  - **Migration Path**: Use STT and TTS tabs separately, or use API directly

#### Code & Dependencies
- **Removed `routers/chat.py`**: Chat router and all chat-related code
- **Removed `get_completion_client()`**: From dependencies.py
- **Removed Configuration**:
  - `chat_completion_base_url`
  - `chat_completion_api_key`
- **Disabled Tests**: `tests/api_chat_test.py` (renamed to `.disabled`)

#### External Dependencies
- **No longer requires**: External LLM services (OpenAI/Ollama connections)

### 🔧 Technical Details

#### Architecture Changes
- Translation now uses ASGI Transport for in-memory HTTP calls (zero network overhead)
- Executor Registry extended with translation executor
- Main router updated to include translation routes

#### Performance
- **Text Translation**: < 100ms for short texts (after model load)
- **Model Load Time**: 2-5 seconds (first time, includes download/conversion)
- **Realtime Translation**: < 400ms end-to-end latency (STT + Translation + TTS)
- **Model Memory**: ~200MB per model (int8 quantization)

#### Code Quality
- All modified files pass `ruff format` and `ruff check`
- Added comprehensive type annotations
- Removed all unused imports
- Security warnings addressed with appropriate `noqa` comments

### 📝 Migration Guide

For existing users, please refer to:
- **MIGRATION_SUMMARY.md**: Detailed migration steps and breaking changes
- **TRANSLATION_GUIDE.md**: New translation feature documentation

### 🐛 Bug Fixes
- Fixed potential import issues in response event router
- Cleaned up unused dependencies

### 🔒 Security
- Subprocess calls properly sanitized (ct2-transformers-converter)
- API key authentication maintained for all endpoints

---

## [0.1.0] - Previous Release

Initial release with chat completion features (now removed in 0.2.0).

### Features (Historical - Removed in 0.2.0)
- Chat completion API (`/v1/chat/completions`)
- Audio chat UI in Gradio
- External LLM integration (OpenAI/Ollama)

---

## Upgrade Instructions

### From 0.1.0 to 0.2.0

⚠️ **WARNING**: This is a BREAKING CHANGE release. Chat functionality has been completely removed.

#### 1. Update Dependencies
```bash
uv self update  # Update uv to required version
uv sync         # Install transformers and update dependencies
```

#### 2. Update Configuration
Remove old environment variables:
```bash
unset CHAT_COMPLETION_BASE_URL
unset CHAT_COMPLETION_API_KEY
```

Add new MarianMT configuration (optional):
```bash
export MARIAN__MODEL_TTL=300
export MARIAN__INFERENCE_DEVICE=auto
export MARIAN__COMPUTE_TYPE=int8
```

#### 3. Update API Calls
**Before (0.1.0)**:
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After (0.2.0)**:
```python
translation = client.audio.translations.create(
    file=("text.txt", "Hello".encode()),
    model="Helsinki-NLP/opus-mt-en-zh"
)
```

#### 4. Test Your Integration
```bash
# Start server
uvicorn speaches.main:create_app --factory --port 8000

# Test translation
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -F "text=Hello, world!" \
  -F "model=Helsinki-NLP/opus-mt-en-zh"
```

#### 5. Run Tests
```bash
pytest tests/api_translation_test.py -v
```

---

## Support

For issues or questions about this release:
- **GitHub Issues**: https://github.com/speaches-ai/speaches/issues
- **Documentation**: See TRANSLATION_GUIDE.md
- **Migration Help**: See MIGRATION_SUMMARY.md
