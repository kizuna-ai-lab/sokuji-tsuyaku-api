from collections.abc import Generator
from typing import Protocol

import numpy as np
import openai.types.audio
from pydantic import BaseModel, ConfigDict

from speaches.api_types import TimestampGranularities
from speaches.audio import Audio

# TODO: Update to silero_vad_v5 when integrated from upstream
# For now, using faster_whisper VAD types
try:
    from speaches.executors.silero_vad_v5 import SpeechTimestamp, VadOptions
except ImportError:
    from typing import TypedDict

    from faster_whisper.vad import VadOptions

    class SpeechTimestamp(TypedDict):
        start: int
        end: int


MimeType = str


class SpeakerEmbeddingRequest(BaseModel):
    model_id: str
    audio: Audio

    model_config = ConfigDict(arbitrary_types_allowed=True)


type SpeakerEmbeddingResponse = np.typing.NDArray[np.float32]


class SpeakerEmbeddingHandler(Protocol):
    def handle_speaker_embedding_request(
        self, request: SpeakerEmbeddingRequest, **kwargs
    ) -> SpeakerEmbeddingResponse: ...


class SpeechRequest(BaseModel):
    model: str
    voice: str
    text: str
    speed: float


SpeechResponse = Generator[Audio]


class SpeechHandler(Protocol):
    def handle_speech_request(self, request: SpeechRequest, **kwargs) -> SpeechResponse: ...


class VadRequest(BaseModel):
    audio: Audio
    vad_options: VadOptions
    model_id: str = "silero_vad_v5"
    sampling_rate: int = 16000

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VadHandler(Protocol):
    def handle_vad_request(self, request: VadRequest, **kwargs) -> list[SpeechTimestamp]: ...


class TranscriptionRequest(BaseModel):
    audio: Audio
    model: str
    stream: bool = False
    language: str | None = None
    prompt: str | None = None
    response_format: openai.types.AudioResponseFormat = "json"
    temperature: float = 0.0
    hotwords: str | None = None
    timestamp_granularities: TimestampGranularities
    speech_segments: list[SpeechTimestamp]
    vad_options: VadOptions
    without_timestamps: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


NonStreamingTranscriptionResponse = (
    tuple[str, MimeType] | openai.types.audio.Transcription | openai.types.audio.TranscriptionVerbose
)
StreamingTranscriptionEvent = (
    openai.types.audio.TranscriptionTextDeltaEvent | openai.types.audio.TranscriptionTextDoneEvent
)


class TranscriptionHandler(Protocol):
    def handle_non_streaming_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse: ...

    def handle_streaming_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> Generator[StreamingTranscriptionEvent]: ...

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)


class AudioTranslationRequest(BaseModel):
    """Audio-to-text translation (e.g., Whisper audio translation)."""

    audio: Audio
    model: str
    prompt: str | None = None
    response_format: openai.types.AudioResponseFormat = "json"
    temperature: float = 0.0
    speech_segments: list[SpeechTimestamp]
    vad_options: VadOptions

    model_config = ConfigDict(arbitrary_types_allowed=True)


AudioTranslationResponse = tuple[str, MimeType] | openai.types.audio.Translation | openai.types.audio.TranslationVerbose


class AudioTranslationHandler(Protocol):
    """Handler for audio-to-text translation (e.g., Whisper)."""

    def handle_audio_translation_request(
        self, request: AudioTranslationRequest, **kwargs
    ) -> AudioTranslationResponse: ...


# Text Translation Handler for MarianMT
class TextTranslationRequest(BaseModel):
    """Text-to-text translation request for MarianMT."""

    text: str
    model: str
    source_language: str | None = None
    target_language: str | None = None


class TextTranslationResponse(BaseModel):
    """Text-to-text translation response."""

    text: str
    model: str
    source_language: str
    target_language: str


class TextTranslationHandler(Protocol):
    """Handler for text-to-text translation (e.g., MarianMT)."""

    def handle_text_translation_request(self, request: TextTranslationRequest, **kwargs) -> TextTranslationResponse: ...
