from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faster_whisper import BatchedInferencePipeline, WhisperModel
import faster_whisper.transcribe
import huggingface_hub
import openai.types.audio
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.text_utils import segments_to_srt, segments_to_vtt

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path

    from openai.types import AudioResponseFormat

    from speaches.api_types import TranscriptionSegment
    from speaches.config import (
        WhisperConfig,
    )
    from speaches.executors.shared.handler_protocol import (
        AudioTranslationRequest,
        AudioTranslationResponse,
        NonStreamingTranscriptionResponse,
        StreamingTranscriptionEvent,
        TranscriptionRequest,
    )

LIBRARY_NAME = "ctranslate2"
TASK_NAME_TAG = "automatic-speech-recognition"

logger = logging.getLogger(__name__)

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


def segments_to_text(segments: Iterable[TranscriptionSegment]) -> str:
    return " ".join(segment.text for segment in segments)


def segments_to_transcription_response(
    segments: list[TranscriptionSegment],
    transcription_info: faster_whisper.transcribe.TranscriptionInfo,
    response_format: AudioResponseFormat,
) -> NonStreamingTranscriptionResponse:
    match response_format:
        case "text":
            return (segments_to_text(segments), "text/plain")
        case "json":
            return openai.types.audio.Transcription(text=segments_to_text(segments))
        case "verbose_json":
            return openai.types.audio.TranscriptionVerbose(
                language=transcription_info.language,
                duration=transcription_info.duration,
                text=segments_to_text(segments),
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=segment.id,
                        seek=segment.seek,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        tokens=segment.tokens,
                        temperature=segment.temperature,
                        avg_logprob=segment.avg_logprob,
                        compression_ratio=segment.compression_ratio,
                        no_speech_prob=segment.no_speech_prob,
                    )
                    for segment in segments
                ],
            )
        case "vtt":
            return ("\n".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), "text/vtt")
        case "srt":
            return ("\n".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), "text/plain")
        case _:
            msg = f"Unsupported response format: {response_format}"
            raise NotImplementedError(msg)


class WhisperModelFiles(BaseModel):
    model: Path
    config: Path
    tokenizer: Path
    preprocessor_config: Path


class WhisperModelRegistry(ModelRegistry[Model, WhisperModelFiles]):
    def list_remote_models(self) -> Generator[Model, None, None]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=extract_language_list(model.card_data),
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model, None, None]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            result = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if result is None:
                continue
            model_card_data, _ = result
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=extract_language_list(model_card_data),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> WhisperModelFiles:
        model_files = list(list_model_files(model_id))

        # the necessary files are specified in `faster_whisper.transcribe`
        model_file_path = next(file_path for file_path in model_files if file_path.name == "model.bin")
        config_file_path = next(
            file_path for file_path in model_files if file_path.name == "config.json"
        )  # NOTE: I don't think this file is used
        tokenizer_file_path = next(file_path for file_path in model_files if file_path.name == "tokenizer.json")
        preprocessor_config_file_path = next(
            file_path for file_path in model_files if file_path.name == "preprocessor_config.json"
        )
        return WhisperModelFiles(
            model=model_file_path,
            config=config_file_path,
            tokenizer=tokenizer_file_path,
            preprocessor_config=preprocessor_config_file_path,
        )

    def download_model_files(self, model_id: str) -> None:
        # Taken from faster_whisper/utils.py
        allow_patterns = [
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ]
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=[*allow_patterns, "README.md"]
        )


whisper_model_registry = WhisperModelRegistry(hf_model_filter=hf_model_filter)


class WhisperModelManager(BaseModelManager[WhisperModel]):
    def __init__(self, ttl: int, whisper_config: WhisperConfig) -> None:
        super().__init__(ttl)
        self.whisper_config = whisper_config

    def _load_fn(self, model_id: str) -> WhisperModel:
        return WhisperModel(
            model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
        )

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)

    def handle_non_streaming_transcription_request(
        self, request: TranscriptionRequest, **_kwargs
    ) -> NonStreamingTranscriptionResponse:
        with self.load_model(request.model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper)

            segments_iter, transcription_info = whisper_model.transcribe(
                request.audio.data,
                task="transcribe",
                language=request.language,
                initial_prompt=request.prompt,
                word_timestamps="word" in request.timestamp_granularities,
                temperature=request.temperature,
                vad_filter=request.vad_options is not None,
                hotwords=request.hotwords,
                without_timestamps=request.without_timestamps,
            )

            from speaches.api_types import TranscriptionSegment

            segments = list(TranscriptionSegment.from_faster_whisper_segments(segments_iter))

            return segments_to_transcription_response(segments, transcription_info, request.response_format)

    def handle_streaming_transcription_request(
        self, request: TranscriptionRequest, **_kwargs
    ) -> Generator[StreamingTranscriptionEvent]:
        with self.load_model(request.model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper)

            segments_iter, _transcription_info = whisper_model.transcribe(
                request.audio.data,
                task="transcribe",
                language=request.language,
                initial_prompt=request.prompt,
                word_timestamps="word" in request.timestamp_granularities,
                temperature=request.temperature,
                vad_filter=request.vad_options is not None,
                hotwords=request.hotwords,
                without_timestamps=request.without_timestamps,
            )

            from speaches.api_types import TranscriptionSegment

            full_text = ""
            for segment in TranscriptionSegment.from_faster_whisper_segments(segments_iter):
                full_text += segment.text
                yield openai.types.audio.TranscriptionTextDeltaEvent(
                    type="transcript.text.delta",
                    delta=segment.text,
                    text=full_text,
                )

            yield openai.types.audio.TranscriptionTextDoneEvent(
                type="transcript.text.done",
                text=full_text,
            )

    def handle_audio_translation_request(self, request: AudioTranslationRequest, **_kwargs) -> AudioTranslationResponse:
        with self.load_model(request.model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper)

            segments_iter, transcription_info = whisper_model.transcribe(
                request.audio.data,
                task="translate",
                initial_prompt=request.prompt,
                temperature=request.temperature,
                vad_filter=request.vad_options is not None,
            )

            from speaches.api_types import TranscriptionSegment

            segments = list(TranscriptionSegment.from_faster_whisper_segments(segments_iter))

            match request.response_format:
                case "text":
                    return (segments_to_text(segments), "text/plain")
                case "json":
                    return openai.types.audio.Translation(text=segments_to_text(segments))
                case "verbose_json":
                    return openai.types.audio.TranslationVerbose(
                        language=transcription_info.language,
                        duration=transcription_info.duration,
                        text=segments_to_text(segments),
                        segments=[
                            openai.types.audio.TranscriptionSegment(
                                id=segment.id,
                                seek=segment.seek,
                                start=segment.start,
                                end=segment.end,
                                text=segment.text,
                                tokens=segment.tokens,
                                temperature=segment.temperature,
                                avg_logprob=segment.avg_logprob,
                                compression_ratio=segment.compression_ratio,
                                no_speech_prob=segment.no_speech_prob,
                            )
                            for segment in segments
                        ],
                    )
                case "vtt":
                    return ("\n".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), "text/vtt")
                case "srt":
                    return ("\n".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), "text/plain")
                case _:
                    msg = f"Unsupported response format: {request.response_format}"
                    raise NotImplementedError(msg)
