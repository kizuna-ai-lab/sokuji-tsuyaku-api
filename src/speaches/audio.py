from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, BinaryIO, cast

import numpy as np
import soundfile as sf

from speaches.config import SAMPLES_PER_SECOND

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from speaches.routers.speech import ResponseFormat

logger = logging.getLogger(__name__)


# NOTE: `signal.resample_poly` **might** be a better option for resampling audio data
def resample_audio_data(
    data: np.typing.NDArray[np.float32], sample_rate: int, target_sample_rate: int
) -> np.typing.NDArray[np.float32]:
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    return np.interp(np.linspace(0, len(data), target_length), np.arange(len(data)), data).astype(np.float32)


# aip 'Write a function `resample_audio` which would take in RAW PCM 16-bit signed, little-endian audio data represented as bytes (`audio_bytes`) and resample it (either downsample or upsample) from `sample_rate` to `target_sample_rate` using numpy'
def resample_audio(audio_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_data) / sample_rate
    target_length = int(duration * target_sample_rate)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), target_length, endpoint=False), np.arange(len(audio_data)), audio_data
    )
    return resampled_data.astype(np.int16).tobytes()


def convert_audio_format(
    audio_bytes: bytes,
    sample_rate: int,
    audio_format: ResponseFormat,
    format: str = "RAW",  # noqa: A002
    channels: int = 1,
    subtype: str = "PCM_16",
    endian: str = "LITTLE",
) -> bytes:
    # NOTE: the default dtype is float64. Should something else be used? Would that improve performance?
    data, _ = sf.read(
        io.BytesIO(audio_bytes),
        samplerate=sample_rate,
        format=format,
        channels=channels,
        subtype=subtype,
        endian=endian,
    )
    converted_audio_bytes_buffer = io.BytesIO()
    sf.write(converted_audio_bytes_buffer, data, samplerate=sample_rate, format=audio_format)
    return converted_audio_bytes_buffer.getvalue()


def audio_samples_from_file(file: BinaryIO, sample_rate: int) -> NDArray[np.float32]:
    audio_data, _sample_rate = sf.read(
        file,
        format="RAW",
        channels=1,
        samplerate=sample_rate,
        subtype="PCM_16",
        dtype="float32",
        endian="LITTLE",
    )
    return cast("NDArray[np.float32]", audio_data)


class Audio:
    def __init__(
        self,
        data: NDArray[np.float32],
        sample_rate: int = SAMPLES_PER_SECOND,
    ) -> None:
        self.data = data
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        return f"Audio(duration={self.duration:.2f}s, sample_rate={self.sample_rate}Hz, samples={len(self.data)})"

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    @property
    def size_in_bits(self) -> int:
        return self.data.nbytes * 8

    @property
    def size_in_bytes(self) -> int:
        return self.data.nbytes

    @property
    def size_in_kb(self) -> float:
        return self.size_in_bytes / 1024.0

    @property
    def size_in_mb(self) -> float:
        return self.size_in_bytes / (1024.0 * 1024.0)

    def extend(self, data: NDArray[np.float32]) -> None:
        self.data = np.append(self.data, data)

    def as_bytes(self) -> bytes:
        # Clip to [-1.0, 1.0] to avoid overflow and scale to int16 range
        audio = (self.data * 32767).astype(np.int16)
        return audio.tobytes()

    def to_base64(self) -> str:
        import base64

        audio_bytes = self.as_bytes()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def resample(self, target_sample_rate: int) -> Audio:
        if self.sample_rate == target_sample_rate:
            return self
        self.data = resample_audio_data(self.data, self.sample_rate, target_sample_rate)
        self.sample_rate = target_sample_rate
        return self

    @classmethod
    def concatenate(cls, audios: list[Audio]) -> Audio:
        if not audios:
            msg = "No audio segments to concatenate"
            raise ValueError(msg)
        sample_rate = audios[0].sample_rate
        for audio in audios:
            if audio.sample_rate != sample_rate:
                msg = "All audio segments must have the same sample rate to concatenate"
                raise ValueError(msg)
        concatenated_data = np.concatenate([audio.data for audio in audios])
        return Audio(concatenated_data, sample_rate=sample_rate)
