from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.beta.realtime.error_event import Error

from speaches.realtime.event_router import EventRouter
from speaches.realtime.utils import parse_model_parameter
from speaches.types.realtime import (
    NOT_GIVEN,
    ErrorEvent,
    Session,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    TurnDetection,
)

if TYPE_CHECKING:
    from speaches.realtime.context import SessionContext

logger = logging.getLogger(__name__)

event_router = EventRouter()


def update_dict(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict):
            original[key] = update_dict(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def unsupported_field_error(field: str) -> ErrorEvent:
    return ErrorEvent(
        error=Error(
            type="invalid_request_error",
            message=f"Specifying `{field}` is not supported. The server either does not support this field or it is not configurable.",
        )
    )


@event_router.register("session.update")
def handle_session_update_event(ctx: SessionContext, event: SessionUpdateEvent) -> None:
    if event.session.input_audio_format != NOT_GIVEN:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.input_audio_format"))
    if event.session.output_audio_format != NOT_GIVEN:
        ctx.pubsub.publish_nowait(unsupported_field_error("session.output_audio_format"))
    if (
        event.session.turn_detection is not None
        and isinstance(event.session.turn_detection, TurnDetection)
        and event.session.turn_detection.prefix_padding_ms != NOT_GIVEN
    ):
        ctx.pubsub.publish_nowait(unsupported_field_error("session.turn_detection.prefix_padding_ms"))

    # Handle composite model parameter if provided
    if event.session.model != NOT_GIVEN:
        try:
            stt_model, tts_model, translation_model = parse_model_parameter(event.session.model)
            logger.info(f"Parsed model parameter: STT={stt_model}, TTS={tts_model}, Translation={translation_model}")

            # Update the individual model fields
            if event.session.speech_model == NOT_GIVEN:
                event.session.speech_model = tts_model
            if event.session.translation_model == NOT_GIVEN:
                event.session.translation_model = translation_model
            if event.session.input_audio_transcription == NOT_GIVEN:
                event.session.input_audio_transcription = ctx.session.input_audio_transcription.model_copy()
            if hasattr(event.session.input_audio_transcription, "model"):
                event.session.input_audio_transcription.model = stt_model
        except ValueError as e:
            logger.error(f"Failed to parse model parameter: {e}")
            ctx.pubsub.publish_nowait(
                ErrorEvent(
                    error=Error(
                        type="invalid_request_error",
                        message=f"Invalid model format: {e}",
                    )
                )
            )
            return

    session_dict = ctx.session.model_dump()
    session_update_dict = event.session.model_dump(
        exclude_defaults=True,
        # https://docs.pydantic.dev/latest/concepts/serialization/#advanced-include-and-exclude
        exclude={
            "input_audio_format": True,
            "output_audio_format": True,
            "turn_detection": {"prefix_padding_ms"},
            "model": True,  # Exclude the composite model field from update
        },
    )

    logger.debug(f"Applying session configuration update: {session_update_dict}")
    logger.debug(f"Session configuration before update: {session_dict}")
    updated_session = update_dict(session_dict, session_update_dict)
    logger.debug(f"Session configuration after update: {updated_session}")
    ctx.session = Session(**updated_session)

    ctx.pubsub.publish_nowait(SessionUpdatedEvent(session=ctx.session))
