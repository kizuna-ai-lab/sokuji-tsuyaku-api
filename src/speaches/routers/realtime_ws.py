import asyncio
import logging

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketException,
    status,
)

from speaches.dependencies import (
    ConfigDependency,
    ExecutorRegistryDependency,
    SpeechClientDependency,
    TranscriptionClientDependency,
    TranslationClientDependency,
)
from speaches.realtime.context import SessionContext
from speaches.realtime.conversation_event_router import event_router as conversation_event_router
from speaches.realtime.event_router import EventRouter
from speaches.realtime.input_audio_buffer_event_router import (
    event_router as input_audio_buffer_event_router,
)
from speaches.realtime.message_manager import WsServerMessageManager
from speaches.realtime.response_event_router import event_router as response_event_router
from speaches.realtime.session import OPENAI_REALTIME_SESSION_DURATION_SECONDS, create_session_object_configuration
from speaches.realtime.session_event_router import event_router as session_event_router
from speaches.realtime.utils import parse_model_parameter, task_done_callback, verify_websocket_api_key
from speaches.types.realtime import SessionCreatedEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

event_router = EventRouter()
event_router.include_router(conversation_event_router)
event_router.include_router(input_audio_buffer_event_router)
event_router.include_router(response_event_router)
event_router.include_router(session_event_router)


async def event_listener(ctx: SessionContext) -> None:
    try:
        async with asyncio.TaskGroup() as tg:
            async for event in ctx.pubsub.poll():
                # logger.debug(f"Received event: {event.type}")

                task = tg.create_task(event_router.dispatch(ctx, event))
                task.add_done_callback(task_done_callback)
    except asyncio.CancelledError:
        logger.info("Event listener task cancelled")
        raise
    finally:
        logger.info("Event listener task finished")


@router.websocket("/v1/realtime")
async def realtime(
    ws: WebSocket,
    model: str,
    config: ConfigDependency,
    transcription_client: TranscriptionClientDependency,
    translation_client: TranslationClientDependency,
    speech_client: SpeechClientDependency,
    executor_registry: ExecutorRegistryDependency,
    intent: str = "conversation",
    language: str | None = None,
) -> None:
    """OpenAI Realtime API compatible WebSocket endpoint.

    Args:
        model: Model string in format "stt_model|tts_model|translation_model"
               Example: "Systran/faster-distil-whisper-small.en|speaches-ai/Kokoro-82M-v1.0-ONNX|Helsinki-NLP/opus-mt-en-zh"
        intent: Session intent (deprecated, kept for compatibility)
        language: Optional language code for transcription auto-detection

    References:
    - https://platform.openai.com/docs/guides/realtime/overview
    - https://platform.openai.com/docs/api-reference/realtime-server-events/session/update

    """
    # Manually verify WebSocket authentication before accepting connection
    try:
        await verify_websocket_api_key(ws, config)
    except WebSocketException:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    await ws.accept()

    # Parse model parameter at API layer
    stt_model, tts_model, translation_model = parse_model_parameter(model)
    logger.info(f"Accepted websocket connection: STT={stt_model}, TTS={tts_model}, Translation={translation_model}")

    ctx = SessionContext(
        transcription_client=transcription_client,
        translation_client=translation_client,
        speech_client=speech_client,
        vad_model_manager=executor_registry.vad.model_manager,
        session=create_session_object_configuration(stt_model, tts_model, translation_model, intent, language),
    )
    message_manager = WsServerMessageManager(ctx.pubsub)
    async with asyncio.TaskGroup() as tg:
        event_listener_task = tg.create_task(event_listener(ctx), name="event_listener")
        async with asyncio.timeout(OPENAI_REALTIME_SESSION_DURATION_SECONDS):
            mm_task = asyncio.create_task(message_manager.run(ws))
            # HACK: a tiny delay to ensure the message_manager.run() task is started. Otherwise, the `SessionCreatedEvent` will not be sent, as it's published before the `sender` task subscribes to the pubsub.
            await asyncio.sleep(0.001)
            ctx.pubsub.publish_nowait(SessionCreatedEvent(session=ctx.session))
            await mm_task
        event_listener_task.cancel()

    logger.info(f"Finished handling '{ctx.session.id}' session")
