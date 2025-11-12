import asyncio
from contextvars import Context
import logging
import random
import string
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketException, status

if TYPE_CHECKING:
    from speaches.config import Config

logger = logging.getLogger(__name__)


def generate_id_suffix() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=21))  # noqa: S311


def generate_event_id() -> str:
    return "event_" + generate_id_suffix()


def generate_conversation_id() -> str:
    return "conv_" + generate_id_suffix()


def generate_item_id() -> str:
    return "item_" + generate_id_suffix()


def generate_response_id() -> str:
    return "resp_" + generate_id_suffix()


def generate_session_id() -> str:
    return "sess_" + generate_id_suffix()


def generate_call_id() -> str:
    return "call_" + generate_id_suffix()


def task_done_callback(task: asyncio.Task, *, context: Context | None = None) -> None:  # noqa: ARG001
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info(f"Task {task.get_name()} cancelled")
    except BaseException:  # TODO: should this be `Exception` instead?
        logger.exception(f"Task {task.get_name()} failed")


def parse_model_parameter(model: str) -> tuple[str, str, str]:
    """Parse model parameter in format: stt_model|tts_model|translation_model.

    Args:
        model: Model string in format "stt|tts|translation"

    Returns:
        Tuple of (stt_model, tts_model, translation_model)

    Raises:
        ValueError: If model format is invalid (not exactly 3 parts)

    Examples:
        >>> parse_model_parameter("whisper|kokoro|marian")
        ("whisper", "kokoro", "marian")

    """
    parts = model.split("|")

    if len(parts) != 3:
        raise ValueError(f"Invalid model format: expected 'stt|tts|translation', got '{model}' with {len(parts)} parts")

    stt_model, tts_model, translation_model = parts

    if not stt_model or not tts_model or not translation_model:
        raise ValueError(f"Invalid model format: all three models must be non-empty in '{model}'")

    return (stt_model, tts_model, translation_model)


async def verify_websocket_api_key(
    websocket: WebSocket,
    config: "Config",
) -> None:
    """Verify API key for WebSocket connections.

    Supports multiple authentication methods:
    - Query parameter: ?api_key=<key>
    - Authorization header: Authorization: Bearer <key>
    - X-API-Key header: X-API-Key: <key>

    References:
    - https://platform.openai.com/docs/guides/realtime/overview

    """
    if config.api_key is None:
        return  # No API key configured, authentication not required

    # Try to get API key from query parameters first
    api_key = websocket.query_params.get("api_key")

    # If not in query params, try Authorization header
    if not api_key:
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix

    # If still no API key found, check for X-API-Key header
    if not api_key:
        api_key = websocket.headers.get("x-api-key")

    if not api_key:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="API key required")

    if api_key != config.api_key.get_secret_value():
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API key")
