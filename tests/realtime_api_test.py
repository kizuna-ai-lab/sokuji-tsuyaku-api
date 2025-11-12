"""Tests for Realtime API WebSocket functionality."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest
from pytest_mock import MockerFixture

from speaches.config import Config, WhisperConfig
from speaches.main import create_app
from speaches.realtime.session import create_session_object_configuration
from speaches.realtime.utils import parse_model_parameter, verify_websocket_api_key


class TestRealtimeWebSocketAuthentication:
    """Test WebSocket authentication functionality."""

    @pytest.mark.asyncio
    async def test_websocket_auth_with_bearer_token(self) -> None:
        """Test WebSocket authentication with Authorization Bearer token."""
        # Mock WebSocket
        mock_ws = MagicMock()
        mock_ws.headers = {"authorization": "Bearer test-api-key"}
        mock_ws.query_params = {}

        # Mock config with API key
        config = Config(
            api_key=SecretStr("test-api-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        # Should not raise exception
        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_with_x_api_key(self) -> None:
        """Test WebSocket authentication with X-API-Key header."""
        mock_ws = MagicMock()
        mock_ws.headers = {"x-api-key": "test-api-key"}
        mock_ws.query_params = {}

        config = Config(
            api_key=SecretStr("test-api-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_with_query_param(self) -> None:
        """Test WebSocket authentication with api_key query parameter."""
        mock_ws = MagicMock()
        mock_ws.headers = {}
        mock_ws.query_params = {"api_key": "test-api-key"}

        config = Config(
            api_key=SecretStr("test-api-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_no_api_key_configured(self) -> None:
        """Test WebSocket authentication when no API key is configured."""
        mock_ws = MagicMock()
        mock_ws.headers = {}
        mock_ws.query_params = {}

        config = Config(
            api_key=None,  # No API key configured
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        # Should not raise exception when no API key is configured
        await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_invalid_key(self) -> None:
        """Test WebSocket authentication with invalid API key."""
        from fastapi import WebSocketException

        mock_ws = MagicMock()
        mock_ws.headers = {"authorization": "Bearer wrong-key"}
        mock_ws.query_params = {}

        config = Config(
            api_key=SecretStr("correct-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        with pytest.raises(WebSocketException):
            await verify_websocket_api_key(mock_ws, config)

    @pytest.mark.asyncio
    async def test_websocket_auth_missing_key(self) -> None:
        """Test WebSocket authentication with missing API key."""
        from fastapi import WebSocketException

        mock_ws = MagicMock()
        mock_ws.headers = {}
        mock_ws.query_params = {}

        config = Config(
            api_key=SecretStr("required-key"),
            whisper=WhisperConfig(),
            enable_ui=False,
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        with pytest.raises(WebSocketException):
            await verify_websocket_api_key(mock_ws, config)


class TestRealtimeSessionConfiguration:
    """Test session configuration with new model format: stt|tts|translation."""

    def test_model_parameter_parsing(self) -> None:
        """Test basic model parameter parsing."""
        stt, tts, trans = parse_model_parameter(
            "Systran/faster-distil-whisper-small.en|speaches-ai/Kokoro-82M-v1.0-ONNX|Helsinki-NLP/opus-mt-en-zh"
        )
        session = create_session_object_configuration(stt, tts, trans)

        assert session.input_audio_transcription.model == "Systran/faster-distil-whisper-small.en"
        assert session.speech_model == "speaches-ai/Kokoro-82M-v1.0-ONNX"
        assert session.translation_model == "Helsinki-NLP/opus-mt-en-zh"
        assert session.turn_detection is not None and session.turn_detection.create_response is True
        assert session.input_audio_transcription.language is None

    def test_model_with_language(self) -> None:
        """Test model parameter with language specification."""
        stt, tts, trans = parse_model_parameter("whisper-1|kokoro|marian")
        session = create_session_object_configuration(stt, tts, trans, intent="conversation", language="ru")

        assert session.input_audio_transcription.model == "whisper-1"
        assert session.speech_model == "kokoro"
        assert session.translation_model == "marian"
        assert session.input_audio_transcription.language == "ru"

    def test_transcription_only_mode(self) -> None:
        """Test transcription-only mode configuration."""
        stt, tts, trans = parse_model_parameter("deepdml/faster-whisper-large-v3-turbo-ct2|kokoro|marian")
        session = create_session_object_configuration(stt, tts, trans, intent="transcription")

        assert session.input_audio_transcription.model == "deepdml/faster-whisper-large-v3-turbo-ct2"
        assert session.turn_detection is not None and session.turn_detection.create_response is False

    def test_invalid_model_format_too_few_parts(self) -> None:
        """Test that invalid model format raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Invalid model format: expected 'stt\\|tts\\|translation'"):
            parse_model_parameter("only-one-model")

    def test_invalid_model_format_too_many_parts(self) -> None:
        """Test that model with too many parts raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Invalid model format"):
            parse_model_parameter("model1|model2|model3|model4")

    def test_invalid_model_format_empty_part(self) -> None:
        """Test that empty model parts raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="all three models must be non-empty"):
            parse_model_parameter("whisper||marian")

    def test_session_configuration_logging(self, caplog) -> None:  # noqa: ANN001
        """Test that session configuration produces appropriate logging."""
        with caplog.at_level("INFO"):
            create_session_object_configuration(stt_model="stt", tts_model="tts", translation_model="translation")

        assert "Using models:" in caplog.text
        assert "STT=stt" in caplog.text
        assert "TTS=tts" in caplog.text
        assert "Translation=translation" in caplog.text


class TestRealtimeWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    def test_websocket_endpoint_exists(self, mocker: MockerFixture) -> None:
        """Test that WebSocket endpoint is properly registered."""
        from pydantic import SecretStr

        from speaches.config import Config, WhisperConfig

        # Create config without UI to avoid gradio dependency
        config = Config(
            api_key=None,
            whisper=WhisperConfig(),
            enable_ui=False,  # Disable UI to avoid gradio import
            chat_completion_base_url="https://api.openai.com/v1",
            chat_completion_api_key=SecretStr("test-key"),
        )

        # Mock get_config before create_app is called
        mocker.patch("speaches.main.get_config", return_value=config)
        mocker.patch("speaches.dependencies.get_config", return_value=config)

        app = create_app()
        client = TestClient(app)

        # Test that the endpoint exists (will fail auth but endpoint should be found)
        with client.websocket_connect("/v1/realtime?model=stt|tts|translation") as _:
            # Connection should be closed due to auth failure, but endpoint exists
            pass


class TestRealtimeAPICompatibility:
    """Test model format compatibility."""

    def test_session_structure(self) -> None:
        """Test that session structure is correct."""
        session = create_session_object_configuration(
            stt_model="stt-model", tts_model="tts-model", translation_model="translation-model"
        )

        # Check required fields exist
        assert hasattr(session, "id")
        assert hasattr(session, "modalities")
        assert hasattr(session, "input_audio_transcription")
        assert hasattr(session, "turn_detection")
        assert hasattr(session, "speech_model")
        assert hasattr(session, "voice")
        assert hasattr(session, "translation_model")

        # Check types
        assert isinstance(session.modalities, list)
        assert "audio" in session.modalities
        assert "text" in session.modalities

        # Check model values
        assert session.input_audio_transcription.model == "stt-model"
        assert session.speech_model == "tts-model"
        assert session.translation_model == "translation-model"
