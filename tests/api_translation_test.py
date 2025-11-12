import pytest
from fastapi.testclient import TestClient

from speaches.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_translation_endpoint_basic(client):
    """Test basic translation endpoint functionality."""
    response = client.post(
        "/v1/audio/translations",
        data={
            "text": "Hello, how are you?",
            "model": "Helsinki-NLP/opus-mt-en-zh",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "model" in data
    assert "source_language" in data
    assert "target_language" in data
    assert data["model"] == "Helsinki-NLP/opus-mt-en-zh"
    assert data["source_language"] == "en"
    assert data["target_language"] == "zh"
    assert len(data["text"]) > 0


def test_translation_endpoint_missing_text(client):
    """Test translation endpoint with missing text parameter."""
    response = client.post(
        "/v1/audio/translations",
        data={
            "model": "Helsinki-NLP/opus-mt-en-zh",
        },
    )
    assert response.status_code == 422


def test_translation_endpoint_missing_model(client):
    """Test translation endpoint with missing model parameter."""
    response = client.post(
        "/v1/audio/translations",
        data={
            "text": "Hello, how are you?",
        },
    )
    assert response.status_code == 422


def test_translation_endpoint_custom_languages(client):
    """Test translation endpoint with custom source and target languages."""
    response = client.post(
        "/v1/audio/translations",
        data={
            "text": "Hello, how are you?",
            "model": "Helsinki-NLP/opus-mt-en-zh",
            "source_language": "en",
            "target_language": "zh",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source_language"] == "en"
    assert data["target_language"] == "zh"


def test_translation_endpoint_empty_text(client):
    """Test translation endpoint with empty text."""
    response = client.post(
        "/v1/audio/translations",
        data={
            "text": "",
            "model": "Helsinki-NLP/opus-mt-en-zh",
        },
    )
    # Should still return 200 but with empty translation
    assert response.status_code == 200


@pytest.mark.skip(reason="Requires actual model download and conversion")
def test_translation_endpoint_real_model(client):
    """Test translation endpoint with a real model (requires model download)."""
    response = client.post(
        "/v1/audio/translations",
        data={
            "text": "Hello, world!",
            "model": "Helsinki-NLP/opus-mt-en-es",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "Hola" in data["text"] or "hola" in data["text"].lower()
