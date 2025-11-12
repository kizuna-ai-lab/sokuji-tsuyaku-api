import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.marian import MarianModelManager
from speaches.executors.shared.handler_protocol import TextTranslationRequest
from speaches.model_aliases import ModelId
from speaches.routers.utils import find_executor_for_model_or_raise, get_model_card_data_or_raise

logger = logging.getLogger(__name__)

router = APIRouter(tags=["translation"])


@router.post("/v1/translations")
async def translate_text(
    executor_registry: ExecutorRegistryDependency,
    text: Annotated[str, Form()],
    model: Annotated[ModelId, Form()],
    source_language: Annotated[str | None, Form()] = None,
    target_language: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Text-to-text translation endpoint.

    Translate text from one language to another using MarianMT models.

    Args:
        executor_registry: Registry containing translation model executors
        text: The source text to translate
        model: The MarianMT model ID (e.g., Helsinki-NLP/opus-mt-en-zh)
        source_language: Optional source language code
        target_language: Optional target language code

    Returns:
        JSONResponse containing translated text and language codes

    """
    model_card_data, model_tags = get_model_card_data_or_raise(model)

    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.translation, model_tags)

    if isinstance(executor.model_manager, MarianModelManager):
        # Create translation request using the handler protocol
        translation_request = TextTranslationRequest(
            text=text,
            model=model,
            source_language=source_language,
            target_language=target_language,
        )

        # Use the handler protocol method
        response = executor.model_manager.handle_text_translation_request(translation_request)

        # Return the response as JSON
        return JSONResponse(
            content=response.model_dump(),
            media_type="application/json",
        )

    msg = f"Unsupported model type for {model}"
    raise HTTPException(status_code=500, detail=msg)
