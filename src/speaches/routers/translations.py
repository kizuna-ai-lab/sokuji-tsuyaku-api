import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.marian import MarianModelManager
from speaches.executors.nllb import NllbModelManager
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

    Translate text from one language to another using translation models (MarianMT or NLLB).

    Args:
        executor_registry: Registry containing translation model executors
        text: The source text to translate
        model: The translation model ID (e.g., Helsinki-NLP/opus-mt-en-zh or Xenova/nllb-200-distilled-600M)
        source_language: Source language code (required for NLLB)
        target_language: Target language code (required for NLLB)

    Returns:
        JSONResponse containing translated text and language codes

    """
    model_card_data, model_tags = get_model_card_data_or_raise(model)

    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.translation, model_tags)

    if isinstance(executor.model_manager, MarianModelManager | NllbModelManager):
        translation_request = TextTranslationRequest(
            text=text,
            model=model,
            source_language=source_language,
            target_language=target_language,
        )

        response = executor.model_manager.handle_text_translation_request(translation_request)

        return JSONResponse(
            content=response.model_dump(),
            media_type="application/json",
        )

    msg = f"Unsupported model type for {model}"
    raise HTTPException(status_code=500, detail=msg)
