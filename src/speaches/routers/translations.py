import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Response

from speaches.dependencies import ExecutorRegistryDependency
from speaches.executors.marian import MarianModelManager
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
) -> Response:
    """Text-to-text translation endpoint.

    Translate text from one language to another using MarianMT models.

    Parameters
    ----------
    - text: The source text to translate
    - model: The MarianMT model ID (e.g., Helsinki-NLP/opus-mt-en-zh)
    - source_language: Optional source language code
    - target_language: Optional target language code

    """
    model_card_data, model_tags = get_model_card_data_or_raise(model)

    executor = find_executor_for_model_or_raise(model, model_card_data, executor_registry.translation, model_tags)

    if isinstance(executor.model_manager, MarianModelManager):
        with executor.model_manager.load_model(model) as marian:
            translated_text = marian.translate(text)

            repo_name = model.split("/")[-1]
            if "opus-mt-" in repo_name:
                lang_part = repo_name.split("opus-mt-", 1)[1]
                parts = lang_part.split("-")
                src_lang = source_language or (parts[0] if len(parts) >= 1 else "unknown")
                tgt_lang = target_language or (parts[1] if len(parts) >= 2 else "unknown")
            else:
                src_lang = source_language or "unknown"
                tgt_lang = target_language or "unknown"

            response_data = {
                "text": translated_text,
                "model": model,
                "source_language": src_lang,
                "target_language": tgt_lang,
            }

            import json

            return Response(
                json.dumps(response_data),
                media_type="application/json",
            )

    raise HTTPException(status_code=500, detail=f"Unsupported model type for {model}")
