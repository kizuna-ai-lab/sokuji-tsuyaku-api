from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import ctranslate2
import huggingface_hub
from pydantic import BaseModel
from transformers import AutoTokenizer

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import TextTranslationRequest, TextTranslationResponse
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
)
from speaches.model_registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Generator

    from speaches.config import MarianConfig

LIBRARY_NAME = "ctranslate2"
TASK_NAME_TAG = "translation"
TAGS = {"marian"}

logger = logging.getLogger(__name__)


class MarianModelFiles(BaseModel):
    model_dir: Path
    tokenizer_dir: Path


class MarianModel(Model):
    source_language: str
    target_language: str


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags=TAGS,
)


class MarianModelRegistry(ModelRegistry[Model, MarianModelFiles]):
    def list_remote_models(self) -> Generator[Model, None, None]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            try:
                if model.created_at is None or getattr(model, "card_data", None) is None:
                    logger.debug(f"Skipping (missing created_at/card_data): {model}")
                    continue
                assert model.card_data is not None

                repo_name = model.id.split("/")[-1]

                # Only support pre-converted CTranslate2 models (ct2fast-opus-mt-*)
                if repo_name.startswith("ct2fast-opus-mt-"):
                    parts = repo_name.replace("ct2fast-opus-mt-", "").split("-")
                elif repo_name.startswith("opus-mt-"):
                    # Also support direct opus-mt-* naming for ctranslate2 models
                    parts = repo_name.replace("opus-mt-", "").split("-")
                else:
                    logger.debug(f"Skipping (not opus-mt model): {model.id}")
                    continue

                if len(parts) < 2:
                    logger.debug(f"Skipping (cannot parse language pair): {model.id}")
                    continue

                source_lang = parts[0]
                target_lang = parts[1]

                languages = extract_language_list(model.card_data)

                yield MarianModel(
                    id=model.id,
                    created=int(model.created_at.timestamp()),
                    owned_by=model.id.split("/")[0],
                    language=languages if languages else [source_lang, target_lang],
                    task=TASK_NAME_TAG,
                    source_language=source_lang,
                    target_language=target_lang,
                )

            except Exception:
                logger.exception(f"Skipping (unexpected error): {model.id}")
                continue

    def list_local_models(self) -> Generator[Model, None, None]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            result = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if result is None:
                continue
            model_card_data, model_tags = result
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data, model_tags):
                repo_name = cached_repo_info.repo_id.split("/")[-1]
                if not repo_name.startswith("opus-mt-"):
                    continue

                parts = repo_name.replace("opus-mt-", "").split("-")
                if len(parts) < 2:
                    continue

                source_lang = parts[0]
                target_lang = parts[1]
                languages = extract_language_list(model_card_data)

                yield MarianModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=languages if languages else [source_lang, target_lang],
                    task=TASK_NAME_TAG,
                    source_language=source_lang,
                    target_language=target_lang,
                )

    def get_model_files(self, model_id: str) -> MarianModelFiles:
        cache_dir = Path(huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model"))

        return MarianModelFiles(
            model_dir=cache_dir,
            tokenizer_dir=cache_dir,
        )

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=["*.json", "*.model", "*.txt", "README.md"]
        )


marian_model_registry = MarianModelRegistry(hf_model_filter=hf_model_filter)


class MarianTranslationModel:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto", compute_type: str = "int8") -> None:
        self.translator = ctranslate2.Translator(model_path, device=device, compute_type=compute_type)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded translation model from {model_path}")

    def translate(self, text: str) -> str:
        source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self.translator.translate_batch([source_tokens])
        target_tokens = results[0].hypotheses[0]
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        translated_text = self.tokenizer.decode(target_ids, skip_special_tokens=True)
        return translated_text


class MarianModelManager(BaseModelManager[MarianTranslationModel]):
    def __init__(self, ttl: int, marian_config: MarianConfig) -> None:
        super().__init__(ttl)
        self.marian_config = marian_config

    def _load_fn(self, model_id: str) -> MarianTranslationModel:
        model_files = marian_model_registry.get_model_files(model_id)
        return MarianTranslationModel(
            model_path=str(model_files.model_dir),
            tokenizer_path=str(model_files.tokenizer_dir),
            device=self.marian_config.inference_device,
            compute_type=self.marian_config.compute_type,
        )

    def handle_text_translation_request(
        self, request: TextTranslationRequest, **kwargs  # noqa: ARG002
    ) -> TextTranslationResponse:
        """Handle text translation request following the TextTranslationHandler protocol."""
        with self.load_model(request.model) as marian:
            translated_text = marian.translate(request.text)

            # Extract language codes from model ID
            repo_name = request.model.split("/")[-1]
            if "opus-mt-" in repo_name:
                lang_part = repo_name.split("opus-mt-", 1)[1]
                parts = lang_part.split("-")
                src_lang = request.source_language or (parts[0] if len(parts) >= 1 else "unknown")
                tgt_lang = request.target_language or (parts[1] if len(parts) >= 2 else "unknown")
            else:
                src_lang = request.source_language or "unknown"
                tgt_lang = request.target_language or "unknown"

            return TextTranslationResponse(
                text=translated_text,
                model=request.model,
                source_language=src_lang,
                target_language=tgt_lang,
            )
