from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

    from speaches.config import NllbConfig

LIBRARY_NAME = "transformers"
TASK_NAME_TAG = "translation"
TAGS = {"nllb"}

logger = logging.getLogger(__name__)

LANGUAGE_CODE_MAP = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "fa": "pes_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ur": "urd_Arab",
}


def normalize_language_code(code: str) -> str:
    code_lower = code.lower().strip()
    if code_lower in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[code_lower]
    if "_" in code:
        return code
    logger.warning(f"Unknown language code: {code}, using as-is")
    return code


class NllbModelFiles(BaseModel):
    model_dir: Path


class NllbModel(Model):
    source_languages: list[str] = Field(default_factory=list)
    target_languages: list[str] = Field(default_factory=list)


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags=TAGS,
)


class NllbModelRegistry(ModelRegistry[Model, NllbModelFiles]):
    def list_remote_models(self) -> Generator[Model, None, None]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            try:
                if model.created_at is None or getattr(model, "card_data", None) is None:
                    logger.debug(f"Skipping (missing created_at/card_data): {model}")
                    continue
                assert model.card_data is not None

                if "nllb" not in model.id.lower():
                    logger.debug(f"Skipping (not NLLB model): {model.id}")
                    continue

                languages = extract_language_list(model.card_data)

                yield NllbModel(
                    id=model.id,
                    created=int(model.created_at.timestamp()),
                    owned_by=model.id.split("/")[0],
                    language=languages if languages else [],
                    task=TASK_NAME_TAG,
                    source_languages=languages if languages else [],
                    target_languages=languages if languages else [],
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
                if "nllb" not in cached_repo_info.repo_id.lower():
                    continue

                languages = extract_language_list(model_card_data)

                yield NllbModel(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=languages if languages else [],
                    task=TASK_NAME_TAG,
                    source_languages=languages if languages else [],
                    target_languages=languages if languages else [],
                )

    def get_model_files(self, model_id: str) -> NllbModelFiles:
        model_path = huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")
        return NllbModelFiles(model_dir=Path(model_path))

    def download_model_files(self, model_id: str) -> None:
        huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")


nllb_model_registry = NllbModelRegistry(hf_model_filter=hf_model_filter)


class NllbTranslationModel:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        beam_size: int = 4,
        max_length: int = 200,
    ) -> None:
        self.device = device
        self.beam_size = beam_size
        self.max_length = max_length

        logger.info(f"Loading NLLB tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info(f"Loading NLLB model from {model_path} on device {device}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        if device.type == "cuda":
            try:
                self.model.half()
                logger.info("Enabled half precision (FP16) for GPU inference")
            except Exception:
                logger.exception("Failed to enable half precision, using FP32")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        text = text.strip()
        if not text:
            return ""

        src_lang_normalized = normalize_language_code(src_lang)
        tgt_lang_normalized = normalize_language_code(tgt_lang)

        self.tokenizer.src_lang = src_lang_normalized

        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if hasattr(self.tokenizer, "lang_code_to_id"):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_normalized]
        else:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_normalized)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=self.beam_size,
                no_repeat_ngram_size=2,
            )

        translated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translated_text


class NllbModelManager(BaseModelManager[NllbTranslationModel]):
    def __init__(self, ttl: int, nllb_config: NllbConfig) -> None:
        super().__init__(ttl)
        self.nllb_config = nllb_config

    def _load_fn(self, model_id: str) -> NllbTranslationModel:
        model_files = nllb_model_registry.get_model_files(model_id)

        if self.nllb_config.inference_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.nllb_config.inference_device)

        return NllbTranslationModel(
            model_path=str(model_files.model_dir),
            device=device,
            beam_size=self.nllb_config.beam_size,
            max_length=self.nllb_config.max_length,
        )

    def handle_text_translation_request(
        self,
        request: TextTranslationRequest,
        **kwargs,  # noqa: ARG002
    ) -> TextTranslationResponse:
        if not request.source_language or not request.target_language:
            msg = "Both source_language and target_language are required for NLLB translation"
            raise ValueError(msg)

        with self.load_model(request.model) as nllb:
            translated_text = nllb.translate(request.text, request.source_language, request.target_language)

            return TextTranslationResponse(
                text=translated_text,
                model=request.model,
                source_language=request.source_language,
                target_language=request.target_language,
            )
