from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import huggingface_hub

    from speaches.executors.shared.executor import Executor

from fastapi import HTTPException
from huggingface_hub.utils._cache_manager import _scan_cached_repo

from speaches.hf_utils import (
    MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE,
    get_model_card_data_from_cached_repo_info,
    get_model_repo_path,
)


def get_model_card_data_or_raise(model_id: str) -> tuple[huggingface_hub.ModelCardData, list[str] | None]:
    model_repo_path = get_model_repo_path(model_id)
    if model_repo_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is not installed locally. You can download the model using `POST /v1/models`",
        )
    cached_repo_info = _scan_cached_repo(model_repo_path)
    result = get_model_card_data_from_cached_repo_info(cached_repo_info)
    if result is None:
        raise HTTPException(
            status_code=500,
            detail=MODEL_CARD_DOESNT_EXISTS_ERROR_MESSAGE.format(model_id=model_id),
        )
    return result


def find_executor_for_model_or_raise(
    model_id: str,
    model_card_data: huggingface_hub.ModelCardData,
    executors: Iterable[Executor],
    model_tags: list[str] | None = None,
) -> Executor:
    for executor in executors:
        if executor.can_handle_model(model_id, model_card_data, model_tags):
            return executor
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' is not supported. If you think this is a mistake, please open an issue.",
    )
