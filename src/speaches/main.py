from __future__ import annotations

import logging
import os
import uuid

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.exception_handlers import (
    http_exception_handler,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse

from speaches.dependencies import ApiKeyDependency, get_config
from speaches.logger import setup_logger
from speaches.routers.misc import (
    router as misc_router,
)
from speaches.routers.models import (
    router as models_router,
)
from speaches.routers.realtime_rtc import (
    router as realtime_rtc_router,
)
from speaches.routers.realtime_ws import (
    router as realtime_ws_router,
)
from speaches.routers.speech import (
    router as speech_router,
)
from speaches.routers.speech_embedding import (
    router as speech_embedding_router,
)
from speaches.routers.stt import (
    router as stt_router,
)
from speaches.routers.translations import (
    router as translations_router,
)
from speaches.routers.vad import (
    router as vad_router,
)
from speaches.utils import APIProxyError

# https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
# https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags
TAGS_METADATA = [
    {"name": "automatic-speech-recognition"},
    {"name": "speech-to-text"},
    {"name": "speaker-embedding"},
    {"name": "realtime"},
    {"name": "models"},
    {"name": "diagnostic"},
    {
        "name": "experimental",
        "description": "Not meant for public use yet. May change or be removed at any time.",
    },
]


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    # Reduce noise from aiortc during normal disconnect operations
    # MediaStreamError is expected when clients disconnect
    logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.ERROR)

    # WORKAROUND: CTranslate2 cuDNN library loading issue
    #
    # Problem: CTranslate2 wheel package is incomplete - it bundles cuDNN 9.1.0 main library
    # but is missing libcudnn_cnn.so.9.1.0. When running under Uvicorn, the library loading
    # fails with "Unable to load libcudnn_cnn.so.9.1.0" even though:
    # 1. We copied the file from system cuDNN 9.15.0 to ctranslate2.libs/
    # 2. We set LD_LIBRARY_PATH to point to ctranslate2.libs/
    #
    # Root cause: CTranslate2's binary has hardcoded RPATH=$ORIGIN/../ctranslate2.libs which
    # takes priority over LD_LIBRARY_PATH. In Uvicorn's complex async environment with multiple
    # event loops and the httpx.AsyncClient making internal ASGI calls, the library loading
    # mechanism fails to properly search the LD_LIBRARY_PATH fallback paths.
    #
    # Solution: Use ctypes.CDLL with RTLD_GLOBAL to preload libcudnn_cnn.so into the global
    # symbol table BEFORE any other library tries to load it. This ensures the correct version
    # is available to all subsequent library loads, regardless of RPATH or loading context.
    import ctypes
    import pathlib

    ctranslate2_libs = (
        pathlib.Path(__file__).parent.parent.parent / ".venv/lib/python3.12/site-packages/ctranslate2.libs"
    )
    cudnn_file = ctranslate2_libs / "libcudnn_cnn.so.9.1.0"

    if cudnn_file.exists():
        try:
            ctypes.CDLL(str(cudnn_file), mode=ctypes.RTLD_GLOBAL)
            logger.debug(f"Preloaded cuDNN CNN library from {cudnn_file}")
        except OSError as e:
            logger.warning(f"Failed to preload cuDNN library: {e}. CTranslate2 CUDA inference may fail.")

    logger.debug(f"Config: {config}")

    # Create main app WITHOUT global authentication
    app = FastAPI(
        title="Speaches",
        version="0.2.0",
        license_info={"name": "MIT License", "identifier": "MIT"},
        openapi_tags=TAGS_METADATA,
    )

    # Register global exception handler for APIProxyError
    @app.exception_handler(APIProxyError)
    async def _api_proxy_error_handler(_request: Request, exc: APIProxyError) -> JSONResponse:
        error_id = str(uuid.uuid4())
        logger.exception(f"[{{error_id}}] {exc.message}")
        content = {
            "detail": exc.message,
            "hint": exc.hint,
            "suggested_fixes": exc.suggestions,
            "error_id": error_id,
        }

        # HACK: replace with something else
        log_level = os.getenv("SPEACHES_LOG_LEVEL", "INFO").upper()
        if log_level == "DEBUG" and exc.debug:
            content["debug"] = exc.debug
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(StarletteHTTPException)
    async def _custom_http_exception_handler(request: Request, exc: HTTPException) -> Response:
        logger.error(f"HTTP error: {exc}")
        return await http_exception_handler(request, exc)

    # HTTP routers WITH authentication (if API key is configured)
    http_dependencies = []
    if config.api_key is not None:
        http_dependencies.append(ApiKeyDependency)

    app.include_router(stt_router, dependencies=http_dependencies)
    app.include_router(translations_router, dependencies=http_dependencies)
    app.include_router(models_router, dependencies=http_dependencies)
    app.include_router(misc_router, dependencies=http_dependencies)
    app.include_router(realtime_rtc_router, dependencies=http_dependencies)
    app.include_router(speech_router, dependencies=http_dependencies)
    app.include_router(speech_embedding_router, dependencies=http_dependencies)
    app.include_router(vad_router, dependencies=http_dependencies)

    # WebSocket router WITHOUT authentication (handles its own)
    app.include_router(realtime_ws_router)

    # HACK: move this elsewhere
    app.get("/v1/realtime", include_in_schema=False)(lambda: RedirectResponse(url="/v1/realtime/"))
    app.mount("/v1/realtime", StaticFiles(directory="realtime-console/dist", html=True))

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if config.enable_ui:
        import gradio as gr

        from speaches.ui.app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="")

        logger = logging.getLogger("speaches.main")
        if config.host and config.port:
            display_host = "localhost" if config.host in ("0.0.0.0", "127.0.0.1") else config.host
            url = f"http://{display_host}:{config.port}/"
            logger.info(f"\n\nTo view the gradio web ui of speaches open your browser and visit:\n\n{url}\n\n")
        # If host or port is missing, do not print a possibly incorrect URL.

    return app
