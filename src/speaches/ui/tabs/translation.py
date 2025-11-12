import logging

import gradio as gr

from speaches.config import Config
from speaches.ui.utils import http_client_from_gradio_req, openai_client_from_gradio_req
from speaches.utils import APIProxyError, format_api_proxy_error

logger = logging.getLogger(__name__)

TRANSLATION_ENDPOINT = "/v1/translations"

DEFAULT_TEXT = "Hello, how are you?"


def create_translation_tab(config: Config) -> None:
    async def update_model_dropdown(request: gr.Request) -> gr.Dropdown:
        openai_client = openai_client_from_gradio_req(request, config)
        models = (await openai_client.models.list(extra_query={"task": "translation"})).data
        model_ids: list[str] = [model.id for model in models]
        return gr.Dropdown(choices=model_ids, label="Model")

    async def handle_translation(
        text: str,
        model: str,
        source_language: str | None,
        target_language: str | None,
        request: gr.Request,
    ) -> str:
        try:
            if not text:
                msg = "No text provided. Please enter text to translate."
                raise APIProxyError(msg, suggestions=["Please enter some text to translate."])

            http_client = http_client_from_gradio_req(request, config)
            data = {
                "text": text,
                "model": model,
            }
            if source_language:
                data["source_language"] = source_language
            if target_language:
                data["target_language"] = target_language

            response = await http_client.post(TRANSLATION_ENDPOINT, data=data)
            response.raise_for_status()
            result = response.json()
            return result["text"]
        except Exception as e:
            logger.exception("Translation error")
            if not isinstance(e, APIProxyError):
                e = APIProxyError(str(e))
            return format_api_proxy_error(e, context="handle_translation")

    with gr.Tab(label="Translation") as tab:
        text = gr.Textbox(label="Input Text", value=DEFAULT_TEXT, lines=3)
        model_dropdown = gr.Dropdown(choices=[], label="Model")
        with gr.Row():
            source_language = gr.Textbox(
                label="Source Language (Optional)",
                placeholder="e.g., en, zh, ja (leave empty for auto-detect)",
                value="",
            )
            target_language = gr.Textbox(
                label="Target Language (Optional)",
                placeholder="e.g., en, zh, ja (leave empty for model default)",
                value="",
            )
        button = gr.Button("Translate")
        output = gr.Textbox(label="Translation Result")

        button.click(
            handle_translation,
            [text, model_dropdown, source_language, target_language],
            output,
        )

        tab.select(update_model_dropdown, inputs=None, outputs=model_dropdown)
