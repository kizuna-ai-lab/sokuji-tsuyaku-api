from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING

import openai
from openai.types.beta.realtime.error_event import Error

from speaches.realtime.chat_utils import items_to_chat_messages
from speaches.realtime.event_router import EventRouter
from speaches.realtime.session_event_router import unsupported_field_error, update_dict
from speaches.realtime.utils import generate_response_id, task_done_callback
from speaches.types.realtime import (
    ConversationItemContentAudio,
    ConversationItemContentText,
    ConversationItemMessage,
    ErrorEvent,
    RealtimeResponse,
    Response,
    # TODO: RealtimeResponseStatus,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ServerConversationItem,
    create_server_error,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from httpx import AsyncClient
    from openai.resources.audio import AsyncSpeech

    from speaches.realtime.context import SessionContext
    from speaches.realtime.conversation_event_router import Conversation
    from speaches.realtime.pubsub import EventPubSub

logger = logging.getLogger(__name__)

event_router = EventRouter()

# TODO: start using this error
conversation_already_has_active_response_error = Error(
    type="invalid_request_error",
    message="Conversation already has an active response",
)


class ResponseHandler:
    def __init__(
        self,
        *,
        translation_client: AsyncClient,
        speech_client: AsyncSpeech,
        translation_model: str,
        configuration: Response,
        conversation: Conversation,
        pubsub: EventPubSub,
    ) -> None:
        self.id = generate_response_id()
        self.translation_client = translation_client
        self.speech_client = speech_client
        self.translation_model = translation_model
        self.configuration = configuration
        self.conversation = conversation
        self.pubsub = pubsub
        self.response = RealtimeResponse(
            id=self.id,
            status="incomplete",
            output=[],
            modalities=configuration.modalities,
        )
        self.task: asyncio.Task[None] | None = None

    @contextmanager
    def add_output_item[T: ServerConversationItem](self, item: T) -> Generator[T, None, None]:
        self.response.output.append(item)
        self.pubsub.publish_nowait(ResponseOutputItemAddedEvent(response_id=self.id, item=item))
        yield item
        assert item.status == "incomplete", item
        item.status = "completed"  # TODO: do not update the status if the response was cancelled
        self.pubsub.publish_nowait(ResponseOutputItemDoneEvent(response_id=self.id, item=item))
        self.response.status = "completed"  # TODO: set to "cancelled" if the response was cancelled. Additionally, populate the `respones.status_details`
        self.pubsub.publish_nowait(ResponseDoneEvent(response=self.response))

    @contextmanager
    def add_item_content[T: ConversationItemContentText | ConversationItemContentAudio](
        self, item: ConversationItemMessage, content: T
    ) -> Generator[T, None, None]:
        item.content.append(content)
        self.pubsub.publish_nowait(
            ResponseContentPartAddedEvent(response_id=self.id, item_id=item.id, part=content.to_part())
        )
        yield content
        self.pubsub.publish_nowait(
            ResponseContentPartDoneEvent(response_id=self.id, item_id=item.id, part=content.to_part())
        )

    async def conversation_item_message_text_handler(self, translated_text: str) -> None:
        with self.add_output_item(ConversationItemMessage(role="assistant", status="incomplete", content=[])) as item:
            self.conversation.create_item(item)

            with self.add_item_content(item, ConversationItemContentText(text="")) as content:
                content.text = translated_text
                self.pubsub.publish_nowait(
                    ResponseTextDeltaEvent(item_id=item.id, response_id=self.id, delta=translated_text)
                )
                self.pubsub.publish_nowait(
                    ResponseTextDoneEvent(item_id=item.id, response_id=self.id, text=content.text)
                )

    async def conversation_item_message_audio_handler(self, translated_text: str) -> None:
        with self.add_output_item(ConversationItemMessage(role="assistant", status="incomplete", content=[])) as item:
            self.conversation.create_item(item)

            with self.add_item_content(item, ConversationItemContentAudio(audio="", transcript="")) as content:
                import base64

                response = await self.speech_client.create(
                    model=self.configuration.speech_model,
                    voice=self.configuration.voice,
                    input=translated_text,
                    response_format="pcm",
                )
                audio_bytes = await response.aread()
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                content.transcript = translated_text
                self.pubsub.publish_nowait(
                    ResponseAudioTranscriptDeltaEvent(item_id=item.id, response_id=self.id, delta=translated_text)
                )
                self.pubsub.publish_nowait(
                    ResponseAudioDeltaEvent(item_id=item.id, response_id=self.id, delta=audio_base64)
                )

                self.pubsub.publish_nowait(ResponseAudioDoneEvent(item_id=item.id, response_id=self.id))
                self.pubsub.publish_nowait(
                    ResponseAudioTranscriptDoneEvent(
                        item_id=item.id, response_id=self.id, transcript=content.transcript
                    )
                )

    async def generate_response(self) -> None:
        try:
            messages = items_to_chat_messages(self.configuration.input)
            if not messages:
                logger.warning("No messages to translate")
                return

            last_message = messages[-1]
            source_text = ""
            if "content" in last_message:
                source_text = last_message["content"]

            if not source_text:
                logger.warning("No source text found in last message")
                return

            response = await self.translation_client.post(
                "/v1/translations",
                data={
                    "text": source_text,
                    "model": self.translation_model,
                },
            )
            response.raise_for_status()
            translated_text = response.json()["text"]

            if not translated_text:
                logger.warning(
                    f"Translation returned empty text for input: {source_text[:50]}... "
                    f"Model {self.translation_model} may not support this language direction"
                )
                error_message = (
                    f"Translation model {self.translation_model} does not support this language. "
                    "Please check your input language or configure a different translation model."
                )
                await self.conversation_item_message_text_handler(error_message)
                return

            if self.configuration.modalities == ["text"]:
                handler = self.conversation_item_message_text_handler
            else:
                handler = self.conversation_item_message_audio_handler

            await handler(translated_text)
        except openai.APIError as e:
            logger.exception("Error while generating response")
            self.pubsub.publish_nowait(
                ErrorEvent(error=Error(type="server_error", message=f"{type(e).__name__}: {e.message}"))
            )
            raise

    def start(self) -> None:
        assert self.task is None
        self.task = asyncio.create_task(self.generate_response())
        self.task.add_done_callback(task_done_callback)

    def stop(self) -> None:
        assert self.task is not None
        self.task.cancel()


@event_router.register("response.create")
async def handle_response_create_event(ctx: SessionContext, event: ResponseCreateEvent) -> None:
    if ctx.response is not None:
        ctx.response.stop()

    configuration = Response(
        conversation="auto", input=list(ctx.conversation.items.values()), **ctx.session.model_dump()
    )
    if event.response is not None:
        if event.response.conversation is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.conversation"))
        if event.response.input is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.input"))
        if event.response.output_audio_format is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.output_audio_format"))
        if event.response.metadata is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.metadata"))

        configuration_dict = configuration.model_dump()
        configuration_update_dict = event.response.model_dump(
            exclude_none=True, exclude={"conversation", "input", "output_audio_format", "metadata"}
        )
        logger.debug(f"Applying response configuration update: {configuration_update_dict}")
        logger.debug(f"Response configuration before update: {configuration_dict}")
        updated_configuration = update_dict(configuration_dict, configuration_update_dict)
        logger.debug(f"Response configuration after update: {updated_configuration}")
        configuration = Response(**updated_configuration)

    ctx.response = ResponseHandler(
        translation_client=ctx.translation_client,
        speech_client=ctx.speech_client,
        translation_model=ctx.session.translation_model,
        configuration=configuration,
        conversation=ctx.conversation,
        pubsub=ctx.pubsub,
    )
    ctx.pubsub.publish_nowait(ResponseCreatedEvent(response=ctx.response.response))
    ctx.response.start()
    assert ctx.response.task is not None
    await ctx.response.task
    ctx.response = None


@event_router.register("response.cancel")
def handle_response_cancel_event(ctx: SessionContext, event: ResponseCancelEvent) -> None:
    ctx.pubsub.publish_nowait(
        create_server_error(f"Handling of the '{event.type}' event is not implemented.", event_id=event.event_id)
    )
