"""Poe chat wrapper."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


async def _stream_request(
    messages: list[BaseMessage], chat_poe: ChatPoe
) -> AsyncIterator[str]:
    from fastapi_poe.client import stream_request
    from fastapi_poe.types import ProtocolMessage, QueryRequest

    def _message_to_protocol_message(message: BaseMessage) -> ProtocolMessage:
        if isinstance(message, ChatMessage):
            if message.role == "Human":
                role = "user"
            elif message.role == "AI":
                role = "bot"
            elif message.role == "System":
                role = "system"
            else:
                raise ValueError(f"Unhandled message role: {message.role}")
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "bot"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unhandled message type: {type(message)}")

        return ProtocolMessage(content=message.content, role=role)

    query_request = QueryRequest(
        query=[_message_to_protocol_message(msg) for msg in messages],
        type="query",
        api_key=chat_poe.api_key,
        conversation_id=chat_poe.conversation_id,
        message_id=chat_poe.message_id,
        user_id=chat_poe.user_id,
        version=chat_poe.version,
    )

    async for bot_message in stream_request(
        query_request, chat_poe.bot_name, chat_poe.api_key
    ):
        yield bot_message.text


class ChatPoe(BaseChatModel):
    """
    Wrapper around the Poe API.

    This is meant to be used in the context of a Poe API bot to call on other Poe bots.

    Example:
        .. code-block:: python
            from fastapi_poe import PoeBot
            from fastapi_poe.types import QueryRequest
            from langchain.chat_models import ChatPoe

            class LangChainPoeBot(PoeBot):
                async def get_response(self, query: QueryRequest):
                    chat_poe = ChatPoe(
                        bot_name="Assistant",
                        api_key=query.api_key,
                        user_id=query.user_id,
                        conversation_id=query.conversation_id,
                        message_id=query.message_id,
                        version=query.version,
                    )
    """

    bot_name: str
    api_key: str
    user_id: str
    conversation_id: str
    message_id: str
    version: str

    def _create_chat_result(self, response: str) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response), text=response)
            ]
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = ""
        stream = _stream_request(messages, self)
        while True:
            try:
                text: str = asyncio.run(stream.__anext__())  # type: ignore[arg-type]
                response += text
                if run_manager:
                    run_manager.on_llm_new_token(text)
            except StopAsyncIteration:
                break
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = ""
        async for text in _stream_request(messages, self):
            response += text
            if run_manager:
                await run_manager.on_llm_new_token(text)
        return self._create_chat_result(response)

    def _llm_type(self) -> str:
        return "chat-poe"
