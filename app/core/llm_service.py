from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMService:
    """Wraps DeepSeek via LangChain's ChatOpenAI with a custom base URL."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self._model_name = model or settings.deepseek_model
        self._client = ChatOpenAI(
            api_key=api_key or settings.deepseek_api_key,
            base_url=base_url or settings.deepseek_base_url,
            model=self._model_name,
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = self._client.invoke(messages)
        usage = result.usage_metadata or {}

        return LLMResponse(
            content=result.content,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    async def agenerate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = await self._client.ainvoke(messages)
        usage = result.usage_metadata or {}

        return LLMResponse(
            content=result.content,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    async def astream(self, prompt: str, system_prompt: str | None = None):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        async for chunk in self._client.astream(messages):
            if chunk.content:
                yield chunk.content

    @property
    def model_name(self) -> str:
        return self._model_name
