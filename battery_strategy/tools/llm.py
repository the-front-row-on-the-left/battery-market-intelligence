from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from battery_strategy.utils.common import safe_json_loads

load_dotenv()


class LLMError(RuntimeError):
    """Raised when an LLM call fails."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    raw: Any | None = None


class BaseLLM:
    def text(self, instructions: str, input_text: str) -> LLMResponse:
        raise NotImplementedError

    def json(self, instructions: str, input_text: str, *, fallback: Any) -> Any:
        response = self.text(instructions, input_text)
        return safe_json_loads(response.text, fallback=fallback)


class MockLLM(BaseLLM):
    def text(self, instructions: str, input_text: str) -> LLMResponse:
        return LLMResponse(text='{"summary": "mock output"}')


class OpenAIResponsesLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 4000,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set.")

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def text(self, instructions: str, input_text: str) -> LLMResponse:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_text,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(str(exc)) from exc

        text = getattr(response, "output_text", "") or ""
        if not text:
            raise LLMError("Empty response text from OpenAI Responses API.")
        return LLMResponse(text=text, raw=response)
