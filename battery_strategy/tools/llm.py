from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
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


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None

    cleaned = base_url.strip().strip("\"'")
    if not cleaned:
        return None

    parsed = urlparse(cleaned)
    if parsed.scheme in {"http", "https"}:
        return cleaned

    if "://" in cleaned:
        raise LLMError(f"Invalid OPENAI_BASE_URL: {cleaned}")

    if cleaned.startswith(("localhost", "127.0.0.1")):
        return f"http://{cleaned}"
    return f"https://{cleaned}"


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
        resolved_base_url = normalize_openai_base_url(base_url or os.getenv("OPENAI_BASE_URL"))
        if resolved_base_url is None:
            resolved_base_url = "https://api.openai.com/v1"

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = resolved_base_url
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set.")

        from openai import OpenAI

        # Ignore malformed proxy-related env vars from the shell unless the app
        # explicitly passes a proxy-aware client.
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(trust_env=False, base_url=self.base_url),
        )

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
