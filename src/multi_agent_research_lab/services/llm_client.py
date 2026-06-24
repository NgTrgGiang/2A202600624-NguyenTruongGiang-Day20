"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
This keeps retry, timeout, and token/cost accounting in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Approximate USD price per 1M tokens (input, output). Fallback to (0, 0) for unknown models.
_MODEL_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float:
    """Rough cost estimate so the benchmark has a comparable number across runs."""

    in_price, out_price = _MODEL_PRICES_PER_M.get(model, (0.0, 0.0))
    cost = (input_tokens or 0) / 1_000_000 * in_price
    cost += (output_tokens or 0) / 1_000_000 * out_price
    return round(cost, 6)


class LLMClient:
    """OpenAI-backed, provider-agnostic LLM client.

    Agents call :meth:`complete`; all retry/timeout/usage logic lives here so the
    agents stay focused on their role.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for LLMClient. "
                "Use get_llm_client() for a mock fallback."
            )
        # Import lazily so the package still imports without the optional [llm] extra installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a model completion with token usage and an estimated cost."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
        )
        def _call() -> object:
            return self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.settings.timeout_seconds,
            )

        response = _call()
        choice = response.choices[0]  # type: ignore[attr-defined]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
        )


class MockLLMClient(LLMClient):
    """Deterministic, offline LLM used for tests and key-less smoke runs.

    It never calls the network. Output is a short, prompt-derived string plus a ``[1]``
    citation marker so downstream citation-coverage logic has something to find.
    """

    def __init__(self, settings: Settings | None = None) -> None:  # noqa: D401 - keep signature
        # Intentionally bypass the parent __init__ (no API key needed).
        self.settings = settings or get_settings()
        self.model = "mock"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        excerpt = " ".join(user_prompt.split())[:240]
        content = f"[mock-llm] {excerpt} [1]"
        return LLMResponse(content=content, input_tokens=120, output_tokens=60, cost_usd=0.0001)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return a real OpenAI client when a key is present, otherwise a deterministic mock."""

    settings = settings or get_settings()
    if settings.openai_api_key:
        return LLMClient(settings)
    logger.warning("OPENAI_API_KEY not set - falling back to MockLLMClient (no real model calls).")
    return MockLLMClient(settings)
