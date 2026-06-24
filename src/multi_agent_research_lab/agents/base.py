"""Base agent contract and shared helpers.

Concrete agents implement :meth:`run`, reading and updating the shared
:class:`ResearchState`. :func:`record_result` keeps the bookkeeping
(``agent_results`` + trace event + token/cost metadata) consistent across agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""


def record_result(
    state: ResearchState,
    agent: AgentName,
    content: str,
    response: LLMResponse | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> None:
    """Append an ``AgentResult`` and a matching trace event in one place."""

    metadata: dict[str, object] = dict(extra_metadata or {})
    if response is not None:
        metadata.update(
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        )
    state.agent_results.append(AgentResult(agent=agent, content=content, metadata=metadata))
    state.add_trace_event(
        agent.value,
        {"content_preview": content[:200], **{k: v for k, v in metadata.items() if k != "content"}},
    )
