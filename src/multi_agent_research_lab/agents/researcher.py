"""Researcher agent: gathers sources and writes concise, cited research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a meticulous research assistant. Read the provided sources and write concise "
    "research notes. Cite every factual claim with a bracketed source index like [1]. "
    "Do not invent sources or facts beyond what the snippets support."
)


def _format_sources(sources: list[SourceDocument]) -> str:
    return "\n".join(
        f"[{i}] {s.title} ({s.url or 'no-url'})\n    {s.snippet}"
        for i, s in enumerate(sources, start=1)
    )


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient, search: SearchClient) -> None:
        self.llm = llm
        self.search = search

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""

        with trace_span("researcher", {"query": state.request.query}):
            try:
                sources = self.search.search(state.request.query, state.request.max_sources)
                state.sources = sources
                user_prompt = (
                    f"Research question: {state.request.query}\n\n"
                    f"Sources:\n{_format_sources(sources)}\n\n"
                    "Write 4-8 bullet research notes, each ending with its [n] citation."
                )
                response = self.llm.complete(_SYSTEM, user_prompt, temperature=0.2)
                state.research_notes = response.content
                record_result(
                    state,
                    AgentName.RESEARCHER,
                    response.content,
                    response,
                    {"num_sources": len(sources)},
                )
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the workflow
                logger.exception("Researcher failed")
                state.errors.append(f"researcher: {exc}")
                # Set a minimal note so the supervisor advances instead of looping.
                state.research_notes = (
                    state.research_notes or "(research unavailable due to an error)"
                )
        return state
