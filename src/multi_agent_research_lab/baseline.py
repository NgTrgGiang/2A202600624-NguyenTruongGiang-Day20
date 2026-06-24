"""Single-agent baseline.

One agent does everything in a single LLM call (search -> answer). This is the control we
benchmark the multi-agent workflow against: simpler and cheaper, but with less separation
of concerns and weaker grounding/guardrails.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import record_result
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client
from multi_agent_research_lab.services.search_client import SearchClient, get_search_client

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a single research assistant doing the whole job alone: read the sources, reason "
    "about them, and write the final answer for the given audience in one pass. Cite claims "
    "with [n] and end with a short Sources list."
)


def run_baseline(
    query: str,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    search: SearchClient | None = None,
) -> ResearchState:
    """Run the single-agent baseline and return the populated state."""

    settings = settings or get_settings()
    llm = llm or get_llm_client(settings)
    search = search or get_search_client(settings)

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    with trace_span("baseline", {"query": query}):
        try:
            sources = search.search(query, request.max_sources)
            state.sources = sources
            sources_block = "\n".join(
                f"[{i}] {s.title} ({s.url or 'no-url'})\n    {s.snippet}"
                for i, s in enumerate(sources, start=1)
            )
            user_prompt = (
                f"Question: {query}\nAudience: {request.audience}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Write the complete answer now."
            )
            response = llm.complete(_SYSTEM, user_prompt, temperature=0.3)
            state.final_answer = response.content
            state.record_route(AgentName.WRITER.value)
            record_result(state, AgentName.WRITER, response.content, response, {"mode": "baseline"})
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.exception("Baseline failed")
            state.errors.append(f"baseline: {exc}")
            state.final_answer = (
                state.final_answer or "(baseline answer unavailable due to an error)"
            )
    return state
