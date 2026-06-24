"""Analyst agent: turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a critical analyst. From the research notes, extract the key claims, compare "
    "differing viewpoints, and explicitly flag weak or unsupported evidence. Preserve the "
    "[n] citations from the notes. Be concise and structured."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``."""

        with trace_span("analyst"):
            try:
                user_prompt = (
                    f"Research question: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes or '(none)'}\n\n"
                    "Produce: (1) key claims with citations, (2) agreements/contradictions, "
                    "(3) gaps or weak evidence to caveat."
                )
                response = self.llm.complete(_SYSTEM, user_prompt, temperature=0.1)
                state.analysis_notes = response.content
                record_result(state, AgentName.ANALYST, response.content, response)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.exception("Analyst failed")
                state.errors.append(f"analyst: {exc}")
                state.analysis_notes = (
                    state.analysis_notes or "(analysis unavailable due to an error)"
                )
        return state
