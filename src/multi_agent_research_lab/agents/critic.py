"""Critic agent (bonus): a lightweight fact-check / grounding review pass."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.text import citation_coverage

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a careful reviewer. Check whether the answer's claims are supported by the "
    "cited sources, flag any unsupported or hallucinated statements, and note missing "
    "citations. Reply with a short bullet review; do not rewrite the answer."
)

# Below this share of sources cited, we raise a risk flag for the benchmark to pick up.
_MIN_COVERAGE = 0.34


class CriticAgent(BaseAgent):
    """Optional fact-checking and grounding-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        """Validate the final answer and append findings + a coverage metric."""

        with trace_span("critic"):
            coverage = citation_coverage(state.final_answer, len(state.sources))
            try:
                user_prompt = (
                    f"Question: {state.request.query}\n\n"
                    f"Answer to review:\n{state.final_answer or '(none)'}\n\n"
                    f"Citation coverage (cited/available sources): {coverage:.2f}\n"
                    "Review the grounding and list concrete risks."
                )
                response = self.llm.complete(_SYSTEM, user_prompt, temperature=0.0)
                record_result(
                    state,
                    AgentName.CRITIC,
                    response.content,
                    response,
                    {"citation_coverage": coverage},
                )
                if coverage < _MIN_COVERAGE:
                    state.errors.append(f"critic: low citation coverage ({coverage:.2f})")
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.exception("Critic failed")
                state.errors.append(f"critic: {exc}")
        return state
