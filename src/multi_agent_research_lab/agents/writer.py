"""Writer agent: produces the final answer from research and analysis notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a clear technical writer. Using the research and analysis notes, write a "
    "well-structured answer for the given audience. Keep the [n] citations and add a short "
    "'Sources' list mapping each [n] to its title/URL. Do not introduce uncited claims."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``."""

        with trace_span("writer", {"audience": state.request.audience}):
            try:
                sources_list = "\n".join(
                    f"[{i}] {s.title} ({s.url or 'no-url'})"
                    for i, s in enumerate(state.sources, start=1)
                )
                user_prompt = (
                    f"Question: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Research notes:\n{state.research_notes or '(none)'}\n\n"
                    f"Analysis notes:\n{state.analysis_notes or '(none)'}\n\n"
                    f"Available sources:\n{sources_list or '(none)'}\n\n"
                    "Write the final answer now."
                )
                response = self.llm.complete(_SYSTEM, user_prompt, temperature=0.4)
                state.final_answer = response.content
                record_result(state, AgentName.WRITER, response.content, response)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.exception("Writer failed")
                state.errors.append(f"writer: {exc}")
                state.final_answer = (
                    state.final_answer or "(final answer unavailable due to an error)"
                )
        return state
