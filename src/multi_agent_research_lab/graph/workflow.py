"""LangGraph workflow wiring supervisor + worker agents.

Orchestration lives here; agent internals stay in ``agents/``. Each node runs one agent and
returns the changed state fields (overwriting channels), which keeps us independent of
LangGraph reducer/version specifics. The supervisor sets ``next_route`` and a conditional
edge maps it to the next node or ``END``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

# Fields a node may change; we overwrite these channels on every node return.
_MUTABLE_FIELDS = (
    "iteration",
    "route_history",
    "next_route",
    "sources",
    "research_notes",
    "analysis_notes",
    "final_answer",
    "agent_results",
    "trace",
    "errors",
)


def _node(agent: BaseAgent) -> Callable[[ResearchState], dict[str, Any]]:
    """Wrap an agent so LangGraph receives only the changed fields."""

    def run_node(state: ResearchState) -> dict[str, Any]:
        new_state = agent.run(state)
        return {field: getattr(new_state, field) for field in _MUTABLE_FIELDS}

    return run_node


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        llm: LLMClient,
        search: SearchClient,
        settings: Settings | None = None,
        use_critic: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.use_critic = use_critic
        self.supervisor = SupervisorAgent(
            max_iterations=self.settings.max_iterations, use_critic=use_critic
        )
        self.researcher = ResearcherAgent(llm, search)
        self.analyst = AnalystAgent(llm)
        self.writer = WriterAgent(llm)
        self.critic = CriticAgent(llm)

    def build(self) -> Any:
        """Create and compile the LangGraph graph."""

        from langgraph.graph import END, StateGraph

        graph: Any = StateGraph(ResearchState)
        graph.add_node("supervisor", _node(self.supervisor))
        graph.add_node("researcher", _node(self.researcher))
        graph.add_node("analyst", _node(self.analyst))
        graph.add_node("writer", _node(self.writer))
        graph.add_node("critic", _node(self.critic))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.next_route or ROUTE_DONE,
            {
                ROUTE_RESEARCHER: "researcher",
                ROUTE_ANALYST: "analyst",
                ROUTE_WRITER: "writer",
                ROUTE_CRITIC: "critic",
                ROUTE_DONE: END,
            },
        )
        for worker in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(worker, "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final ``ResearchState``."""

        compiled = self.build()
        recursion_limit = self.settings.max_iterations * 3 + 10
        try:
            result = compiled.invoke(state, config={"recursion_limit": recursion_limit})
        except Exception as exc:  # noqa: BLE001 - surface as a recoverable workflow error
            logger.exception("Workflow execution failed")
            state.errors.append(f"workflow: {exc}")
            if state.final_answer is None:
                state.final_answer = "(workflow did not complete; see errors)"
            return state

        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
