"""Agent behavior tests using deterministic mock clients (no API key required)."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_researcher_first() -> None:
    state = SupervisorAgent().run(_state())
    assert state.next_route == ROUTE_RESEARCHER
    assert state.iteration == 1


def test_supervisor_routes_writer_when_analysis_ready() -> None:
    state = _state()
    state.research_notes = "notes [1]"
    state.analysis_notes = "analysis [1]"
    assert SupervisorAgent(use_critic=False).run(state).next_route == ROUTE_WRITER


def test_supervisor_done_when_complete() -> None:
    state = _state()
    state.research_notes = "notes [1]"
    state.analysis_notes = "analysis [1]"
    state.final_answer = "answer [1]"
    assert SupervisorAgent(use_critic=False).run(state).next_route == ROUTE_DONE


def test_supervisor_stops_at_max_iterations() -> None:
    state = _state()
    state.iteration = 6
    assert SupervisorAgent(max_iterations=6).run(state).next_route == ROUTE_DONE


def test_researcher_populates_sources_and_notes() -> None:
    state = ResearcherAgent(MockLLMClient(), MockSearchClient()).run(_state())
    assert state.sources, "expected sources from search"
    assert state.research_notes
    assert state.agent_results


def test_analyst_then_writer_populate_outputs() -> None:
    state = _state()
    state.research_notes = "Some research notes [1]"
    state = AnalystAgent(MockLLMClient()).run(state)
    assert state.analysis_notes
    state = WriterAgent(MockLLMClient()).run(state)
    assert state.final_answer
