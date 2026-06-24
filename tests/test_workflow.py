"""End-to-end workflow + baseline tests with mock clients (no API key required)."""

from multi_agent_research_lab.baseline import run_baseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def test_multi_agent_workflow_runs_end_to_end() -> None:
    settings = get_settings()
    workflow = MultiAgentWorkflow(MockLLMClient(), MockSearchClient(), settings)
    state = workflow.run(ResearchState(request=ResearchQuery(query="Explain GraphRAG tradeoffs")))

    assert state.final_answer
    assert state.research_notes and state.analysis_notes
    # Supervisor visits are bounded by the guardrail.
    assert state.iteration <= settings.max_iterations
    assert state.route_history[0] == "researcher"
    assert state.route_history[-1] == "done"


def test_baseline_runs_end_to_end() -> None:
    state = run_baseline(
        "Explain GraphRAG tradeoffs",
        llm=MockLLMClient(),
        search=MockSearchClient(),
    )
    assert state.final_answer
    assert state.sources
    assert len(state.agent_results) == 1  # single agent
