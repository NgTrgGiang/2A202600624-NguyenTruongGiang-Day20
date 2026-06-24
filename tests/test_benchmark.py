"""Benchmark metric tests with mock clients."""

from multi_agent_research_lab.baseline import run_baseline
from multi_agent_research_lab.evaluation.benchmark import run_benchmark, total_cost
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def test_run_benchmark_measures_latency_and_cost() -> None:
    state, metrics = run_benchmark(
        "baseline",
        "Explain multi-agent systems",
        lambda q: run_baseline(q, llm=MockLLMClient(), search=MockSearchClient()),
    )
    assert metrics.latency_seconds >= 0
    assert metrics.estimated_cost_usd is not None
    assert metrics.estimated_cost_usd > 0
    assert total_cost(state) == metrics.estimated_cost_usd
    assert "route=" in metrics.notes
