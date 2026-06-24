"""Benchmark: single-agent vs multi-agent.

Measures latency and derives comparable signals from the final state: estimated token cost,
citation coverage (grounding proxy), error count, and number of agent steps. Quality is left
to peer review (``quality_score=None``); plug an LLM-judge into :func:`score_quality` if wanted.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.utils.text import citation_coverage

Runner = Callable[[str], ResearchState]


def total_cost(state: ResearchState) -> float:
    """Sum per-call cost recorded in each agent result's metadata."""

    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
    return round(total, 6)


def summarize_state(state: ResearchState) -> dict[str, float | int]:
    """Derive comparable run signals from the final state."""

    return {
        "agent_steps": len(state.agent_results),
        "errors": len(state.errors),
        "citation_coverage": citation_coverage(state.final_answer, len(state.sources)),
        "cost_usd": total_cost(state),
    }


def score_quality(state: ResearchState) -> float | None:
    """Placeholder for an automated quality score.

    TODO(optional): call an LLM-judge here to rate the final answer 0-10. Left as ``None`` so
    quality comes from the peer-review rubric in ``docs/peer_review_rubric.md``.
    """

    return None


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run ``runner`` on ``query`` and return the state plus comparable metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    summary = summarize_state(state)
    notes = (
        f"steps={summary['agent_steps']}, errors={summary['errors']}, "
        f"citation_coverage={summary['citation_coverage']}, "
        f"route={'>'.join(state.route_history) or 'n/a'}"
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=summary["cost_usd"],
        quality_score=score_quality(state),
        notes=notes,
    )
    return state, metrics
