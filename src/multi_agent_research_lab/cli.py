"""Command-line entrypoint for the lab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.baseline import run_baseline
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark, summarize_state
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import JsonTraceExporter, maybe_enable_langsmith
from multi_agent_research_lab.services.llm_client import get_llm_client
from multi_agent_research_lab.services.search_client import get_search_client
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level)
    maybe_enable_langsmith(settings)
    return settings


def _build_workflow(settings: Settings) -> MultiAgentWorkflow:
    llm = get_llm_client(settings)
    search = get_search_client(settings)
    return MultiAgentWorkflow(llm, search, settings)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    settings = _init()
    state = run_baseline(query, settings)
    JsonTraceExporter().export(state, "baseline")
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    settings = _init()
    workflow = _build_workflow(settings)
    state = workflow.run(ResearchState(request=ResearchQuery(query=query)))
    JsonTraceExporter().export(state, "multi_agent")
    summary = summarize_state(state)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(
        f"route: {' > '.join(state.route_history)}\n"
        f"steps={summary['agent_steps']} cost=${summary['cost_usd']} "
        f"citation_coverage={summary['citation_coverage']} errors={summary['errors']}"
    )


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option("--config", "-c", help="Lab config YAML")] = Path(
        "configs/lab_default.yaml"
    ),
) -> None:
    """Benchmark baseline vs multi-agent across the config's queries; write a markdown report."""

    settings = _init()
    queries: list[str] = yaml.safe_load(config.read_text(encoding="utf-8"))["benchmark"]["queries"]
    workflow = _build_workflow(settings)

    all_metrics = []
    base_latency = base_cost = multi_latency = multi_cost = 0.0
    for i, query in enumerate(queries, start=1):
        b_state, b_metrics = run_benchmark(
            f"baseline-q{i}", query, lambda q: run_baseline(q, settings)
        )
        m_state, m_metrics = run_benchmark(
            f"multi-agent-q{i}",
            query,
            lambda q: workflow.run(ResearchState(request=ResearchQuery(query=q))),
        )
        all_metrics += [b_metrics, m_metrics]
        base_latency += b_metrics.latency_seconds
        base_cost += b_metrics.estimated_cost_usd or 0.0
        multi_latency += m_metrics.latency_seconds
        multi_cost += m_metrics.estimated_cost_usd or 0.0
        JsonTraceExporter().export(b_state, f"baseline_q{i}")
        JsonTraceExporter().export(m_state, f"multi_agent_q{i}")

    n = max(len(queries), 1)
    analysis = (
        f"Across {len(queries)} queries (averages):\n\n"
        f"- Baseline: latency {base_latency / n:.2f}s, cost ${base_cost / n:.4f}\n"
        f"- Multi-agent: latency {multi_latency / n:.2f}s, cost ${multi_cost / n:.4f}\n\n"
        "Multi-agent typically trades higher latency/cost for better grounding and separation "
        "of concerns; the baseline is cheaper but does everything in one pass. Add peer-review "
        "quality scores to complete the comparison."
    )
    report = render_markdown_report(all_metrics, analysis=analysis)
    path = LocalArtifactStore().write_text("benchmark_report.md", report)
    console.print(Panel.fit(f"Report written to {path}", title="Benchmark"))
    console.print(report)


if __name__ == "__main__":
    app()
