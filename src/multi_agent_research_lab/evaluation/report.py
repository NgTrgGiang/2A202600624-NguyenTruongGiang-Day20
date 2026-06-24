"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics], analysis: str | None = None) -> str:
    """Render benchmark metrics to a markdown report.

    Keeps a comparison table; ``analysis`` (optional) is appended as a discussion section.
    """

    lines = [
        "# Benchmark Report",
        "",
        "Single-agent baseline vs multi-agent. Quality is scored separately via peer review.",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    if analysis:
        lines += ["", "## Analysis", "", analysis.strip()]

    return "\n".join(lines) + "\n"
