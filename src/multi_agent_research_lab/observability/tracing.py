"""Tracing hooks.

This file avoids binding to one provider. The skeleton uses a lightweight in-process span
plus a JSON trace export; LangSmith can be enabled via env when a key is present.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multi_agent_research_lab.core.config import Settings
    from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    Logs span duration; pair with :class:`JsonTraceExporter` (or LangSmith) for persistence.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("span %s finished in %.3fs", name, span["duration_seconds"])


class JsonTraceExporter:
    """Write a run's collected trace events to a JSON file under ``reports/``."""

    def __init__(self, root: Path = Path("reports")) -> None:
        self.root = root

    def export(self, state: ResearchState, run_name: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"trace_{run_name}.json"
        payload = {
            "run_name": run_name,
            "query": state.request.query,
            "route_history": state.route_history,
            "iterations": state.iteration,
            "errors": state.errors,
            "trace": state.trace,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("trace exported to %s", path)
        return path


def maybe_enable_langsmith(settings: Settings) -> bool:
    """Enable LangSmith tracing via env vars when a key is configured. No-op otherwise."""

    if not settings.langsmith_api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info("LangSmith tracing enabled for project %s", settings.langsmith_project)
    return True
