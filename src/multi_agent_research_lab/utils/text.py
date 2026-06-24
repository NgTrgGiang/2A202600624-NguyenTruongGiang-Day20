"""Small text helpers shared across agents and evaluation."""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[\d+\]")


def count_citation_markers(text: str | None) -> int:
    """Count bracketed citation markers like ``[1]`` in ``text``."""

    if not text:
        return 0
    return len(_CITATION_RE.findall(text))


def citation_coverage(answer: str | None, num_sources: int) -> float:
    """Fraction of available sources that are actually cited in the answer (0..1).

    A coarse proxy for grounding: distinct cited indices / available sources.
    """

    if not answer or num_sources <= 0:
        return 0.0
    cited = {int(m.strip("[]")) for m in _CITATION_RE.findall(answer)}
    cited = {c for c in cited if 1 <= c <= num_sources}
    return round(len(cited) / num_sources, 3)
