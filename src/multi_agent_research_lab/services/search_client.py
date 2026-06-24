"""Search client abstraction for ResearcherAgent.

Agents depend on :class:`SearchClient`; the concrete provider is selected by
:func:`get_search_client`. A deterministic mock keeps the pipeline runnable without keys.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Tavily-backed, provider-agnostic search client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY is required for SearchClient. "
                "Use get_search_client() for a mock fallback."
            )
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=self.settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        response = self._client.search(query=query, max_results=max_results)
        results = response.get("results", []) if isinstance(response, dict) else []
        documents: list[SourceDocument] = []
        for item in results[:max_results]:
            documents.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score")},
                )
            )
        return documents


class MockSearchClient(SearchClient):
    """Deterministic offline search used for tests and key-less smoke runs."""

    def __init__(self, settings: Settings | None = None) -> None:  # noqa: D401 - keep signature
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for i in range(1, max_results + 1):
            documents.append(
                SourceDocument(
                    title=f"Mock source {i} for: {query[:60]}",
                    url=f"https://example.com/mock/{i}",
                    snippet=(
                        f"Synthetic snippet {i} discussing '{query[:80]}'. "
                        "Used offline so the research pipeline runs without a search API key."
                    ),
                    metadata={"mock": True, "rank": i},
                )
            )
        return documents


def get_search_client(settings: Settings | None = None) -> SearchClient:
    """Return a Tavily client when a key is present, otherwise a deterministic mock."""

    settings = settings or get_settings()
    if settings.tavily_api_key:
        return SearchClient(settings)
    logger.warning("TAVILY_API_KEY not set - falling back to MockSearchClient (synthetic sources).")
    return MockSearchClient(settings)
