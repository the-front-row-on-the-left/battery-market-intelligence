from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from battery_strategy.utils.common import domain_from_url, parse_annotated_query, utc_today
from battery_strategy.utils.types import SearchHit, SourceType

OFFICIAL_DOMAINS = {"lgensol.com", "catl.com"}
INDUSTRY_DOMAINS = {"iea.org"}
ACADEMIC_DOMAINS = {"sciencedirect.com", "springer.com", "nature.com", "ieee.org"}


def infer_source_type(url: str) -> SourceType:
    domain = domain_from_url(url)
    if any(domain.endswith(item) for item in OFFICIAL_DOMAINS):
        return "official_pr"
    if any(domain.endswith(item) for item in INDUSTRY_DOMAINS):
        return "industry_report"
    if any(domain.endswith(item) for item in ACADEMIC_DOMAINS):
        return "academic"
    return "news"


class BaseSearcher:
    enabled: bool = False

    def search_many(self, queries: Iterable[str]) -> list[SearchHit]:
        raise NotImplementedError


@dataclass(slots=True)
class NoOpSearcher(BaseSearcher):
    enabled: bool = False

    def search_many(self, queries: Iterable[str]) -> list[SearchHit]:
        return []


@dataclass(slots=True)
class DuckDuckGoSearcher(BaseSearcher):
    region: str = "kr-kr"
    max_results_per_query: int = 5
    fetch_full_text: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        try:
            from duckduckgo_search import DDGS
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "duckduckgo-search package is required for DuckDuckGoSearcher."
            ) from exc
        self._ddgs_cls = DDGS

    def search_many(self, queries: Iterable[str]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for annotated_query in queries:
            label, query = parse_annotated_query(annotated_query)
            with self._ddgs_cls() as ddgs:
                for item in ddgs.text(
                    query,
                    region=self.region,
                    safesearch="moderate",
                    max_results=self.max_results_per_query,
                ):
                    url = item.get("href", "")
                    snippet = item.get("body", "")
                    content = snippet
                    if self.fetch_full_text and url:
                        fetched = self._fetch_full_text(url)
                        if fetched:
                            content = fetched
                    hits.append(
                        {
                            "query": query,
                            "query_label": label,  # type: ignore[typeddict-item]
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": snippet,
                            "content": content,
                            "source_type": infer_source_type(url),  # type: ignore[typeddict-item]
                            "domain": domain_from_url(url),
                            "searched_at": utc_today(),
                        }
                    )
        return hits

    @staticmethod
    def _fetch_full_text(url: str) -> str:
        try:
            import trafilatura
        except Exception:  # noqa: BLE001
            return ""
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        return extracted or ""
