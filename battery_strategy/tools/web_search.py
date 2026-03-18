from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from battery_strategy.utils.common import domain_from_url, parse_annotated_query, utc_today
from battery_strategy.utils.types import SearchHit, SourceType

OFFICIAL_DOMAINS = {"lgensol.com", "catl.com"}
INDUSTRY_DOMAINS = {"iea.org", "bnef.com", "about.bnef.com", "sneresearch.com"}
ACADEMIC_DOMAINS = {"sciencedirect.com", "springer.com", "nature.com", "ieee.org"}
BLOCKED_DOMAINS = {
    "zerohedge.com",
    "enkiai.com",
    "zhihu.com",
    "zhidao.baidu.com",
    "stocktitan.net",
    "marketscreener.com",
}
PREFERRED_NEWS_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "nikkei.com",
    "koreajoongangdaily.joins.com",
    "joongang.co.kr",
    "energy-storage.news",
    "ess-news.com",
    "pv-magazine.com",
}
OFFICIAL_PATH_KEYWORDS = (
    "newsroom",
    "news",
    "press",
    "media",
    "investor",
    "ir",
    "report",
    "annual",
    "sustainability",
    "esg",
    "business",
    "article",
)
LOW_SIGNAL_PATTERNS = (
    "i accept",
    "accept cookies",
    "cookie policy",
    "use of cookies",
    "privacy preference",
    "privacy policy",
    "cookie settings",
    "allow all cookies",
    "this website uses cookies",
    "grant us the permission",
    "consent with usage of cookies",
)
GARBLED_PATTERNS = (
    "by clicking on the button",
    "express consent with usage of cookies",
    "personal data about your activity",
)
TITLE_BLOCKLIST = (
    "cookie",
    "privacy",
)


def infer_source_type(url: str) -> SourceType:
    domain = domain_from_url(url)
    if any(domain.endswith(item) for item in OFFICIAL_DOMAINS):
        return "official_pr"
    if any(domain.endswith(item) for item in INDUSTRY_DOMAINS):
        return "industry_report"
    if any(domain.endswith(item) for item in ACADEMIC_DOMAINS):
        return "academic"
    return "news"


def is_blocked_url(url: str) -> bool:
    domain = domain_from_url(url)
    return any(domain.endswith(item) for item in BLOCKED_DOMAINS)


def is_low_signal_text(text: str) -> bool:
    lowered = (text or "").lower()
    if len(lowered.strip()) < 80:
        return True
    if any(pattern in lowered for pattern in LOW_SIGNAL_PATTERNS):
        return True
    # Filter text dominated by boilerplate/cookie banners.
    unique_words = {token for token in lowered.split() if len(token) > 2}
    return len(unique_words) < 20


def is_homepage(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path in {"", "/"} and not parsed.query


def has_preferred_domain(url: str) -> bool:
    domain = domain_from_url(url)
    return any(domain.endswith(item) for item in (*OFFICIAL_DOMAINS, *INDUSTRY_DOMAINS, *PREFERRED_NEWS_DOMAINS))


def has_allowed_official_path(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if not path or path in {"/", "/index.html"}:
        return False
    return any(keyword in path for keyword in OFFICIAL_PATH_KEYWORDS)


def is_low_signal_title(title: str) -> bool:
    lowered = (title or "").strip().lower()
    if len(lowered) < 8:
        return True
    return any(token in lowered for token in TITLE_BLOCKLIST)


def looks_garbled(text: str) -> bool:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in GARBLED_PATTERNS):
        return True
    if cleaned.count("�") > 0:
        return True
    if cleaned.count("http") >= 2:
        return True
    long_tokens = [token for token in cleaned.split() if len(token) >= 35]
    return len(long_tokens) >= 4


def is_low_quality_result(url: str, title: str, snippet: str, content: str) -> bool:
    if is_blocked_url(url):
        return True
    source_type = infer_source_type(url)
    if is_homepage(url):
        return True
    if source_type == "official_pr" and not has_allowed_official_path(url):
        return True
    if is_low_signal_title(title):
        return True
    if is_low_signal_text(snippet) and is_low_signal_text(content):
        return True
    if looks_garbled(content or snippet):
        return True
    return False


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
    _ddgs_cls: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from ddgs import DDGS
        except Exception as exc:  # noqa: BLE001
            try:
                from duckduckgo_search import DDGS
            except Exception:
                raise RuntimeError("ddgs package is required for DuckDuckGoSearcher.") from exc
        self._ddgs_cls = DDGS

    def search_many(self, queries: Iterable[str]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen_urls: set[str] = set()
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
                    title = item.get("title", "")
                    if not url or url in seen_urls:
                        continue
                    if not has_preferred_domain(url) and len(hits) >= self.max_results_per_query * 2:
                        continue
                    snippet = item.get("body", "")
                    if is_low_signal_text(snippet) or is_low_signal_title(title):
                        continue
                    content = snippet
                    if self.fetch_full_text and url:
                        fetched = self._fetch_full_text(url)
                        if fetched and not is_low_signal_text(fetched):
                            content = fetched
                    if is_low_quality_result(url, title, snippet, content):
                        continue
                    seen_urls.add(url)
                    hits.append(
                        {
                            "query": query,
                            "query_label": label,  # type: ignore[typeddict-item]
                            "title": title,
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
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_precision=True,
        )
        return extracted or ""
