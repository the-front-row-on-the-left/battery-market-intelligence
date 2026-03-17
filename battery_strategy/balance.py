from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from battery_strategy.types import SearchHit
from battery_strategy.utils import parse_annotated_query


@dataclass(slots=True)
class SearchBalanceResult:
    flags: list[str]
    retry_queries: list[str]


@dataclass(slots=True)
class SearchBalanceChecker:
    def evaluate(self, annotated_queries: Iterable[str], web_hits: list[SearchHit]) -> SearchBalanceResult:
        labels = [parse_annotated_query(query)[0] for query in annotated_queries]
        label_counter = Counter(labels)
        hit_counter = Counter(hit["query_label"] for hit in web_hits)
        source_counter = Counter(hit["source_type"] for hit in web_hits)
        domain_counter = Counter(hit["domain"] for hit in web_hits if hit.get("domain"))

        flags: list[str] = []
        retry_queries: list[str] = []

        for label in ("neutral", "positive", "negative"):
            if label_counter.get(label, 0) > 0 and hit_counter.get(label, 0) == 0:
                flags.append(f"missing_{label}_web_hits")
                retry_queries.append(f"{label}|{label} evidence retry")

        if len(domain_counter) < 2 and web_hits:
            flags.append("low_domain_diversity")

        if source_counter and len(source_counter) == 1:
            flags.append("source_type_skew")

        return SearchBalanceResult(flags=flags, retry_queries=retry_queries)
