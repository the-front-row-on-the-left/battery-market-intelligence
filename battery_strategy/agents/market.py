from __future__ import annotations

from typing import Any

from battery_strategy.agents.postprocess import build_evidence_bank, fallback_market_context
from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.tools.planning import build_market_queries
from battery_strategy.tools.prompts import market_prompt
from battery_strategy.utils.types import Axis, MarketState


class MarketAnalysisAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        goal: str,
        comparison_axes: list[Axis],
        *,
        target_axis: Axis | None = None,
        query_hint: str = "",
    ) -> MarketState:
        queries = build_market_queries(goal, target_axis=target_axis, query_hint=query_hint)
        rag_hits = self._retrieve_rag(queries)
        web_hits = (
            self.runtime.searcher.search_many(queries)
            if self.runtime.config.web_search.enabled
            else []
        )

        balance = self.runtime.balance_checker.evaluate(queries, web_hits)
        if balance.flags and self.runtime.config.execution.max_subgraph_balance_retries > 0:
            retry_queries = [query for query in balance.retry_queries if query.strip()]
            if retry_queries:
                queries = queries + retry_queries
                rag_hits = self._retrieve_rag(queries)
                web_hits = (
                    self.runtime.searcher.search_many(queries)
                    if self.runtime.config.web_search.enabled
                    else []
                )
                balance = self.runtime.balance_checker.evaluate(queries, web_hits)

        evidence_bank = build_evidence_bank([*rag_hits, *web_hits])
        fallback = {
            "market_context": fallback_market_context(rag_hits, web_hits),
            "unresolved_conflicts": [],
        }
        instructions, user_prompt = market_prompt(goal, comparison_axes, rag_hits, web_hits)
        parsed = self.runtime.llm.json(instructions, user_prompt, fallback=fallback)
        market_context = parsed.get("market_context", fallback["market_context"])

        return {
            "query_set": queries,
            "rag_hits": rag_hits,
            "web_hits": web_hits,
            "evidence_bank": evidence_bank,
            "normalized_evidence": market_context.get("normalized_evidence", []),
            "market_context": market_context,
            "balance_flags": balance.flags,
            "unresolved_conflicts": parsed.get("unresolved_conflicts", []),
            "target_axis": target_axis,
            "retry_from": None,
        }

    def _retrieve_rag(self, queries: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for annotated_query in queries:
            label, raw_query = annotated_query.split("|", maxsplit=1)
            results.extend(
                self.runtime.retriever.search(
                    raw_query,
                    label=label,
                    source_groups=["MARKET"],
                )
            )
        deduped = {hit["chunk_id"]: hit for hit in results}
        return list(deduped.values())
