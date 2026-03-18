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
        rag_only_fallback = {
            "market_context": fallback_market_context(rag_hits, []),
            "unresolved_conflicts": [],
        }
        instructions, user_prompt = market_prompt(goal, comparison_axes, rag_hits, [])
        rag_only_parsed = self.runtime.llm.json(
            instructions,
            user_prompt,
            fallback=rag_only_fallback,
        )
        market_context = rag_only_parsed.get("market_context", rag_only_fallback["market_context"])
        unresolved_conflicts = rag_only_parsed.get("unresolved_conflicts", [])

        web_hits: list[dict[str, Any]] = []
        balance_flags: list[str] = []
        if self.runtime.config.web_search.enabled and self._needs_web_augmentation(
            market_context,
            target_axis=target_axis,
        ):
            web_hits = self.runtime.searcher.search_many(queries)
            balance = self.runtime.balance_checker.evaluate(queries, web_hits)
            balance_flags = list(balance.flags)
            if balance.flags and self.runtime.config.execution.max_subgraph_balance_retries > 0:
                retry_queries = [query for query in balance.retry_queries if query.strip()]
                if retry_queries:
                    queries = queries + retry_queries
                    extra_web_hits = self.runtime.searcher.search_many(retry_queries)
                    web_hits = [*web_hits, *extra_web_hits]
                    balance = self.runtime.balance_checker.evaluate(queries, web_hits)
                    balance_flags = list(balance.flags)

            evidence_bank = build_evidence_bank([*rag_hits, *web_hits])
            fallback = {
                "market_context": fallback_market_context(rag_hits, web_hits),
                "unresolved_conflicts": [],
            }
            instructions, user_prompt = market_prompt(goal, comparison_axes, rag_hits, web_hits)
            parsed = self.runtime.llm.json(instructions, user_prompt, fallback=fallback)
            market_context = parsed.get("market_context", fallback["market_context"])
            unresolved_conflicts = parsed.get("unresolved_conflicts", [])
        else:
            evidence_bank = build_evidence_bank(rag_hits)

        return {
            "query_set": queries,
            "rag_hits": rag_hits,
            "web_hits": web_hits,
            "evidence_bank": evidence_bank,
            "normalized_evidence": market_context.get("normalized_evidence", []),
            "market_context": market_context,
            "balance_flags": balance_flags,
            "unresolved_conflicts": unresolved_conflicts,
            "target_axis": target_axis,
            "retry_from": None,
        }

    @staticmethod
    def _needs_web_augmentation(
        market_context: dict[str, Any],
        *,
        target_axis: Axis | None,
    ) -> bool:
        normalized = market_context.get("normalized_evidence", [])
        if len(normalized) < 3:
            return True
        if not market_context.get("summary", "").strip():
            return True
        if target_axis is None:
            return False
        return not any(item.get("axis") == target_axis for item in normalized)

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
