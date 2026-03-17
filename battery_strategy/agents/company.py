from __future__ import annotations

from typing import Any

from battery_strategy.agents.postprocess import build_evidence_bank, fallback_company_result
from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.tools.planning import build_company_queries
from battery_strategy.tools.prompts import company_prompt
from battery_strategy.utils.types import Axis, CompanyName, CompanyState


class CompanyAnalysisAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        company: CompanyName,
        goal: str,
        comparison_axes: list[Axis],
        *,
        target_axis: Axis | None = None,
        query_hint: str = "",
    ) -> CompanyState:
        queries = build_company_queries(
            company, goal, target_axis=target_axis, query_hint=query_hint
        )
        rag_hits = self._retrieve_rag(company, queries)
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
                rag_hits = self._retrieve_rag(company, queries)
                web_hits = (
                    self.runtime.searcher.search_many(queries)
                    if self.runtime.config.web_search.enabled
                    else []
                )
                balance = self.runtime.balance_checker.evaluate(queries, web_hits)

        evidence_bank = build_evidence_bank([*rag_hits, *web_hits])
        fallback = fallback_company_result(company, rag_hits, web_hits)
        instructions, user_prompt = company_prompt(
            company, goal, comparison_axes, rag_hits, web_hits
        )
        parsed = self.runtime.llm.json(
            instructions,
            user_prompt,
            fallback={
                "profile": fallback["profile"],
                "normalized_evidence": fallback["normalized_evidence"],
                "swot_inputs": fallback["swot_inputs"],
                "unresolved_conflicts": fallback["unresolved_conflicts"],
            },
        )

        return {
            "company_name": company,
            "comparison_axes": comparison_axes,
            "query_set": queries,
            "rag_hits": rag_hits,
            "web_hits": web_hits,
            "evidence_bank": evidence_bank,
            "normalized_evidence": parsed.get(
                "normalized_evidence", fallback["normalized_evidence"]
            ),
            "profile": parsed.get("profile", fallback["profile"]),
            "swot_inputs": parsed.get("swot_inputs", fallback["swot_inputs"]),
            "balance_flags": balance.flags,
            "unresolved_conflicts": parsed.get("unresolved_conflicts", []),
            "target_axis": target_axis,
            "retry_from": None,
        }

    def _retrieve_rag(self, company: CompanyName, queries: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for annotated_query in queries:
            label, raw_query = annotated_query.split("|", maxsplit=1)
            results.extend(
                self.runtime.retriever.search(
                    raw_query,
                    label=label,
                    source_groups=[company],
                )
            )
        deduped = {hit["chunk_id"]: hit for hit in results}
        return list(deduped.values())
