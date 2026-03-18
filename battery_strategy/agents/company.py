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
        rag_only_fallback = fallback_company_result(company, rag_hits, [])
        instructions, user_prompt = company_prompt(
            company, goal, comparison_axes, rag_hits, []
        )
        rag_only_parsed = self.runtime.llm.json(
            instructions,
            user_prompt,
            fallback={
                "profile": rag_only_fallback["profile"],
                "normalized_evidence": rag_only_fallback["normalized_evidence"],
                "swot_inputs": rag_only_fallback["swot_inputs"],
                "unresolved_conflicts": rag_only_fallback["unresolved_conflicts"],
            },
        )
        profile = rag_only_parsed.get("profile", rag_only_fallback["profile"])
        normalized_evidence = rag_only_parsed.get(
            "normalized_evidence", rag_only_fallback["normalized_evidence"]
        )
        swot_inputs = rag_only_parsed.get("swot_inputs", rag_only_fallback["swot_inputs"])
        unresolved_conflicts = rag_only_parsed.get("unresolved_conflicts", [])

        web_hits: list[dict[str, Any]] = []
        balance_flags: list[str] = []
        if self.runtime.config.web_search.enabled and self._needs_web_augmentation(
            comparison_axes,
            profile,
            normalized_evidence,
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
            profile = parsed.get("profile", fallback["profile"])
            normalized_evidence = parsed.get(
                "normalized_evidence", fallback["normalized_evidence"]
            )
            swot_inputs = parsed.get("swot_inputs", fallback["swot_inputs"])
            unresolved_conflicts = parsed.get("unresolved_conflicts", [])
        else:
            evidence_bank = build_evidence_bank(rag_hits)

        return {
            "company_name": company,
            "comparison_axes": comparison_axes,
            "query_set": queries,
            "rag_hits": rag_hits,
            "web_hits": web_hits,
            "evidence_bank": evidence_bank,
            "normalized_evidence": normalized_evidence,
            "profile": profile,
            "swot_inputs": swot_inputs,
            "balance_flags": balance_flags,
            "unresolved_conflicts": unresolved_conflicts,
            "target_axis": target_axis,
            "retry_from": None,
        }

    @staticmethod
    def _needs_web_augmentation(
        comparison_axes: list[Axis],
        profile: dict[Axis, dict[str, Any]],
        normalized_evidence: list[dict[str, Any]],
        *,
        target_axis: Axis | None,
    ) -> bool:
        if len(normalized_evidence) < 4:
            return True
        if target_axis is not None:
            return not profile.get(target_axis, {}).get("summary", "").strip()
        covered_axes = [
            axis
            for axis in comparison_axes
            if profile.get(axis, {}).get("summary", "").strip()
        ]
        return len(covered_axes) < max(3, len(comparison_axes) // 2)

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
