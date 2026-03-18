from __future__ import annotations

from dataclasses import dataclass

from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.tools.prompts import bias_audit_prompt
from battery_strategy.utils.types import GlobalState, RetryPlan


@dataclass(slots=True)
class BiasAuditResult:
    bias_flags: list[str]
    retry_recommendation: RetryPlan | None


class BiasAuditAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def run(self, global_state: GlobalState) -> BiasAuditResult:
        flags = self._deterministic_flags(global_state)
        recommendation = self._deterministic_recommendation(global_state, flags)

        instructions, user_prompt = bias_audit_prompt(global_state)
        parsed = self.runtime.llm.json(
            instructions,
            user_prompt,
            fallback={
                "bias_flags": flags,
                "retry_recommendation": recommendation,
            },
        )
        parsed_flags = parsed.get("bias_flags", flags)
        parsed_recommendation = parsed.get("retry_recommendation", recommendation)
        return BiasAuditResult(
            bias_flags=parsed_flags,
            retry_recommendation=parsed_recommendation,
        )

    @staticmethod
    def _deterministic_flags(global_state: GlobalState) -> list[str]:
        flags: list[str] = []
        comparison_axes = global_state["comparison_axes"]
        market_evidence_count = len(global_state.get("market_context", {}).get("normalized_evidence", []))
        if market_evidence_count < 4:
            flags.append("low_market_evidence")
        for company_name, result in global_state.get("company_results", {}).items():
            missing_axes = [
                axis
                for axis in comparison_axes
                if not result["profile"].get(axis, {}).get("summary")
            ]
            evidence_count = len(result.get("normalized_evidence", []))
            if missing_axes:
                flags.append(f"missing_axis::{company_name}::{','.join(missing_axes)}")
            if evidence_count < 5:
                flags.append(f"low_company_evidence::{company_name}::{evidence_count}")
            if len(result.get("balance_flags", [])) > 0:
                flags.append(f"search_balance::{company_name}")
        if global_state.get("unresolved_conflicts"):
            flags.append("unresolved_conflicts")
        if not global_state.get("comparison_matrix"):
            flags.append("missing_comparison_matrix")
        if (
            market_evidence_count < 4
            or any(len(result.get("normalized_evidence", [])) < 5 for result in global_state.get("company_results", {}).values())
        ):
            flags.append("weak_comparison_support")
        return flags

    @staticmethod
    def _deterministic_recommendation(
        global_state: GlobalState,
        flags: list[str],
    ) -> RetryPlan | None:
        if not flags:
            return None
        for flag in flags:
            if flag.startswith("low_company_evidence::"):
                _, company, _ = flag.split("::", maxsplit=2)
                return {
                    "target_scope": "company",
                    "target_company": company,  # type: ignore[typeddict-item]
                    "target_axis": "risk",  # type: ignore[typeddict-item]
                    "retry_from": "query",
                    "reason": "기업 근거 부족",
                    "query_hint": "Focus on official disclosures, primary evidence, and external validation for thinly supported axes.",
                }
            if flag.startswith("missing_axis::"):
                _, company, axes = flag.split("::", maxsplit=2)
                axis = axes.split(",", maxsplit=1)[0]
                return {
                    "target_scope": "company",
                    "target_company": company,  # type: ignore[typeddict-item]
                    "target_axis": axis,  # type: ignore[typeddict-item]
                    "retry_from": "query",
                    "reason": "근거 부족",
                    "query_hint": f"Focus on {axis} evidence with numeric metrics and external validation.",
                }
        if "low_market_evidence" in flags:
            return {
                "target_scope": "market",
                "target_company": None,
                "target_axis": None,
                "retry_from": "query",
                "reason": "시장 근거 부족",
                "query_hint": "Focus on primary or industry-report evidence for market demand, policy, and battery supply chain risks in 2024-2026.",
            }
        if "missing_comparison_matrix" in flags:
            return {
                "target_scope": "comparison",
                "target_company": None,
                "target_axis": None,
                "retry_from": "compare",
                "reason": "비교 결과 누락",
                "query_hint": "Rebuild comparison matrix using existing company profiles.",
            }
        if "unresolved_conflicts" in flags:
            return {
                "target_scope": "comparison",
                "target_company": None,
                "target_axis": None,
                "retry_from": "compare",
                "reason": "충돌",
                "query_hint": "Surface conflicting metrics and keep both values with basis labels.",
            }
        return None
