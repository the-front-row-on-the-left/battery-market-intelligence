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
        parsed_recommendation = self._validated_recommendation(
            global_state,
            parsed.get("retry_recommendation", recommendation),
            fallback=recommendation,
        )
        return BiasAuditResult(
            bias_flags=parsed_flags,
            retry_recommendation=parsed_recommendation,
        )

    @staticmethod
    def _retry_signature(plan: RetryPlan) -> str:
        return "::".join(
            [
                str(plan.get("target_scope") or "-"),
                str(plan.get("target_company") or "-"),
                str(plan.get("target_axis") or "-"),
                str(plan.get("retry_from") or "-"),
            ]
        )

    @classmethod
    def _validated_recommendation(
        cls,
        global_state: GlobalState,
        recommendation: RetryPlan | None,
        *,
        fallback: RetryPlan | None,
    ) -> RetryPlan | None:
        retry_history = set(global_state.get("retry_history", []))
        if recommendation is not None:
            signature = cls._retry_signature(recommendation)
            if signature not in retry_history:
                return recommendation
        if fallback is not None:
            signature = cls._retry_signature(fallback)
            if signature not in retry_history:
                return fallback
        return None

    @staticmethod
    def _deterministic_flags(global_state: GlobalState) -> list[str]:
        flags: list[str] = []
        comparison_axes = global_state["comparison_axes"]
        for company_name, result in global_state.get("company_results", {}).items():
            missing_axes = [
                axis
                for axis in comparison_axes
                if not result["profile"].get(axis, {}).get("summary")
            ]
            if missing_axes:
                flags.append(f"missing_axis::{company_name}::{','.join(missing_axes)}")
            if len(result.get("balance_flags", [])) > 0:
                flags.append(f"search_balance::{company_name}")
        if global_state.get("unresolved_conflicts"):
            flags.append("unresolved_conflicts")
        if not global_state.get("comparison_matrix"):
            flags.append("missing_comparison_matrix")
        return flags

    @staticmethod
    def _deterministic_recommendation(
        global_state: GlobalState,
        flags: list[str],
    ) -> RetryPlan | None:
        if not flags:
            return None
        retry_history = set(global_state.get("retry_history", []))
        for flag in flags:
            if flag.startswith("missing_axis::"):
                _, company, axes = flag.split("::", maxsplit=2)
                candidate_axes = [axis for axis in axes.split(",") if axis]
                for axis in candidate_axes:
                    recommendation: RetryPlan = {
                        "target_scope": "company",
                        "target_company": company,  # type: ignore[typeddict-item]
                        "target_axis": axis,  # type: ignore[typeddict-item]
                        "retry_from": "query",
                        "reason": "근거 부족",
                        "query_hint": f"Focus on {axis} evidence with numeric metrics and external validation.",
                    }
                    if BiasAuditAgent._retry_signature(recommendation) not in retry_history:
                        return recommendation
        if "missing_comparison_matrix" in flags:
            recommendation = {
                "target_scope": "comparison",
                "target_company": None,
                "target_axis": None,
                "retry_from": "compare",
                "reason": "비교 결과 누락",
                "query_hint": "Rebuild comparison matrix using existing company profiles.",
            }
            if BiasAuditAgent._retry_signature(recommendation) not in retry_history:
                return recommendation
        if "unresolved_conflicts" in flags:
            recommendation = {
                "target_scope": "comparison",
                "target_company": None,
                "target_axis": None,
                "retry_from": "compare",
                "reason": "충돌",
                "query_hint": "Surface conflicting metrics and keep both values with basis labels.",
            }
            if BiasAuditAgent._retry_signature(recommendation) not in retry_history:
                return recommendation
        return None
