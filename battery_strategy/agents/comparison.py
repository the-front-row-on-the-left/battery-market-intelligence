from __future__ import annotations

from battery_strategy.agents.postprocess import fallback_comparison
from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.tools.prompts import comparison_prompt
from battery_strategy.utils.types import COMPARISON_AXES
from battery_strategy.utils.types import ComparisonState, GlobalState


class ComparisonAndSwotAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def run(self, global_state: GlobalState) -> ComparisonState:
        instructions, user_prompt = comparison_prompt(
            global_state["market_context"],
            global_state["company_results"],
            global_state["comparison_axes"],
        )
        rows, insights, swot = fallback_comparison(global_state["company_results"])
        parsed = self.runtime.llm.json(
            instructions,
            user_prompt,
            fallback={
                "comparison_matrix": rows,
                "insights": insights,
                "swot": swot,
                "validation_flags": [],
            },
        )
        normalized_rows = self._normalize_comparison_matrix(
            parsed.get("comparison_matrix", rows),
            rows,
        )
        return {
            "company_results": global_state["company_results"],
            "market_context": global_state["market_context"],
            "comparison_axes": global_state["comparison_axes"],
            "comparison_matrix": normalized_rows,
            "insights": parsed.get("insights", insights),
            "swot": parsed.get("swot", swot),
            "validation_flags": parsed.get("validation_flags", []),
        }

    @staticmethod
    def _normalize_comparison_matrix(
        parsed_rows: list[dict] | None,
        fallback_rows: list[dict],
    ) -> list[dict]:
        fallback_by_axis = {
            row.get("axis"): row for row in fallback_rows if row.get("axis") in COMPARISON_AXES
        }
        parsed_by_axis = {
            row.get("axis"): row
            for row in (parsed_rows or [])
            if isinstance(row, dict) and row.get("axis") in COMPARISON_AXES
        }

        normalized: list[dict] = []
        for axis in COMPARISON_AXES:
            base = dict(fallback_by_axis.get(axis, {}))
            base.update(parsed_by_axis.get(axis, {}))
            if not base:
                continue
            base["axis"] = axis
            normalized.append(base)
        return normalized
