from __future__ import annotations

from battery_strategy.postprocess import fallback_comparison
from battery_strategy.prompts import comparison_prompt
from battery_strategy.runtime import AgentRuntime
from battery_strategy.types import ComparisonState, GlobalState


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
        return {
            "company_results": global_state["company_results"],
            "market_context": global_state["market_context"],
            "comparison_axes": global_state["comparison_axes"],
            "comparison_matrix": parsed.get("comparison_matrix", rows),
            "insights": parsed.get("insights", insights),
            "swot": parsed.get("swot", swot),
            "validation_flags": parsed.get("validation_flags", []),
        }
