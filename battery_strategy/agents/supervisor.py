from __future__ import annotations

from dataclasses import dataclass

from battery_strategy.agents.bias_audit import BiasAuditAgent
from battery_strategy.agents.company import CompanyAnalysisAgent
from battery_strategy.agents.comparison import ComparisonAndSwotAgent
from battery_strategy.agents.market import MarketAnalysisAgent
from battery_strategy.agents.postprocess import (
    collect_references,
    merge_company_into_global,
    merge_comparison_into_global,
    merge_market_into_global,
)
from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.agents.writer import WriterAgent
from battery_strategy.utils.types import GlobalState, RetryPlan


@dataclass(slots=True)
class Supervisor:
    runtime: AgentRuntime

    def __post_init__(self) -> None:
        self.market_agent = MarketAnalysisAgent(self.runtime)
        self.company_agent = CompanyAnalysisAgent(self.runtime)
        self.comparison_agent = ComparisonAndSwotAgent(self.runtime)
        self.bias_audit_agent = BiasAuditAgent(self.runtime)
        self.writer_agent = WriterAgent(self.runtime)

    def run(self, global_state: GlobalState) -> GlobalState:
        global_state["status"] = "running"
        global_state["next_step"] = "market"

        market_state = self.market_agent.run(global_state["goal"], global_state["comparison_axes"])
        merge_market_into_global(global_state, market_state)

        for company in self.runtime.config.workflow.companies:
            global_state["next_step"] = f"company:{company}"
            company_state = self.company_agent.run(
                company,
                global_state["goal"],
                global_state["comparison_axes"],
            )
            merge_company_into_global(global_state, company_state)

        global_state["next_step"] = "comparison"
        comparison_state = self.comparison_agent.run(global_state)
        merge_comparison_into_global(global_state, comparison_state)

        retries = 0
        while retries < self.runtime.config.execution.max_bias_retries:
            global_state["next_step"] = "bias_audit"
            audit_result = self.bias_audit_agent.run(global_state)
            global_state["bias_flags"] = audit_result.bias_flags
            if not audit_result.retry_recommendation:
                break
            global_state["retry_plan"] = audit_result.retry_recommendation
            self._apply_retry(global_state, audit_result.retry_recommendation)
            comparison_state = self.comparison_agent.run(global_state)
            merge_comparison_into_global(global_state, comparison_state)
            retries += 1

        global_state["references"] = collect_references(global_state)
        global_state["next_step"] = "writer"
        global_state = self.writer_agent.run(global_state)
        global_state["status"] = "completed"
        global_state["next_step"] = "done"
        return global_state

    def _apply_retry(self, global_state: GlobalState, retry_plan: RetryPlan) -> None:
        target_scope = retry_plan["target_scope"]
        if target_scope == "market":
            market_state = self.market_agent.run(
                global_state["goal"],
                global_state["comparison_axes"],
                target_axis=retry_plan["target_axis"],
                query_hint=retry_plan["query_hint"],
            )
            merge_market_into_global(global_state, market_state)
            return

        if target_scope == "company" and retry_plan["target_company"]:
            company_state = self.company_agent.run(
                retry_plan["target_company"],
                global_state["goal"],
                global_state["comparison_axes"],
                target_axis=retry_plan["target_axis"],
                query_hint=retry_plan["query_hint"],
            )
            merge_company_into_global(global_state, company_state)
            return

        if target_scope == "comparison":
            comparison_state = self.comparison_agent.run(global_state)
            merge_comparison_into_global(global_state, comparison_state)
