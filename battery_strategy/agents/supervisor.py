from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

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
from battery_strategy.graph import build_supervisor_graph
from battery_strategy.utils.types import CompanyName, GlobalState
from langgraph.types import Send
from rich.console import Console


@dataclass(slots=True)
class Supervisor:
    runtime: AgentRuntime
    market_agent: MarketAnalysisAgent = field(init=False)
    company_agent: CompanyAnalysisAgent = field(init=False)
    comparison_agent: ComparisonAndSwotAgent = field(init=False)
    bias_audit_agent: BiasAuditAgent = field(init=False)
    writer_agent: WriterAgent = field(init=False)
    graph: Any = field(init=False, repr=False)
    console: Console = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.console = Console(stderr=True)
        self.market_agent = MarketAnalysisAgent(self.runtime)
        self.company_agent = CompanyAnalysisAgent(self.runtime)
        self.comparison_agent = ComparisonAndSwotAgent(self.runtime)
        self.bias_audit_agent = BiasAuditAgent(self.runtime)
        self.writer_agent = WriterAgent(self.runtime)
        self.graph = build_supervisor_graph(self)

    def run(self, global_state: GlobalState) -> GlobalState:
        return self.graph.invoke(global_state)

    def _dispatch_research_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["status"] = "running"
        updated["next_step"] = "dispatch_research"
        companies = list(self.runtime.config.workflow.companies)
        if companies:
            self._log(f"시장/회사 분석 병렬 시작 - MARKET, {', '.join(companies)}")
        else:
            self._log("시장 분석 시작")
        return updated  # type: ignore[return-value]

    def _dispatch_research_sends(self, global_state: GlobalState) -> list[Send]:
        sends: list[Send] = [Send("market", global_state)]
        for company in self.runtime.config.workflow.companies:
            sends.append(
                Send(
                    "company_worker",
                    {
                        "company_name": company,
                        "goal": global_state["goal"],
                        "comparison_axes": global_state["comparison_axes"],
                        "is_retry": False,
                        "retry_plan": global_state.get("retry_plan"),
                    },
                )
            )
        return sends

    def _market_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["next_step"] = "market"
        self._log("시장 분석 시작")
        started = perf_counter()
        market_state = self.market_agent.run(updated["goal"], updated["comparison_axes"])
        merge_market_into_global(updated, market_state)
        self._log(
            f"시장 분석 완료 ({self._elapsed(started)}) - 근거 {len(market_state.get('normalized_evidence', []))}건"
        )
        return updated  # type: ignore[return-value]

    def _dispatch_retry_company_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["next_step"] = "retry_company_dispatch"
        updated["retry_count"] = updated.get("retry_count", 0) + 1
        retry_plan = updated.get("retry_plan")
        if retry_plan and retry_plan.get("target_company"):
            updated["retry_history"] = updated.get("retry_history", []) + [
                self._retry_signature(retry_plan)
            ]
            self._log(
                "회사 재시도 시작 - "
                f"{retry_plan['target_company']} / axis={retry_plan.get('target_axis') or 'all'}"
                f" / from={retry_plan.get('retry_from') or '-'}"
                f" / reason={retry_plan.get('reason') or '-'}"
                + (
                    f" / hint={retry_plan.get('query_hint')}"
                    if retry_plan.get("query_hint")
                    else ""
                )
            )
        return updated  # type: ignore[return-value]

    def _dispatch_retry_company_send(self, global_state: GlobalState) -> list[Send]:
        retry_plan = global_state.get("retry_plan")
        if not retry_plan or not retry_plan.get("target_company"):
            return []
        return [
            Send(
                "company_worker",
                {
                    "company_name": retry_plan["target_company"],
                    "goal": global_state["goal"],
                    "comparison_axes": global_state["comparison_axes"],
                    "is_retry": True,
                    "retry_plan": retry_plan,
                },
            )
        ]

    def _company_parallel_node(self, state: dict[str, Any]) -> GlobalState:
        company = state["company_name"]
        goal = state["goal"]
        comparison_axes = state["comparison_axes"]
        retry_plan = state.get("retry_plan")
        is_retry = bool(state.get("is_retry"))
        phase = "회사 재분석" if is_retry else "회사 분석"
        self._log(f"{phase} 시작 - {company}")
        started = perf_counter()

        target_axis = None
        query_hint = ""
        if is_retry and retry_plan:
            target_axis = retry_plan["target_axis"]
            query_hint = retry_plan["query_hint"]

        company_state = self.company_agent.run(
            company,
            goal,
            comparison_axes,
            target_axis=target_axis,
            query_hint=query_hint,
        )
        partial_state: GlobalState = {
            "goal": goal,
            "criteria": [],
            "comparison_axes": comparison_axes,
            "corpus_manifest": [],
            "query_plan": {},
            "market_context": {"summary": "", "bullet_points": [], "normalized_evidence": []},
            "company_results": {},
            "comparison_matrix": [],
            "swot": {},
            "bias_flags": [],
            "unresolved_conflicts": [],
            "retry_plan": retry_plan,
            "next_step": f"company:{company}",
            "status": "running",
            "draft_sections": {},
            "references": [],
            "retry_count": 0,
        }
        merge_company_into_global(partial_state, company_state)
        self._log(
            f"{phase} 완료 - {company} ({self._elapsed(started)}) "
            f"/ 근거 {len(company_state.get('normalized_evidence', []))}건"
        )
        return {
            "company_results": partial_state["company_results"],
            "unresolved_conflicts": partial_state["unresolved_conflicts"],
        }  # type: ignore[return-value]

    def _comparison_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["next_step"] = "comparison"
        self._log("비교/SWOT 분석 시작")
        started = perf_counter()
        comparison_state = self.comparison_agent.run(updated)
        merge_comparison_into_global(updated, comparison_state)
        self._log(
            f"비교/SWOT 분석 완료 ({self._elapsed(started)}) - "
            f"비교축 {len(comparison_state.get('comparison_matrix', []))}개"
        )
        return updated  # type: ignore[return-value]

    def _bias_audit_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["next_step"] = "bias_audit"
        self._log("편향 점검 시작")
        started = perf_counter()
        audit_result = self.bias_audit_agent.run(updated)
        updated["bias_flags"] = audit_result.bias_flags
        updated["retry_plan"] = audit_result.retry_recommendation
        if audit_result.retry_recommendation:
            retry = audit_result.retry_recommendation
            flags_text = ", ".join(audit_result.bias_flags) if audit_result.bias_flags else "없음"
            self._log(
                f"편향 점검 완료 ({self._elapsed(started)}) - flags: {flags_text}"
            )
            self._log(
                "재시도 계획 - "
                f"scope={retry['target_scope']}"
                f" / company={retry.get('target_company') or '-'}"
                f" / axis={retry.get('target_axis') or '-'}"
                f" / from={retry.get('retry_from') or '-'}"
                f" / reason={retry.get('reason') or '-'}"
                + (
                    f" / hint={retry.get('query_hint')}"
                    if retry.get("query_hint")
                    else ""
                )
                + f" / history={len(updated.get('retry_history', []))}"
            )
        else:
            flags_text = ", ".join(audit_result.bias_flags) if audit_result.bias_flags else "없음"
            self._log(
                f"편향 점검 완료 ({self._elapsed(started)}) - flags: {flags_text} / 추가 재시도 없음"
            )
        return updated  # type: ignore[return-value]

    def _retry_market_node(self, global_state: GlobalState) -> GlobalState:
        retry_plan = global_state.get("retry_plan")
        updated = dict(global_state)
        updated["next_step"] = "retry_market"
        if not retry_plan:
            return updated  # type: ignore[return-value]
        updated["retry_history"] = updated.get("retry_history", []) + [
            self._retry_signature(retry_plan)
        ]
        self._log(
            "시장 재시도 시작 - "
            f"axis={retry_plan.get('target_axis') or 'all'}"
            f" / from={retry_plan.get('retry_from') or '-'}"
            f" / reason={retry_plan.get('reason') or '-'}"
            + (
                f" / hint={retry_plan.get('query_hint')}"
                if retry_plan.get("query_hint")
                else ""
            )
        )
        started = perf_counter()
        market_state = self.market_agent.run(
            updated["goal"],
            updated["comparison_axes"],
            target_axis=retry_plan["target_axis"],
            query_hint=retry_plan["query_hint"],
        )
        merge_market_into_global(updated, market_state)
        updated["retry_count"] = updated.get("retry_count", 0) + 1
        self._log(f"시장 재시도 완료 ({self._elapsed(started)})")
        return updated  # type: ignore[return-value]

    def _collect_references_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["references"] = collect_references(updated)
        updated["retry_plan"] = None
        self._log(f"참고문헌 수집 완료 - {len(updated['references'])}건")
        return updated  # type: ignore[return-value]

    def _retry_comparison_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["next_step"] = "retry_comparison"
        updated["retry_count"] = updated.get("retry_count", 0) + 1
        retry_plan = updated.get("retry_plan")
        if retry_plan:
            updated["retry_history"] = updated.get("retry_history", []) + [
                self._retry_signature(retry_plan)
            ]
        self._log(
            "비교 재시도 준비 - "
            f"from={(retry_plan.get('retry_from') if retry_plan else '-') or '-'}"
            f" / reason={(retry_plan.get('reason') if retry_plan else '-') or '-'}"
            + (
                f" / hint={retry_plan.get('query_hint')}"
                if retry_plan and retry_plan.get("query_hint")
                else ""
            )
        )
        return updated  # type: ignore[return-value]

    def _writer_node(self, global_state: GlobalState) -> GlobalState:
        updated = dict(global_state)
        updated["next_step"] = "writer"
        self._log("보고서 작성 시작")
        started = perf_counter()
        written = self.writer_agent.run(updated)
        written["status"] = "completed"
        written["next_step"] = "done"
        written["retry_plan"] = None
        self._log(f"보고서 작성 완료 ({self._elapsed(started)})")
        return written

    def _route_after_bias_audit(self, global_state: GlobalState) -> str:
        retry_plan = global_state.get("retry_plan")
        retry_count = global_state.get("retry_count", 0)
        if not retry_plan:
            return "collect_references"
        if retry_count >= self.runtime.config.execution.max_bias_retries:
            return "collect_references"

        target_scope = retry_plan["target_scope"]
        if target_scope == "market":
            return "retry_market"
        if target_scope == "company" and retry_plan.get("target_company"):
            available_companies = set(self.runtime.config.workflow.companies)
            if retry_plan["target_company"] in available_companies:
                return "retry_company_dispatch"
            return "collect_references"
        if target_scope == "comparison":
            return "retry_comparison"
        return "collect_references"

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [bold cyan]Supervisor[/bold cyan] {message}")

    @staticmethod
    def _elapsed(started: float) -> str:
        return f"{perf_counter() - started:.1f}s"

    @staticmethod
    def _retry_signature(retry_plan: dict[str, Any]) -> str:
        return "::".join(
            [
                str(retry_plan.get("target_scope") or "-"),
                str(retry_plan.get("target_company") or "-"),
                str(retry_plan.get("target_axis") or "-"),
                str(retry_plan.get("retry_from") or "-"),
            ]
        )
