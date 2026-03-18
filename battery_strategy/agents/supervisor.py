from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Annotated, Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

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
from battery_strategy.utils.logging import get_logger
from battery_strategy.utils.types import AXIS_VALUES, CompanyName, CompanyResult, GlobalState, RetryPlan


NextAction = Literal[
    "dispatch_initial",
    "run_comparison",
    "run_bias_audit",
    "retry_market",
    "retry_company_lges",
    "retry_company_catl",
    "retry_comparison",
    "run_writer",
    "finish",
]


def _merge_company_results(
    left: dict[CompanyName, CompanyResult],
    right: dict[CompanyName, CompanyResult],
) -> dict[CompanyName, CompanyResult]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _dedupe_merge(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys((left or []) + (right or [])))


class SupervisorGraphState(TypedDict):
    goal: str
    criteria: list[str]
    comparison_axes: list[str]
    corpus_manifest: list[dict[str, Any]]
    query_plan: dict[str, list[str]]
    market_context: dict[str, Any]
    company_results: Annotated[dict[CompanyName, CompanyResult], _merge_company_results]
    comparison_matrix: list[dict[str, Any]]
    swot: dict[str, dict[str, list[str]]]
    bias_flags: list[str]
    unresolved_conflicts: Annotated[list[str], _dedupe_merge]
    retry_plan: RetryPlan | None
    next_step: str
    status: str
    draft_sections: dict[str, str]
    references: list[str]
    supervisor_phase: str
    completed_workers: Annotated[list[str], _dedupe_merge]
    bias_retry_count: int


@dataclass(slots=True)
class Supervisor:
    runtime: AgentRuntime
    logger: logging.Logger = field(default_factory=lambda: get_logger("supervisor"))
    runtime_factory: Any | None = None
    market_agent: MarketAnalysisAgent = field(init=False)
    company_agent: CompanyAnalysisAgent = field(init=False)
    comparison_agent: ComparisonAndSwotAgent = field(init=False)
    bias_audit_agent: BiasAuditAgent = field(init=False)
    writer_agent: WriterAgent = field(init=False)
    parallel_market_agent: MarketAnalysisAgent = field(init=False)
    parallel_company_agents: dict[CompanyName, CompanyAnalysisAgent] = field(init=False)
    _graph: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.market_agent = MarketAnalysisAgent(self.runtime)
        self.company_agent = CompanyAnalysisAgent(self.runtime)
        self.comparison_agent = ComparisonAndSwotAgent(self.runtime)
        self.bias_audit_agent = BiasAuditAgent(self.runtime)
        self.writer_agent = WriterAgent(self.runtime)

        if self.runtime_factory is not None:
            self.parallel_market_agent = MarketAnalysisAgent(self.runtime_factory())
            self.parallel_company_agents = {
                "LGES": CompanyAnalysisAgent(self.runtime_factory()),
                "CATL": CompanyAnalysisAgent(self.runtime_factory()),
            }
        else:
            self.parallel_market_agent = self.market_agent
            self.parallel_company_agents = {
                "LGES": self.company_agent,
                "CATL": self.company_agent,
            }

        self._graph = self._build_graph()

    def run(self, global_state: GlobalState) -> GlobalState:
        self.logger.info("Supervisor 실행 시작.")
        working_state = dict(global_state)
        working_state["status"] = "running"
        working_state["next_step"] = "supervisor"
        working_state["supervisor_phase"] = "start"
        working_state["completed_workers"] = []
        working_state["bias_retry_count"] = 0
        result = self._graph.invoke(cast(SupervisorGraphState, working_state))
        self.logger.info("Supervisor 실행 완료.")
        return cast(GlobalState, result)

    @property
    def compiled_graph(self):
        return self._graph

    def _build_graph(self):
        graph = StateGraph(SupervisorGraphState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("market", self._run_market)
        graph.add_node("company_lges", self._run_company_lges)
        graph.add_node("company_catl", self._run_company_catl)
        graph.add_node("comparison", self._run_comparison)
        graph.add_node("bias_audit", self._run_bias_audit)
        graph.add_node("writer", self._run_writer)

        graph.add_edge(START, "supervisor")
        graph.add_edge("market", "supervisor")
        graph.add_edge("company_lges", "supervisor")
        graph.add_edge("company_catl", "supervisor")
        graph.add_edge("comparison", "supervisor")
        graph.add_edge("bias_audit", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_conditional_edges("supervisor", self._route_from_supervisor)
        return graph.compile()

    def _supervisor_node(self, state: SupervisorGraphState) -> dict[str, Any]:
        phase = state.get("supervisor_phase", "start")
        completed = set(state.get("completed_workers", []))
        self.logger.info("Supervisor 노드 진입: phase=%s completed=%s", phase, sorted(completed))

        if phase == "start":
            return {
                "supervisor_phase": "awaiting_initial_workers",
                "next_step": "dispatch_initial",
            }

        if phase == "awaiting_initial_workers":
            required = {"market", "company_lges", "company_catl"}
            if required.issubset(completed):
                return {
                    "supervisor_phase": "awaiting_comparison",
                    "next_step": "run_comparison",
                }
            return {}

        if phase == "awaiting_comparison":
            if "comparison" in completed:
                return {
                    "supervisor_phase": "awaiting_bias_audit",
                    "next_step": "run_bias_audit",
                }
            return {}

        if phase == "awaiting_bias_audit":
            retry_plan = state.get("retry_plan")
            if retry_plan is None:
                return {
                    "supervisor_phase": "awaiting_writer",
                    "next_step": "run_writer",
                }
            if retry_plan["target_scope"] == "market":
                return {
                    "supervisor_phase": "awaiting_retry_worker",
                    "next_step": "retry_market",
                }
            if retry_plan["target_scope"] == "company" and retry_plan["target_company"] == "LGES":
                return {
                    "supervisor_phase": "awaiting_retry_worker",
                    "next_step": "retry_company_lges",
                }
            if retry_plan["target_scope"] == "company" and retry_plan["target_company"] == "CATL":
                return {
                    "supervisor_phase": "awaiting_retry_worker",
                    "next_step": "retry_company_catl",
                }
            return {
                "supervisor_phase": "awaiting_comparison",
                "next_step": "retry_comparison",
            }

        if phase == "awaiting_retry_worker":
            return {
                "supervisor_phase": "awaiting_comparison",
                "next_step": "run_comparison",
            }

        if phase == "awaiting_writer":
            if "writer" in completed:
                return {
                    "supervisor_phase": "done",
                    "next_step": "finish",
                    "status": "completed",
                }
            return {}

        return {
            "supervisor_phase": "done",
            "next_step": "finish",
            "status": state.get("status", "completed"),
        }

    def _route_from_supervisor(self, state: SupervisorGraphState):
        action = state.get("next_step")
        if action == "dispatch_initial":
            return ["market", "company_lges", "company_catl"]
        if action == "run_comparison" or action == "retry_comparison":
            return "comparison"
        if action == "run_bias_audit":
            return "bias_audit"
        if action == "retry_market":
            return "market"
        if action == "retry_company_lges":
            return "company_lges"
        if action == "retry_company_catl":
            return "company_catl"
        if action == "run_writer":
            return "writer"
        if action == "finish":
            return END
        return END

    def _run_market(self, global_state: SupervisorGraphState) -> dict[str, Any]:
        is_retry = global_state.get("next_step") == "retry_market"
        self.logger.info("시장 분석 단계 진입. retry=%s", is_retry)
        if is_retry and global_state.get("retry_plan"):
            retry_plan = cast(RetryPlan, global_state["retry_plan"])
            market_state = self.parallel_market_agent.run(
                global_state["goal"],
                global_state["comparison_axes"],
                target_axis=retry_plan["target_axis"],
                query_hint=retry_plan["query_hint"],
            )
        else:
            market_state = self.parallel_market_agent.run(
                global_state["goal"],
                global_state["comparison_axes"],
            )
        working_state = cast(GlobalState, dict(global_state))
        merge_market_into_global(working_state, market_state)
        self.logger.info(
            "시장 분석 완료. 증거 건수=%s",
            len(working_state.get("market_context", {}).get("normalized_evidence", [])),
        )
        return {
            "market_context": working_state["market_context"],
            "unresolved_conflicts": working_state["unresolved_conflicts"],
            "completed_workers": ["market"],
        }

    def _run_company_lges(self, global_state: SupervisorGraphState) -> dict[str, Any]:
        return self._run_company(global_state, "LGES")

    def _run_company_catl(self, global_state: SupervisorGraphState) -> dict[str, Any]:
        return self._run_company(global_state, "CATL")

    def _run_company(self, global_state: SupervisorGraphState, company: CompanyName) -> dict[str, Any]:
        next_step = global_state.get("next_step", "")
        is_retry = next_step in {"retry_company_lges", "retry_company_catl"}
        self.logger.info("기업 분석 시작: %s retry=%s", company, is_retry)

        kwargs: dict[str, Any] = {}
        if is_retry and global_state.get("retry_plan"):
            retry_plan = cast(RetryPlan, global_state["retry_plan"])
            kwargs = {
                "target_axis": retry_plan["target_axis"],
                "query_hint": retry_plan["query_hint"],
            }

        company_state = self.parallel_company_agents[company].run(
            company,
            global_state["goal"],
            global_state["comparison_axes"],
            **kwargs,
        )
        working_state = cast(GlobalState, dict(global_state))
        merge_company_into_global(working_state, company_state)
        worker_name = "company_lges" if company == "LGES" else "company_catl"
        self.logger.info(
            "기업 분석 완료: %s | 증거 건수=%s",
            company,
            len(working_state.get("company_results", {}).get(company, {}).get("normalized_evidence", [])),
        )
        return {
            "company_results": {company: working_state["company_results"][company]},
            "unresolved_conflicts": working_state["unresolved_conflicts"],
            "completed_workers": [worker_name],
        }

    def _run_comparison(self, global_state: SupervisorGraphState) -> dict[str, Any]:
        self.logger.info("비교 분석 단계 진입.")
        working_state = cast(GlobalState, dict(global_state))
        comparison_state = self.comparison_agent.run(working_state)
        merge_comparison_into_global(working_state, comparison_state)
        self.logger.info("비교 분석 완료.")
        return {
            "comparison_matrix": working_state["comparison_matrix"],
            "swot": working_state["swot"],
            "completed_workers": ["comparison"],
        }

    def _run_bias_audit(self, global_state: SupervisorGraphState) -> dict[str, Any]:
        self.logger.info("편향 점검 단계 진입.")
        working_state = cast(GlobalState, dict(global_state))
        audit_result = self.bias_audit_agent.run(working_state)
        current_retry_count = global_state.get("bias_retry_count", 0)
        working_state["bias_flags"] = audit_result.bias_flags
        retry_plan = self._sanitize_retry_plan(audit_result.retry_recommendation)
        max_retries = self.runtime.config.execution.max_bias_retries
        if retry_plan and current_retry_count >= max_retries:
            self.logger.warning(
                "편향 재시도 한도 도달: retries=%s max=%s. 추가 재시도 없이 writer로 진행합니다.",
                current_retry_count,
                max_retries,
            )
            retry_plan = None
        working_state["retry_plan"] = retry_plan
        if retry_plan:
            self.logger.info(
                "편향 재시도 계획: %s | retry %s/%s",
                retry_plan,
                current_retry_count + 1,
                max_retries,
            )
        else:
            self.logger.info("편향 재시도 없음.")
        return {
            "bias_flags": working_state["bias_flags"],
            "retry_plan": working_state["retry_plan"],
            "completed_workers": ["bias_audit"],
            "bias_retry_count": current_retry_count + (1 if retry_plan else 0),
        }

    def _sanitize_retry_plan(self, retry_plan: RetryPlan | None) -> RetryPlan | None:
        if retry_plan is None:
            return None

        sanitized = dict(retry_plan)
        raw_axis = sanitized.get("target_axis")
        if raw_axis not in AXIS_VALUES:
            if raw_axis not in (None, ""):
                self.logger.warning(
                    "유효하지 않은 retry axis 감지: %s. 일반 보강 검색으로 대체합니다.",
                    raw_axis,
                )
            sanitized["target_axis"] = None

        target_scope = sanitized.get("target_scope")
        target_company = sanitized.get("target_company")
        if target_scope == "company" and target_company not in {"LGES", "CATL"}:
            self.logger.warning(
                "유효하지 않은 retry company 감지: %s. 재시도 계획을 무효화합니다.",
                target_company,
            )
            return None

        return cast(RetryPlan, sanitized)

    def _run_writer(self, global_state: SupervisorGraphState) -> dict[str, Any]:
        self.logger.info("참고 근거 집계 단계 진입.")
        working_state = cast(GlobalState, dict(global_state))
        working_state["references"] = collect_references(working_state)
        working_state["next_step"] = "writer"
        self.logger.info("참고 근거 건수: %s", len(working_state.get("references", [])))
        self.logger.info("보고서 작성 단계 진입.")
        working_state = self.writer_agent.run(working_state)
        return {
            "draft_sections": working_state["draft_sections"],
            "references": working_state["references"],
            "completed_workers": ["writer"],
        }
