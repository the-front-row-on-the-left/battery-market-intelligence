from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from battery_strategy.utils.types import GlobalState

if TYPE_CHECKING:
    from battery_strategy.agents.supervisor import Supervisor


def build_supervisor_graph(supervisor: "Supervisor") -> Any:
    companies = list(supervisor.runtime.config.workflow.companies)
    builder = StateGraph(GlobalState)
    builder.add_node("dispatch_research", supervisor._dispatch_research_node)
    builder.add_node("market", supervisor._market_node)
    builder.add_node("company_worker", supervisor._company_parallel_node)
    builder.add_node("comparison", supervisor._comparison_node)
    builder.add_node("bias_audit", supervisor._bias_audit_node)
    builder.add_node("retry_market", supervisor._retry_market_node)
    builder.add_node("retry_company_dispatch", supervisor._dispatch_retry_company_node)
    builder.add_node("retry_comparison", supervisor._retry_comparison_node)
    builder.add_node("collect_references", supervisor._collect_references_node)
    builder.add_node("writer", supervisor._writer_node)

    builder.add_edge(START, "dispatch_research")
    builder.add_conditional_edges("dispatch_research", supervisor._dispatch_research_sends)
    if companies:
        builder.add_edge("market", "comparison")
        builder.add_edge("company_worker", "comparison")
    else:
        builder.add_edge("market", "comparison")

    builder.add_edge("retry_market", "comparison")
    builder.add_conditional_edges(
        "retry_company_dispatch",
        supervisor._dispatch_retry_company_send,
    )
    builder.add_edge("retry_comparison", "comparison")
    builder.add_edge("comparison", "bias_audit")

    route_map = {
        "retry_market": "retry_market",
        "retry_company_dispatch": "retry_company_dispatch",
        "retry_comparison": "retry_comparison",
        "collect_references": "collect_references",
    }
    builder.add_conditional_edges(
        "bias_audit",
        supervisor._route_after_bias_audit,
        route_map,
    )

    builder.add_edge("collect_references", "writer")
    builder.add_edge("writer", END)
    return builder.compile()
