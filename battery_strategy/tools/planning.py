from __future__ import annotations

from collections.abc import Iterable

from battery_strategy.utils.types import Axis, CompanyName

GENERAL_COMPANY_HINTS: dict[CompanyName, list[str]] = {
    "LGES": [
        "ESS",
        "BMTS",
        "BaaS",
        "EaaS",
        "LFP",
        "46 series",
        "North America",
        "Non-EV",
    ],
    "CATL": [
        "ESS",
        "data center storage",
        "swapping",
        "recycling",
        "sodium-ion",
        "new applications",
        "overseas expansion",
    ],
}


AXIS_HINTS: dict[Axis, str] = {
    "portfolio": "application diversification, ESS, BBU, robot, ship, aircraft, data center",
    "commercialization": "orders, revenue mix, project deployment, sales, mass production",
    "manufacturing": "capacity, utilization, plant expansion, yield, cost structure",
    "technology": "LFP, NCM, sodium-ion, energy density, cycle life, safety, roadmap",
    "ecosystem": "recycling, swapping, BMS, BMTS, software, VPP, partnerships",
    "risk": "policy, tariff, customer dependence, fixed cost, utilization, geopolitics",
    "strategy_horizon": "near-term profitability, mid-term expansion, long-term roadmap 2025 2026 2030",
}


def build_market_queries(
    goal: str, target_axis: Axis | None = None, query_hint: str = ""
) -> list[str]:
    extra = f" {query_hint}" if query_hint else ""
    axis_hint = (
        f" {AXIS_HINTS[target_axis]}"
        if target_axis
        else " EV battery market demand, prices, trade, policy"
    )
    return [
        f"neutral|global EV battery market slowdown 2025 2026{axis_hint}{extra}",
        f"positive|ESS battery demand AI data center storage growth 2025 2026{extra}",
        f"negative|battery overcapacity tariffs utilization slowdown policy risk 2025 2026{extra}",
    ]


def build_company_queries(
    company: CompanyName,
    goal: str,
    *,
    target_axis: Axis | None = None,
    query_hint: str = "",
) -> list[str]:
    entity_name = "LG Energy Solution" if company == "LGES" else "CATL"
    general = ", ".join(GENERAL_COMPANY_HINTS[company])
    axis_hint = AXIS_HINTS[target_axis] if target_axis else general
    extra = f" {query_hint}" if query_hint else ""
    return [
        f"neutral|{entity_name} portfolio diversification strategy 2025 2026 {axis_hint}{extra}",
        f"positive|{entity_name} ESS orders technology expansion partnership 2025 2026 {axis_hint}{extra}",
        f"negative|{entity_name} utilization fixed cost policy risk slowdown tariff 2025 2026 {axis_hint}{extra}",
    ]


def append_retry_hint(queries: Iterable[str], query_hint: str) -> list[str]:
    if not query_hint:
        return list(queries)
    enriched: list[str] = []
    for item in queries:
        label, query = item.split("|", maxsplit=1)
        enriched.append(f"{label}|{query} {query_hint}")
    return enriched
