from __future__ import annotations

from collections.abc import Iterable

from battery_strategy.utils.types import AXIS_VALUES, Axis, CompanyName

BASE_MARKET_REQUIREMENTS = (
    "2024 2025 2026 official industry report regulator primary source external validation "
    "EV slowdown diversification ESS battery demand supply chain policy tariff risk"
)

BASE_COMPANY_REQUIREMENTS = (
    "2024 2025 2026 official filing official report primary source external validation "
    "portfolio diversification non-EV ESS supply chain policy risk commercialization"
)

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

CATL_CHINESE_HINTS = (
    "宁德时代 储能 新应用 产能 利用率 在建产能 钠离子 固态电池 回收 换电 "
    "原材料 成本 关税 政策风险 2024 2025 2026"
)


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
    resolved_axis = target_axis if target_axis in AXIS_VALUES else None
    extra = f" {query_hint}" if query_hint else ""
    axis_hint = (
        f" {AXIS_HINTS[resolved_axis]}"
        if resolved_axis
        else " EV battery market demand prices trade policy overcapacity utilization recycling"
    )
    return [
        (
            f"neutral|{goal} global EV battery market slowdown diversification "
            f"{BASE_MARKET_REQUIREMENTS}{axis_hint}{extra}"
        ),
        (
            f"positive|{goal} ESS battery demand AI data center storage growth "
            f"{BASE_MARKET_REQUIREMENTS}{axis_hint}{extra}"
        ),
        (
            f"negative|{goal} battery overcapacity tariffs utilization slowdown policy risk "
            f"{BASE_MARKET_REQUIREMENTS}{axis_hint}{extra}"
        ),
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
    resolved_axis = target_axis if target_axis in AXIS_VALUES else None
    axis_hint = AXIS_HINTS[resolved_axis] if resolved_axis else general
    extra = f" {query_hint}" if query_hint else ""
    return [
        *[
            (
                f"neutral|{goal} {entity_name} portfolio diversification strategy "
                f"{BASE_COMPANY_REQUIREMENTS} {axis_hint}{extra}"
            ),
            (
                f"positive|{goal} {entity_name} ESS orders technology expansion partnership "
                f"{BASE_COMPANY_REQUIREMENTS} {axis_hint}{extra}"
            ),
            (
                f"negative|{goal} {entity_name} utilization fixed cost policy risk slowdown tariff "
                f"{BASE_COMPANY_REQUIREMENTS} {axis_hint}{extra}"
            ),
        ],
        *(
            [
                f"neutral|{goal} 宁德时代 多元化战略 储能 新应用 {CATL_CHINESE_HINTS} {axis_hint}{extra}",
                f"negative|{goal} 宁德时代 风险 原材料 成本 关税 政策风险 {CATL_CHINESE_HINTS} {axis_hint}{extra}",
            ]
            if company == "CATL"
            else []
        ),
    ]


def append_retry_hint(queries: Iterable[str], query_hint: str) -> list[str]:
    if not query_hint:
        return list(queries)
    enriched: list[str] = []
    for item in queries:
        label, query = item.split("|", maxsplit=1)
        enriched.append(f"{label}|{query} {query_hint}")
    return enriched
