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

COMPANY_OFFICIAL_QUERY_HINTS: dict[CompanyName, str] = {
    "LGES": "site:lgensol.com annual report business report investor relations press release",
    "CATL": "site:catl.com annual report investor relations press release sustainability",
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

AXIS_QUERY_TEMPLATES: dict[Axis, tuple[str, str]] = {
    "portfolio": (
        "portfolio diversification ESS LFP non-EV applications product mix",
        "official report investor presentation product application diversification",
    ),
    "commercialization": (
        "orders sales shipments revenue mix project deployment mass production",
        "official report investor relations orders shipments revenue project delivery",
    ),
    "manufacturing": (
        "capacity utilization plant expansion yield cost structure production footprint",
        "official report capacity utilization plant expansion manufacturing footprint",
    ),
    "technology": (
        "LFP sodium-ion energy density cycle life safety roadmap next generation battery",
        "official report technology roadmap LFP sodium-ion next generation battery",
    ),
    "ecosystem": (
        "recycling swapping partnerships supply chain BMS software ecosystem",
        "official report partnerships recycling supply chain ecosystem",
    ),
    "risk": (
        "risk factors supply chain raw material tariff geopolitics utilization customer concentration",
        "official report risk factors supply chain raw material tariff geopolitical uncertainty",
    ),
    "strategy_horizon": (
        "2025 2026 2030 roadmap medium term long term expansion profitability",
        "official report 2025 2026 2030 roadmap medium term long term strategy",
    ),
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
        f"neutral|site:iea.org battery market outlook 2025 2026 demand prices policy risk{axis_hint}{extra}",
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
    template_hint, official_hint = (
        AXIS_QUERY_TEMPLATES[target_axis]
        if target_axis
        else (general, "official report investor relations press release")
    )
    extra = f" {query_hint}" if query_hint else ""
    return [
        f"neutral|{entity_name} 2024 2025 2026 {template_hint} {axis_hint}{extra}",
        f"positive|{entity_name} 2024 2025 2026 ESS orders technology expansion partnership commercialization {axis_hint}{extra}",
        f"negative|{entity_name} 2024 2025 2026 risk factors utilization fixed cost policy tariff supply chain {axis_hint}{extra}",
        f"neutral|{COMPANY_OFFICIAL_QUERY_HINTS[company]} {entity_name} 2024 2025 2026 {official_hint} {axis_hint}{extra}",
    ]


def append_retry_hint(queries: Iterable[str], query_hint: str) -> list[str]:
    if not query_hint:
        return list(queries)
    enriched: list[str] = []
    for item in queries:
        label, query = item.split("|", maxsplit=1)
        enriched.append(f"{label}|{query} {query_hint}")
    return enriched
