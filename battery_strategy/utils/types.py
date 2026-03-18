from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

Axis = Literal[
    "portfolio",
    "commercialization",
    "manufacturing",
    "technology",
    "ecosystem",
    "risk",
    "strategy_horizon",
]
CompanyName = Literal["LGES", "CATL"]
EntityName = Literal["LGES", "CATL", "MARKET"]
SearchLabel = Literal["neutral", "positive", "negative"]
Scope = Literal["market", "company", "comparison"]
RetryFrom = Literal["query", "retrieval", "normalize", "profile", "compare", "swot"]
SourceType = Literal["official_report", "official_pr", "industry_report", "academic", "news"]
BasisType = Literal["consolidated", "standalone", "segment", "project", "region", "none"]

COMPARISON_AXES: list[Axis] = [
    "portfolio",
    "commercialization",
    "manufacturing",
    "technology",
    "ecosystem",
    "risk",
    "strategy_horizon",
]

AXIS_VALUES: tuple[Axis, ...] = tuple(COMPARISON_AXES)

AXIS_LABELS: dict[Axis, str] = {
    "portfolio": "포트폴리오 다각화",
    "commercialization": "상용화 및 매출화",
    "manufacturing": "제조 및 생산능력",
    "technology": "기술 개발",
    "ecosystem": "생태계 및 파트너십",
    "risk": "리스크",
    "strategy_horizon": "전략 시계열",
}

AXIS_KEYWORDS: dict[Axis, tuple[str, ...]] = {
    "portfolio": ("portfolio", "diversification", "application", "ess", "bbu", "robot", "uav", "uav", "ship", "aircraft", "new application", "신사업", "다각화", "포트폴리오", "储能", "新兴应用"),
    "commercialization": ("orders", "deployment", "revenue", "commercialization", "profit", "sales", "수주", "매출", "양산", "상용화", "收入", "销量", "出货"),
    "manufacturing": ("capacity", "plant", "utilization", "cost", "manufacturing", "yield", "생산", "공장", "가동률", "원가", "产能", "产量", "良率", "成本"),
    "technology": ("energy density", "cycle life", "safety", "roadmap", "lfp", "ncm", "sodium", "technology", "기술", "에너지밀도", "수명", "안전", "钠离子", "技术", "能量密度"),
    "ecosystem": ("recycling", "swapping", "bms", "bmts", "platform", "partnership", "vpp", "재활용", "교환", "생태계", "回收", "换电", "合作"),
    "risk": ("tariff", "policy", "geopolitics", "dependency", "fixed cost", "risk", "관세", "정책", "지정학", "위험", "固定成本", "风险", "政策"),
    "strategy_horizon": ("2025", "2026", "2030", "long-term", "near-term", "mid-term", "short-term", "중장기", "단기", "장기", "长期", "短期"),
}


class SourceManifestItem(TypedDict):
    id: str
    group: EntityName
    title: str
    local_path: str
    language: str
    source_type: SourceType
    reference: str
    url: str
    note: NotRequired[str]


class ChunkRecord(TypedDict):
    chunk_id: str
    source_id: str
    source_group: EntityName
    source_title: str
    source_type: SourceType
    source_url: str
    reference: str
    page_start: int
    page_end: int
    display_page_start: str
    display_page_end: str
    text: str
    language: str


class SearchHit(TypedDict):
    query: str
    query_label: SearchLabel
    query_axis: NotRequired[str]
    title: str
    url: str
    snippet: str
    content: str
    source_type: SourceType
    domain: str
    published_at: NotRequired[str]
    searched_at: str


class RetrievedChunk(TypedDict):
    query: str
    query_label: SearchLabel
    chunk_id: str
    source_id: str
    source_group: EntityName
    source_title: str
    source_type: SourceType
    source_url: str
    reference: str
    page_range: str
    text: str
    score: float
    retrieval_mode: Literal["dense", "sparse", "hybrid"]


class EvidenceItem(TypedDict):
    company: EntityName
    axis: Axis
    io_type: Literal["internal", "external"]
    stance: SearchLabel
    claim: str
    metric_name: str
    value: Any
    unit: str
    currency: str
    basis: BasisType
    fiscal_year: int | None
    source_type: SourceType
    source_title: str
    source_page: str
    citation: str
    date: str
    confidence: float


class AxisProfile(TypedDict):
    summary: str
    key_evidence: list[EvidenceItem]
    strengths: list[str]
    weaknesses: list[str]
    metrics: list[EvidenceItem]


class CompanyResult(TypedDict):
    profile: dict[Axis, AxisProfile]
    normalized_evidence: list[EvidenceItem]
    swot_inputs: dict[str, list[EvidenceItem]]
    balance_flags: list[str]
    unresolved_conflicts: list[str]


class MarketContext(TypedDict):
    summary: str
    bullet_points: list[str]
    normalized_evidence: list[EvidenceItem]


class ComparisonRow(TypedDict):
    axis: Axis
    lges_summary: str
    catl_summary: str
    lges_metrics: list[EvidenceItem]
    catl_metrics: list[EvidenceItem]
    difference: str
    implication: str
    confidence: float


class RetryPlan(TypedDict):
    target_scope: Scope
    target_company: CompanyName | None
    target_axis: Axis | None
    retry_from: RetryFrom
    reason: str
    query_hint: str


class GlobalState(TypedDict):
    goal: str
    criteria: list[str]
    comparison_axes: list[Axis]
    corpus_manifest: list[SourceManifestItem]
    query_plan: dict[str, list[str]]
    market_context: MarketContext
    company_results: dict[CompanyName, CompanyResult]
    comparison_matrix: list[ComparisonRow]
    swot: dict[str, dict[str, list[str]]]
    bias_flags: list[str]
    unresolved_conflicts: list[str]
    retry_plan: RetryPlan | None
    next_step: str
    status: str
    draft_sections: dict[str, str]
    references: list[str]


class MarketState(TypedDict):
    query_set: list[str]
    rag_hits: list[RetrievedChunk]
    web_hits: list[SearchHit]
    evidence_bank: dict[str, list[dict[str, Any]]]
    normalized_evidence: list[EvidenceItem]
    market_context: MarketContext
    balance_flags: list[str]
    unresolved_conflicts: list[str]
    target_axis: Axis | None
    retry_from: str | None


class CompanyState(TypedDict):
    company_name: CompanyName
    comparison_axes: list[Axis]
    query_set: list[str]
    rag_hits: list[RetrievedChunk]
    web_hits: list[SearchHit]
    evidence_bank: dict[str, list[dict[str, Any]]]
    normalized_evidence: list[EvidenceItem]
    profile: dict[Axis, AxisProfile]
    swot_inputs: dict[str, list[EvidenceItem]]
    balance_flags: list[str]
    unresolved_conflicts: list[str]
    target_axis: Axis | None
    retry_from: str | None


class ComparisonState(TypedDict):
    company_results: dict[CompanyName, CompanyResult]
    market_context: MarketContext
    comparison_axes: list[Axis]
    comparison_matrix: list[ComparisonRow]
    insights: list[str]
    swot: dict[str, dict[str, list[str]]]
    validation_flags: list[str]
