from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Iterable

from battery_strategy.utils.types import (
    AXIS_LABELS,
    AXIS_KEYWORDS,
    COMPARISON_AXES,
    Axis,
    CompanyResult,
    ComparisonRow,
    EvidenceItem,
    MarketContext,
    RetrievedChunk,
    SearchHit,
)
from battery_strategy.utils.common import domain_from_url, utc_today



def infer_axis(text: str) -> Axis:
    lowered = text.lower()
    for axis, keywords in AXIS_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return axis
    return "portfolio"


def _looks_non_korean(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    has_hangul = re.search(r"[가-힣]", cleaned) is not None
    has_latin_or_cjk = re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned) is not None
    return has_latin_or_cjk and not has_hangul



def to_evidence_from_rag(company: str, hit: RetrievedChunk) -> EvidenceItem:
    raw_claim = hit["text"][:300].replace("\n", " ")
    claim = raw_claim if not _looks_non_korean(raw_claim) else "원문 근거 요약은 최종 보고서에서 한국어로 정리함"
    return {
        "company": company,  # type: ignore[typeddict-item]
        "axis": infer_axis(hit["text"]),
        "io_type": "external" if company == "MARKET" else "internal",
        "stance": hit["query_label"],
        "claim": claim,
        "metric_name": "",
        "value": None,
        "unit": "",
        "currency": "",
        "basis": "none",
        "fiscal_year": None,
        "source_type": hit["source_type"],
        "source_title": hit["source_title"],
        "source_page": hit["page_range"],
        "citation": hit["page_range"],
        "date": "",
        "confidence": 0.5,
    }



def to_evidence_from_web(company: str, hit: SearchHit) -> EvidenceItem:
    raw_claim = (hit.get("content") or hit.get("snippet") or hit["title"])[:300].replace("\n", " ")
    claim = raw_claim if not _looks_non_korean(raw_claim) else "원문 근거 요약은 최종 보고서에서 한국어로 정리함"
    return {
        "company": company,  # type: ignore[typeddict-item]
        "axis": infer_axis(hit.get("content") or hit.get("snippet") or hit["title"]),
        "io_type": "external",
        "stance": hit["query_label"],
        "claim": claim,
        "metric_name": "",
        "value": None,
        "unit": "",
        "currency": "",
        "basis": "none",
        "fiscal_year": None,
        "source_type": hit["source_type"],
        "source_title": hit["title"],
        "source_page": hit["url"],
        "citation": hit["url"],
        "date": hit.get("published_at") or hit["searched_at"],
        "confidence": 0.4,
    }



def build_evidence_bank(hits: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        text = hit.get("text") or hit.get("content") or hit.get("snippet") or hit.get("title", "")
        bucket[infer_axis(text)].append(hit)
    return dict(bucket)



def fallback_market_context(rag_hits: list[RetrievedChunk], web_hits: list[SearchHit]) -> MarketContext:
    evidence = [to_evidence_from_rag("MARKET", hit) for hit in rag_hits[:4]]
    evidence.extend(to_evidence_from_web("MARKET", hit) for hit in web_hits[:4])
    bullets = [item["claim"] for item in evidence[:5]] or ["시장 배경 근거가 부족해 요약을 단순화했습니다."]
    return {
        "summary": "EV 수요 구조 변화, ESS 확대, 정책·무역 환경 변화가 배터리 기업의 포트폴리오 다각화를 자극하고 있다.",
        "bullet_points": bullets,
        "normalized_evidence": evidence,
    }



def empty_axis_profile() -> dict[str, Any]:
    return {
        "summary": "",
        "key_evidence": [],
        "strengths": [],
        "weaknesses": [],
        "metrics": [],
    }



def fallback_company_result(company: str, rag_hits: list[RetrievedChunk], web_hits: list[SearchHit]) -> CompanyResult:
    evidence = [to_evidence_from_rag(company, hit) for hit in rag_hits[:6]]
    evidence.extend(to_evidence_from_web(company, hit) for hit in web_hits[:4])
    profile = {axis: empty_axis_profile() for axis in COMPARISON_AXES}
    for item in evidence:
        axis_profile = profile[item["axis"]]
        if not axis_profile["summary"]:
            axis_profile["summary"] = item["claim"]
        axis_profile["key_evidence"].append(item)
        if item["value"] is not None:
            axis_profile["metrics"].append(item)
    return {
        "profile": profile,  # type: ignore[typeddict-item]
        "normalized_evidence": evidence,
        "swot_inputs": {"Strength": [], "Weakness": [], "Opportunity": [], "Threat": []},
        "balance_flags": [],
        "unresolved_conflicts": [],
    }



def fallback_comparison(
    company_results: dict[str, CompanyResult],
) -> tuple[list[ComparisonRow], list[str], dict[str, dict[str, list[str]]]]:
    rows: list[ComparisonRow] = []
    for axis in COMPARISON_AXES:
        lges_profile = company_results["LGES"]["profile"].get(axis) if "LGES" in company_results else None
        catl_profile = company_results["CATL"]["profile"].get(axis) if "CATL" in company_results else None
        axis_label = AXIS_LABELS[axis]
        rows.append(
            {
                "axis": axis,
                "lges_summary": (lges_profile or {}).get("summary", ""),
                "catl_summary": (catl_profile or {}).get("summary", ""),
                "lges_metrics": (lges_profile or {}).get("metrics", []),
                "catl_metrics": (catl_profile or {}).get("metrics", []),
                "difference": f"{axis_label} 관련 근거가 제한적이어서 공개 자료 기준의 요약 비교를 사용합니다.",
                "implication": f"{axis_label} 축은 추가 근거 확보 전까지 보수적으로 해석할 필요가 있습니다.",
                "confidence": 0.3,
            }
        )
    swot = {
        "LGES": {"Strength": [], "Weakness": [], "Opportunity": [], "Threat": []},
        "CATL": {"Strength": [], "Weakness": [], "Opportunity": [], "Threat": []},
    }
    return rows, ["비교 결과를 fallback 방식으로 생성했습니다."], swot



def merge_market_into_global(global_state: dict[str, Any], market_state: dict[str, Any]) -> None:
    global_state["market_context"] = market_state["market_context"]
    global_state["unresolved_conflicts"] = list(
        dict.fromkeys(global_state.get("unresolved_conflicts", []) + market_state.get("unresolved_conflicts", []))
    )



def merge_company_into_global(global_state: dict[str, Any], company_state: dict[str, Any]) -> None:
    company_name = company_state["company_name"]
    global_state.setdefault("company_results", {})[company_name] = {
        "profile": company_state["profile"],
        "normalized_evidence": company_state["normalized_evidence"],
        "swot_inputs": company_state["swot_inputs"],
        "balance_flags": company_state["balance_flags"],
        "unresolved_conflicts": company_state["unresolved_conflicts"],
    }
    global_state["unresolved_conflicts"] = list(
        dict.fromkeys(global_state.get("unresolved_conflicts", []) + company_state.get("unresolved_conflicts", []))
    )



def merge_comparison_into_global(global_state: dict[str, Any], comparison_state: dict[str, Any]) -> None:
    global_state["comparison_matrix"] = comparison_state["comparison_matrix"]
    global_state["swot"] = comparison_state["swot"]



def collect_references(global_state: dict[str, Any]) -> list[str]:
    references: list[str] = []
    for item in global_state.get("corpus_manifest", []):
        reference = item.get("reference")
        if reference:
            references.append(reference)

    seen_urls: set[str] = set()
    company_results = global_state.get("company_results", {})
    for company_result in company_results.values():
        for evidence in company_result.get("normalized_evidence", []):
            citation = evidence.get("citation", "")
            if citation.startswith("http") and citation not in seen_urls:
                seen_urls.add(citation)
                domain = domain_from_url(citation)
                date = evidence.get("date") or utc_today()
                title = evidence.get("source_title") or citation
                references.append(f"{domain}({date}). {title}. {domain}, {citation}")

    return list(dict.fromkeys(references))
