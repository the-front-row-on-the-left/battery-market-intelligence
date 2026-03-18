from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from battery_strategy.utils.types import (
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


def _clean_claim_text(text: str) -> str:
    cleaned = " ".join((text or "").replace("\u00a0", " ").split())
    if not cleaned:
        return ""
    for separator in (". ", "。", "\n"):
        if separator in cleaned:
            first = cleaned.split(separator, maxsplit=1)[0].strip()
            if len(first) >= 40:
                cleaned = first
                break
    return cleaned[:260].strip()


def _select_web_claim(hit: SearchHit) -> str:
    candidates = [
        hit.get("snippet", ""),
        hit.get("content", ""),
        hit.get("title", ""),
    ]
    for candidate in candidates:
        cleaned = _clean_claim_text(candidate)
        if len(cleaned) >= 30 and "cookie" not in cleaned.lower():
            return cleaned
    return _clean_claim_text(hit.get("title", "")) or "웹 근거 요약을 자동 추출하지 못했습니다."



def infer_axis(text: str) -> Axis:
    lowered = text.lower()
    for axis, keywords in AXIS_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return axis
    return "portfolio"



def to_evidence_from_rag(company: str, hit: RetrievedChunk) -> EvidenceItem:
    claim = _clean_claim_text(hit["text"]) or hit["text"][:180].replace("\n", " ").strip()
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
    claim = _select_web_claim(hit)
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
        rows.append(
            {
                "axis": axis,
                "lges_summary": (lges_profile or {}).get("summary", ""),
                "catl_summary": (catl_profile or {}).get("summary", ""),
                "lges_metrics": (lges_profile or {}).get("metrics", []),
                "catl_metrics": (catl_profile or {}).get("metrics", []),
                "difference": "자동 비교 결과 생성 실패로 요약 비교를 사용합니다.",
                "implication": "세부 비교는 원문 evidence를 추가 확인하세요.",
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
