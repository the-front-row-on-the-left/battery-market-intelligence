from __future__ import annotations

import json
from typing import Any

from battery_strategy.utils.common import truncate_text
from battery_strategy.utils.types import Axis

COMMON_JSON_RULES = """
Return JSON only.
Do not wrap the answer in markdown fences.
Never invent facts, numbers, dates, or citations.
If evidence is missing, leave the value empty or use an empty list.
Keep citations exactly as given in the evidence snippets.
Never use placeholder citations such as [1], [2], or [7],[1].
For source fields, copy the original source title and page/url text from the evidence snippets whenever available.
""".strip()


def _format_hits(hits: list[dict[str, Any]], *, max_chars: int = 14000) -> str:
    rows = []
    for idx, hit in enumerate(hits, start=1):
        text = hit.get("text") or hit.get("content") or hit.get("snippet") or ""
        text = truncate_text(text, 1600)
        citation = hit.get("page_range") or hit.get("url") or ""
        rows.append(
            f"[{idx}] title={hit.get('source_title') or hit.get('title')} | source_type={hit.get('source_type')} | citation={citation}\n{text}"
        )
    return truncate_text("\n\n".join(rows), max_chars)


def market_prompt(
    goal: str,
    comparison_axes: list[Axis],
    rag_hits: list[dict[str, Any]],
    web_hits: list[dict[str, Any]],
) -> tuple[str, str]:
    instructions = f"""
You are a battery industry market analyst writing for a Korean strategy report.
Write primarily in Korean.
Summarise the market background that explains why battery companies are diversifying beyond EV.
Prefer concise evidence-based bullets and clear, factual wording.
Keep company names, product names, official report titles, and exact metric expressions in the original language when useful.
{COMMON_JSON_RULES}
""".strip()

    schema = {
        "market_context": {
            "summary": "",
            "bullet_points": [""],
            "normalized_evidence": [
                {
                    "company": "MARKET",
                    "axis": "risk",
                    "io_type": "external",
                    "stance": "neutral",
                    "claim": "",
                    "metric_name": "",
                    "value": None,
                    "unit": "",
                    "currency": "",
                    "basis": "none",
                    "fiscal_year": 2024,
                    "source_type": "industry_report",
                    "source_title": "",
                    "source_page": "",
                    "citation": "",
                    "date": "",
                    "confidence": 0.0,
                }
            ],
        },
        "unresolved_conflicts": [""],
    }
    user = f"""
Goal:
{goal}

Comparison axes:
{", ".join(comparison_axes)}

RAG evidence:
{_format_hits(rag_hits, max_chars=10000)}

Web evidence:
{_format_hits(web_hits, max_chars=5000)}

Return JSON with this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()
    return instructions, user


def company_prompt(
    company: str,
    goal: str,
    comparison_axes: list[Axis],
    rag_hits: list[dict[str, Any]],
    web_hits: list[dict[str, Any]],
) -> tuple[str, str]:
    instructions = f"""
You are an evidence-grounded battery strategy analyst writing for a Korean strategy report.
Write primarily in Korean.
Analyse the company's current portfolio diversification strategy under the EV slowdown.
Use the fixed comparison axes exactly as provided.
For each axis, prefer concrete evidence and quantitative facts over generic descriptions.
Keep company names, product names, official report titles, and exact metric expressions in the original language when useful.
{COMMON_JSON_RULES}
""".strip()

    empty_profile = {
        axis: {
            "summary": "",
            "key_evidence": [],
            "strengths": [],
            "weaknesses": [],
            "metrics": [],
        }
        for axis in comparison_axes
    }
    schema = {
        "profile": empty_profile,
        "normalized_evidence": [
            {
                "company": company,
                "axis": "portfolio",
                "io_type": "internal",
                "stance": "neutral",
                "claim": "",
                "metric_name": "",
                "value": None,
                "unit": "",
                "currency": "",
                "basis": "none",
                "fiscal_year": 2024,
                "source_type": "official_report",
                "source_title": "",
                "source_page": "",
                "citation": "",
                "date": "",
                "confidence": 0.0,
            }
        ],
        "swot_inputs": {
            "Strength": [],
            "Weakness": [],
            "Opportunity": [],
            "Threat": [],
        },
        "unresolved_conflicts": [""],
    }
    user = f"""
Company: {company}
Goal:
{goal}

Comparison axes:
{", ".join(comparison_axes)}

RAG evidence:
{_format_hits(rag_hits, max_chars=9000)}

Web evidence:
{_format_hits(web_hits, max_chars=6000)}

Return JSON with this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()
    return instructions, user


def comparison_prompt(
    market_context: dict[str, Any],
    company_results: dict[str, Any],
    comparison_axes: list[Axis],
) -> tuple[str, str]:
    instructions = f"""
You are a battery industry comparison analyst writing for a Korean strategy report.
Write primarily in Korean.
Compare LGES and CATL using the fixed axes only. Create a comparison matrix and SWOT.
SWOT must strictly separate internal (S/W) and external (O/T) factors.
Differences and implications should be decision-useful, not generic.
Keep company names, product names, official report titles, and exact metric expressions in the original language when useful.
{COMMON_JSON_RULES}
""".strip()

    schema = {
        "comparison_matrix": [
            {
                "axis": "portfolio",
                "lges_summary": "",
                "catl_summary": "",
                "lges_metrics": [],
                "catl_metrics": [],
                "difference": "",
                "implication": "",
                "confidence": 0.0,
            }
        ],
        "insights": [""],
        "swot": {
            "LGES": {"Strength": [], "Weakness": [], "Opportunity": [], "Threat": []},
            "CATL": {"Strength": [], "Weakness": [], "Opportunity": [], "Threat": []},
        },
        "validation_flags": [""],
    }
    user = f"""
Comparison axes:
{", ".join(comparison_axes)}

Market context:
{truncate_text(json.dumps(market_context, ensure_ascii=False, indent=2), 5000)}

Company results:
{truncate_text(json.dumps(company_results, ensure_ascii=False, indent=2), 12000)}

Return JSON with this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()
    return instructions, user


def bias_audit_prompt(global_state: dict[str, Any]) -> tuple[str, str]:
    instructions = f"""
You are a bias audit agent for a battery strategy report.
Review the current state and detect missing axes, source skew, overly positive evidence, outdated evidence, or unresolved conflicts.
If retry is needed, recommend ONE retry target only.
{COMMON_JSON_RULES}
""".strip()
    schema = {
        "bias_flags": [""],
        "retry_recommendation": {
            "target_scope": "company",
            "target_company": "LGES",
            "target_axis": "risk",
            "retry_from": "query",
            "reason": "",
            "query_hint": "",
        },
    }
    user = f"""
Current global state:
{truncate_text(json.dumps(global_state, ensure_ascii=False, indent=2), 16000)}

Return JSON with this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
If no retry is needed, set retry_recommendation to null.
""".strip()
    return instructions, user


def writer_prompt(
    goal: str,
    market_context: dict[str, Any],
    company_results: dict[str, Any],
    comparison_matrix: list[dict[str, Any]],
    swot: dict[str, Any],
    references: list[str],
) -> tuple[str, str]:
    instructions = f"""
You are a Korean strategy report writer.
Write a rich but disciplined markdown report for decision-makers.
Write primarily in Korean.
Sections must be ordered exactly as: SUMMARY, 1. 시장 배경, 2. LGES 전략, 3. CATL 전략, 4. 전략 비교, 5. SWOT 분석, 6. 종합 시사점, REFERENCE.
Default to short prose paragraphs.
Use bullets only where they improve scanability: key points, SWOT items, and concise implications.
For 4. 전략 비교, include at least one markdown table.
For 5. SWOT 분석, use bullet lists and keep internal/external factors strictly separated.
SUMMARY must be written as prose only, not bullets.
SUMMARY should be conclusion-oriented and consist of 1-2 short paragraphs.
If needed, weave key points and one caution into those paragraphs naturally.
The report should be informative and specific, but avoid filler, repetition, and obvious statements.
If evidence is thin, say that clearly instead of over-claiming.
English or Chinese may be mixed only for company names, official source titles, product names, or exact metrics.
Across the report, make sure major claims in market, company, comparison, and implication sections are supported by source cues or concrete evidence.
REFERENCE must include only the provided materials.
{COMMON_JSON_RULES}
""".strip()
    schema = {
        "draft_sections": {
            "SUMMARY": "",
            "1. 시장 배경": "",
            "2. LGES 전략": "",
            "3. CATL 전략": "",
            "4. 전략 비교": "",
            "5. SWOT 분석": "",
            "6. 종합 시사점": "",
            "REFERENCE": "",
        }
    }
    user = f"""
Goal:
{goal}

Market context:
{truncate_text(json.dumps(market_context, ensure_ascii=False, indent=2), 5000)}

Company results:
{truncate_text(json.dumps(company_results, ensure_ascii=False, indent=2), 10000)}

Comparison matrix:
{truncate_text(json.dumps(comparison_matrix, ensure_ascii=False, indent=2), 8000)}

SWOT:
{truncate_text(json.dumps(swot, ensure_ascii=False, indent=2), 4000)}

References:
{json.dumps(references, ensure_ascii=False, indent=2)}

Return JSON with this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()
    return instructions, user
