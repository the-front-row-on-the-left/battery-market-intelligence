from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.tools.prompts import writer_prompt
from battery_strategy.utils.common import dump_json
from battery_strategy.utils.types import GlobalState

SECTION_ORDER = [
    "SUMMARY",
    "1. 시장 배경",
    "2. LGES 전략",
    "3. CATL 전략",
    "4. 전략 비교",
    "5. SWOT 분석",
    "6. 종합 시사점",
    "REFERENCE",
]

CHROME_BINARY = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


class WriterAgent:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def run(self, global_state: GlobalState) -> GlobalState:
        instructions, user_prompt = writer_prompt(
            goal=global_state["goal"],
            market_context=global_state["market_context"],
            company_results=global_state["company_results"],
            comparison_matrix=global_state["comparison_matrix"],
            swot=global_state["swot"],
            references=global_state["references"],
        )
        fallback = {"draft_sections": self._fallback_sections(global_state)}
        parsed = self.runtime.llm.json(instructions, user_prompt, fallback=fallback)
        global_state["draft_sections"] = parsed.get("draft_sections", fallback["draft_sections"])
        self._write_outputs(global_state)
        return global_state

    def _write_outputs(self, global_state: GlobalState) -> None:
        output_dir = Path(self.runtime.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_at = self._report_timestamp()
        final_markdown = self._sections_to_markdown(global_state["draft_sections"], generated_at)
        final_html = self._build_html_report(global_state, generated_at)

        markdown_path = output_dir / "final_report.md"
        html_path = output_dir / "final_report.html"
        latest_pdf_path = output_dir / "final_report.pdf"
        archived_pdf_path = output_dir / f"final_report_{self._report_file_timestamp()}.pdf"

        markdown_path.write_text(final_markdown, encoding="utf-8")
        html_path.write_text(final_html, encoding="utf-8")
        self._write_pdf_from_html(html_path, latest_pdf_path)
        if latest_pdf_path.exists():
            archived_pdf_path.write_bytes(latest_pdf_path.read_bytes())

        (output_dir / "references.txt").write_text(
            "\n".join(global_state.get("references", [])),
            encoding="utf-8",
        )
        dump_json(global_state, output_dir / "final_state.json")

    @classmethod
    def render_pdf_from_html(cls, html_path: Path, pdf_path: Path) -> Path:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        cls._write_pdf_from_html(html_path, pdf_path)
        return pdf_path

    @staticmethod
    def _sections_to_markdown(sections: dict[str, str], generated_at: str) -> str:
        blocks: list[str] = [f"> 생성 시각: {generated_at}"]
        for title in SECTION_ORDER:
            prefix = "#" if title in {"SUMMARY", "REFERENCE"} else "##"
            blocks.append(f"{prefix} {title}\n\n{sections.get(title, '').strip()}")
        return "\n\n".join(blocks).strip() + "\n"

    @staticmethod
    def _fallback_sections(global_state: GlobalState) -> dict[str, str]:
        comparison_lines: list[str] = []
        for row in global_state.get("comparison_matrix", []):
            axis = row.get("axis", "")
            comparison_lines.append(
                f"- **{axis}**: LGES={row.get('lges_summary', '')} / CATL={row.get('catl_summary', '')} / 차이={row.get('difference', '')}"
            )

        references = "\n".join(f"- {item}" for item in global_state.get("references", []))
        return {
            "SUMMARY": (
                "전기차 수요 둔화와 정책 불확실성이 이어지는 환경에서도 LGES와 CATL은 모두 EV 외 수요처 확대를 통해 "
                "포트폴리오 리스크를 낮추고 새로운 성장 동력을 확보하려는 공통된 방향을 보이고 있다. LGES는 기술, "
                "ESS, 생산 유연성과 같은 질적 경쟁력에 상대적으로 무게를 두는 반면, CATL은 생산능력, 제품군 확장, "
                "비용 경쟁력과 같은 양적 우위를 바탕으로 대응하는 차이가 나타난다.\n\n"
                "결과적으로 두 기업 모두 EV 외 응용처와 생태계 확장을 핵심 성장 축으로 삼고 있지만, 향후 성과는 "
                "정책 변화, 수요 회복 속도, ESS 및 재활용 사업의 실질적 수익화 여부에 따라 갈릴 가능성이 높다."
            ),
            "1. 시장 배경": global_state.get("market_context", {}).get("summary", ""),
            "2. LGES 전략": global_state.get("company_results", {})
            .get("LGES", {})
            .get("profile", {})
            .get("portfolio", {})
            .get("summary", ""),
            "3. CATL 전략": global_state.get("company_results", {})
            .get("CATL", {})
            .get("profile", {})
            .get("portfolio", {})
            .get("summary", ""),
            "4. 전략 비교": "\n".join(comparison_lines),
            "5. SWOT 분석": "SWOT은 내부 강점/약점과 외부 기회/위협을 구분해 정리한다.",
            "6. 종합 시사점": (
                "향후 경쟁력은 EV 외 수요처 확대, 공급망 안정성, 재활용 및 ESS 확장 역량에서 갈릴 가능성이 높다."
            ),
            "REFERENCE": references,
        }

    @classmethod
    def _build_html_report(cls, global_state: GlobalState, generated_at: str) -> str:
        sections = global_state["draft_sections"]
        summary_html = cls._markdown_to_html(sections.get("SUMMARY", ""))
        market_html = cls._markdown_to_html(sections.get("1. 시장 배경", ""))
        lges_html = cls._markdown_to_html(sections.get("2. LGES 전략", ""))
        catl_html = cls._markdown_to_html(sections.get("3. CATL 전략", ""))
        comparison_html = cls._markdown_to_html(sections.get("4. 전략 비교", ""))
        implication_html = cls._markdown_to_html(sections.get("6. 종합 시사점", ""))
        reference_html = cls._reference_html(global_state.get("references", []))
        swot_html = cls._swot_html(global_state.get("swot", {}))

        market_refs = cls._evidence_panel(
            "주요 근거",
            global_state.get("market_context", {}).get("normalized_evidence", []),
            limit=4,
        )
        lges_refs = cls._evidence_panel(
            "주요 근거",
            global_state.get("company_results", {}).get("LGES", {}).get("normalized_evidence", []),
            limit=4,
        )
        catl_refs = cls._evidence_panel(
            "주요 근거",
            global_state.get("company_results", {}).get("CATL", {}).get("normalized_evidence", []),
            limit=4,
        )
        comparison_refs = cls._comparison_evidence_panel(global_state)
        implication_refs = cls._evidence_panel(
            "참고 근거",
            [
                *global_state.get("market_context", {}).get("normalized_evidence", [])[:2],
                *global_state.get("company_results", {}).get("LGES", {}).get("normalized_evidence", [])[:2],
                *global_state.get("company_results", {}).get("CATL", {}).get("normalized_evidence", [])[:2],
            ],
            limit=6,
        )

        title = escape(global_state.get("goal", "배터리 시장 전략 비교 보고서"))
        css = cls._report_css()

        return f"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
{css}
    </style>
  </head>
  <body>
    <main class="report-shell">
      <section class="cover">
        <div class="eyebrow">Battery Market Intelligence</div>
        <h1>{title}</h1>
        <p class="subtitle">LGES와 CATL의 포트폴리오 다각화 전략 비교 보고서</p>
        <p class="timestamp">생성 시각: {escape(generated_at)}</p>
      </section>

      <section class="section summary">
        <h2>SUMMARY</h2>
        <div class="section-body prose">{summary_html}</div>
      </section>

      <section class="section">
        <h2>1. 시장 배경</h2>
        <div class="section-body prose">{market_html}</div>
        {market_refs}
      </section>

      <section class="section">
        <h2>2. LGES 전략</h2>
        <div class="section-body prose">{lges_html}</div>
        {lges_refs}
      </section>

      <section class="section">
        <h2>3. CATL 전략</h2>
        <div class="section-body prose">{catl_html}</div>
        {catl_refs}
      </section>

      <section class="section">
        <h2>4. 전략 비교</h2>
        <div class="section-body prose">{comparison_html}</div>
        {comparison_refs}
      </section>

      <section class="section">
        <h2>5. SWOT 분석</h2>
        <div class="section-body">{swot_html}</div>
      </section>

      <section class="section">
        <h2>6. 종합 시사점</h2>
        <div class="section-body prose">{implication_html}</div>
        {implication_refs}
      </section>

      <section class="section references">
        <h2>REFERENCE</h2>
        <div class="section-body prose">{reference_html}</div>
      </section>
    </main>
  </body>
</html>
"""

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        try:
            import markdown

            return markdown.markdown(
                text,
                extensions=["extra", "tables", "sane_lists"],
                output_format="html5",
            )
        except Exception:
            paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
            return "\n".join(f"<p>{escape(part)}</p>" for part in paragraphs)

    @classmethod
    def _evidence_panel(cls, title: str, evidence: list[dict[str, Any]], *, limit: int) -> str:
        rows: list[str] = []
        for item in evidence[:limit]:
            claim = cls._clean_text(str(item.get("claim", "")))
            if not claim:
                continue
            source_label = cls._format_evidence_source(item)
            if not source_label:
                continue
            rows.append(
                "<li><span class=\"evidence-claim\">{claim}</span><span class=\"evidence-cite\">출처: {cite}</span></li>".format(
                    claim=escape(claim[:220]),
                    cite=escape(source_label),
                )
            )
        if not rows:
            return ""
        return f"""
<div class="evidence-panel">
  <div class="evidence-title">{escape(title)}</div>
  <ul class="evidence-list">
    {''.join(rows)}
  </ul>
</div>
"""

    @classmethod
    def _comparison_evidence_panel(cls, global_state: GlobalState) -> str:
        items = [
            *global_state.get("company_results", {}).get("LGES", {}).get("normalized_evidence", [])[:3],
            *global_state.get("company_results", {}).get("CATL", {}).get("normalized_evidence", [])[:3],
        ]
        return cls._evidence_panel("비교 근거", items, limit=6)

    @classmethod
    def _swot_html(cls, swot: dict[str, Any]) -> str:
        companies = ["LGES", "CATL"]
        cards: list[str] = []
        for company in companies:
            company_swot = swot.get(company, {}) if isinstance(swot, dict) else {}
            cards.append(
                f"""
<div class="swot-company">
  <h3>{escape(company)}</h3>
  <div class="swot-grid">
    {cls._swot_card("Strength", company_swot.get("Strength", []), "strength")}
    {cls._swot_card("Weakness", company_swot.get("Weakness", []), "weakness")}
    {cls._swot_card("Opportunity", company_swot.get("Opportunity", []), "opportunity")}
    {cls._swot_card("Threat", company_swot.get("Threat", []), "threat")}
  </div>
</div>
"""
            )
        return "\n".join(cards)

    @classmethod
    def _swot_card(cls, title: str, items: list[Any], tone: str) -> str:
        rows = []
        for item in (items or [])[:5]:
            cleaned = cls._clean_text(str(item))
            if cleaned:
                rows.append(f"<li>{escape(cleaned)}</li>")
        if not rows:
            rows.append("<li>근거가 충분하지 않아 명시적으로 정리되지 않았습니다.</li>")
        return f"""
<section class="swot-card {tone}">
  <div class="swot-title">{escape(title)}</div>
  <ul>{''.join(rows)}</ul>
</section>
"""

    @staticmethod
    def _reference_html(references: list[str]) -> str:
        if not references:
            return "<p>참고문헌 없음</p>"
        rows = "".join(f"<li>{escape(item)}</li>" for item in references)
        return f"<ol class=\"reference-list\">{rows}</ol>"

    @classmethod
    def _format_evidence_source(cls, item: dict[str, Any]) -> str:
        source_title = cls._clean_text(str(item.get("source_title") or ""))
        source_page = cls._clean_text(str(item.get("source_page") or ""))
        citation = cls._clean_text(str(item.get("citation") or ""))
        date = cls._clean_text(str(item.get("date") or ""))

        opaque_citation = bool(citation) and re.fullmatch(r"(\[\d+\]\s*,?\s*)+", citation) is not None
        source_marker = source_page or citation
        if opaque_citation:
            source_marker = source_page

        if source_marker.startswith("http"):
            if source_title:
                parts = [source_title, source_marker]
            else:
                parts = [source_marker]
        else:
            if not source_title:
                return ""
            parts = [source_title]
            if source_marker:
                parts.append(source_marker)

        if date:
            parts.append(date)
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _write_pdf_from_html(html_path: Path, pdf_path: Path) -> None:
        errors: list[str] = []
        try:
            WriterAgent._write_pdf_with_playwright(html_path, pdf_path)
            return
        except Exception as exc:
            errors.append(f"[playwright]\n{exc}")
        try:
            WriterAgent._write_pdf_with_chrome_cli(html_path, pdf_path)
            return
        except Exception as exc:
            errors.append(f"[chrome-cli]\n{exc}")
        error_path = pdf_path.parent / "chrome_pdf_error.log"
        error_path.write_text("\n\n".join(errors), encoding="utf-8")
        WriterAgent._write_pdf_fallback(html_path, pdf_path)

    @staticmethod
    def _write_pdf_with_playwright(html_path: Path, pdf_path: Path) -> None:
        html = html_path.read_text(encoding="utf-8")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            launch_attempts: list[dict[str, Any]] = [
                {
                    "headless": True,
                    "args": WriterAgent._chrome_launch_args(),
                }
            ]
            if CHROME_BINARY.exists():
                launch_attempts.append(
                    {
                        "channel": "chrome",
                        "headless": True,
                        "args": WriterAgent._chrome_launch_args(),
                    }
                )
            browser = None
            launch_errors: list[str] = []
            for launch_kwargs in launch_attempts:
                try:
                    browser = playwright.chromium.launch(**launch_kwargs)
                    break
                except Exception as exc:
                    label = launch_kwargs.get("channel", "bundled-chromium")
                    launch_errors.append(f"{label}: {exc}")
            if browser is None:
                raise RuntimeError(" ; ".join(launch_errors))
            try:
                page = browser.new_page()
                page.emulate_media(media="print")
                page.set_content(html, wait_until="load")
                page.pdf(
                    path=str(pdf_path.resolve()),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                    margin={
                        "top": "14mm",
                        "right": "14mm",
                        "bottom": "14mm",
                        "left": "14mm",
                    },
                )
            finally:
                browser.close()

    @staticmethod
    def _write_pdf_with_chrome_cli(html_path: Path, pdf_path: Path) -> None:
        if not CHROME_BINARY.exists():
            raise RuntimeError("Google Chrome binary not found for HTML to PDF conversion.")
        with tempfile.TemporaryDirectory(prefix="battery_strategy_chrome_") as profile_dir:
            command = [
                str(CHROME_BINARY),
                "--headless=new",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile_dir}",
                f"--print-to-pdf={pdf_path.resolve()}",
                html_path.resolve().as_uri(),
            ]
            if sys.platform.startswith("linux"):
                command.append("--no-sandbox")
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
            )
            if result.returncode != 0 or not pdf_path.exists():
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                raise RuntimeError(
                    "Chrome CLI PDF conversion failed.\n"
                    f"exit_code={result.returncode}\nstdout={stdout}\nstderr={stderr}"
                )

    @staticmethod
    def _chrome_launch_args() -> list[str]:
        args = [
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        if sys.platform.startswith("linux"):
            args.append("--no-sandbox")
        return args

    @staticmethod
    def _write_pdf_fallback(html_path: Path, pdf_path: Path) -> None:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        except Exception:
            return

        font_name = WriterAgent._register_reportlab_font(pdfmetrics, TTFont)
        styles = getSampleStyleSheet()
        body = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15,
        )

        raw_html = html_path.read_text(encoding="utf-8")
        stripped = re.sub(r"<style.*?</style>", "", raw_html, flags=re.DOTALL)
        stripped = re.sub(r"<script.*?</script>", "", stripped, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "\n", stripped)
        paragraphs = [WriterAgent._clean_text(item) for item in text.split("\n") if WriterAgent._clean_text(item)]

        story: list[Any] = []
        for item in paragraphs:
            story.extend([Paragraph(escape(item), body), Spacer(1, 4)])

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=16 * mm,
            rightMargin=16 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
        )
        doc.build(story)

    @staticmethod
    def _register_reportlab_font(pdfmetrics: Any, ttfont_cls: Any) -> str:
        candidates = [
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                font_name = f"ReportFont-{candidate.stem}"
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(ttfont_cls(font_name, str(candidate)))
                return font_name
            except Exception:
                continue
        return "Helvetica"

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _report_timestamp() -> str:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")

    @staticmethod
    def _report_file_timestamp() -> str:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _report_css() -> str:
        return """
@page {
  size: A4;
  margin: 16mm;
}

:root {
  --ink: #1f2937;
  --navy: #16324f;
  --sand: #f5efe6;
  --paper: #fffdfa;
  --line: #e5d8c7;
  --accent: #c98b4a;
  --strength: #eaf4ec;
  --weakness: #fbefef;
  --opportunity: #eef4fb;
  --threat: #fbf4e8;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  color: var(--ink);
  background: #efe8dc;
  font-family: "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", sans-serif;
  line-height: 1.65;
  font-size: 11pt;
}

.report-shell {
  padding: 14px;
}

.cover {
  background: linear-gradient(135deg, #16324f 0%, #214c72 58%, #c98b4a 100%);
  color: white;
  border-radius: 20px;
  padding: 28px 30px;
  margin-bottom: 18px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 10pt;
  opacity: 0.82;
}

.cover h1 {
  margin: 10px 0 10px;
  font-size: 24pt;
  line-height: 1.25;
}

.subtitle, .timestamp {
  margin: 0;
  font-size: 11pt;
  opacity: 0.92;
}

.timestamp {
  margin-top: 12px;
  font-size: 9.5pt;
}

.section {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 20px;
  margin-bottom: 14px;
  box-shadow: 0 8px 22px rgba(22, 50, 79, 0.05);
}

.section h2 {
  margin: 0 0 14px;
  color: var(--navy);
  font-size: 16pt;
  line-height: 1.3;
}

.section h3 {
  margin: 0 0 10px;
  color: var(--navy);
  font-size: 12pt;
}

.prose p {
  margin: 0 0 12px;
}

.prose ul, .prose ol {
  margin: 8px 0 12px 20px;
}

.prose li + li {
  margin-top: 4px;
}

.summary .prose p {
  font-size: 11.5pt;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 14px;
  table-layout: fixed;
  font-size: 9.5pt;
}

th, td {
  border: 1px solid #d9ccb8;
  padding: 8px 9px;
  vertical-align: top;
  word-break: break-word;
}

th {
  background: var(--navy);
  color: white;
}

tr:nth-child(even) td {
  background: #fbf7f1;
}

.evidence-panel {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px dashed #d9ccb8;
}

.evidence-title {
  color: #7d5b3f;
  font-size: 9pt;
  font-weight: 700;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.evidence-list {
  margin: 0;
  padding-left: 18px;
  font-size: 9.2pt;
}

.evidence-list li + li {
  margin-top: 6px;
}

.evidence-claim {
  display: block;
}

.evidence-cite {
  display: block;
  color: #7c6b57;
  margin-top: 2px;
}

.swot-company + .swot-company {
  margin-top: 18px;
}

.swot-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.swot-card {
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid var(--line);
}

.swot-card ul {
  margin: 8px 0 0 18px;
}

.swot-title {
  font-weight: 700;
  color: var(--navy);
}

.swot-card.strength { background: var(--strength); }
.swot-card.weakness { background: var(--weakness); }
.swot-card.opportunity { background: var(--opportunity); }
.swot-card.threat { background: var(--threat); }

.reference-list {
  margin: 0;
  padding-left: 20px;
}
"""
