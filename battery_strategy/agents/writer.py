from __future__ import annotations

from pathlib import Path

from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.tools.prompts import writer_prompt
from battery_strategy.utils.common import dump_json
from battery_strategy.utils.types import GlobalState


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
        fallback = {
            "draft_sections": self._fallback_sections(global_state),
        }
        parsed = self.runtime.llm.json(instructions, user_prompt, fallback=fallback)
        draft_sections = parsed.get("draft_sections", fallback["draft_sections"])
        global_state["draft_sections"] = draft_sections
        self._write_outputs(global_state)
        return global_state

    def _write_outputs(self, global_state: GlobalState) -> None:
        output_dir = Path(self.runtime.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        final_markdown = self._sections_to_markdown(global_state["draft_sections"])
        (output_dir / "final_report.md").write_text(final_markdown, encoding="utf-8")
        (output_dir / "references.txt").write_text(
            "\n".join(global_state["references"]), encoding="utf-8"
        )
        dump_json(global_state, output_dir / "final_state.json")

    @staticmethod
    def _sections_to_markdown(sections: dict[str, str]) -> str:
        ordered = [
            "SUMMARY",
            "1. 시장 배경",
            "2. LGES 전략",
            "3. CATL 전략",
            "4. 전략 비교",
            "5. SWOT 분석",
            "6. 종합 시사점",
            "REFERENCE",
        ]
        blocks: list[str] = []
        for title in ordered:
            if title == "SUMMARY":
                blocks.append(f"# {title}\n\n{sections.get(title, '').strip()}")
            elif title == "REFERENCE":
                blocks.append(f"# {title}\n\n{sections.get(title, '').strip()}")
            else:
                blocks.append(f"## {title}\n\n{sections.get(title, '').strip()}")
        return "\n\n".join(blocks).strip() + "\n"

    @staticmethod
    def _fallback_sections(global_state: GlobalState) -> dict[str, str]:
        comparison_lines = []
        for row in global_state.get("comparison_matrix", []):
            comparison_lines.append(
                f"- **{row['axis']}**: LGES={row['lges_summary']} / CATL={row['catl_summary']} / 차이={row['difference']}"
            )
        references = "\n".join(f"- {item}" for item in global_state.get("references", []))
        return {
            "SUMMARY": "LGES와 CATL 모두 EV 둔화에 대응해 포트폴리오 다각화를 추진하고 있으나, LGES는 고객/생산 거점/서비스 확장과 ESS·BaaS 중심으로, CATL은 ESS·재활용·교환형 배터리·나트륨이온 등 생태계 확장 중심으로 접근한다.",
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
            "5. SWOT 분석": str(global_state.get("swot", {})),
            "6. 종합 시사점": "시장 구조 변화 속에서 EV 외 수요처와 생태계 확장이 경쟁 우위를 좌우한다.",
            "REFERENCE": references,
        }
