from __future__ import annotations

from dataclasses import dataclass

from battery_strategy.agents.runtime import AgentRuntime
from battery_strategy.agents.supervisor import Supervisor
from battery_strategy.rag.retrieval import HybridRetriever
from battery_strategy.tools.balance import SearchBalanceChecker
from battery_strategy.tools.llm import BaseLLM, OpenAIResponsesLLM
from battery_strategy.tools.web_search import DuckDuckGoSearcher, NoOpSearcher
from battery_strategy.utils.settings import (
    Manifest,
    RuntimeConfig,
    load_manifest,
    load_runtime_config,
)
from battery_strategy.utils.types import GlobalState


@dataclass(slots=True)
class PipelineFactory:
    config: RuntimeConfig
    manifest: Manifest

    @classmethod
    def from_config(cls, config_path: str) -> PipelineFactory:
        config = load_runtime_config(config_path)
        manifest = load_manifest(config.manifest_path)
        return cls(config=config, manifest=manifest)

    def build_runtime(self) -> AgentRuntime:
        llm: BaseLLM = OpenAIResponsesLLM(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_output_tokens=self.config.llm.max_output_tokens,
        )
        retriever = HybridRetriever.from_dir(
            str(self.config.index_dir),
            self.config.retrieval.embedding_model,
            dense_top_k=self.config.retrieval.dense_top_k,
            sparse_top_k=self.config.retrieval.sparse_top_k,
            final_top_k=self.config.retrieval.final_top_k,
            use_reranker=self.config.retrieval.use_reranker,
            reranker_model=self.config.retrieval.reranker_model,
        )
        if self.config.web_search.enabled:
            searcher = DuckDuckGoSearcher(
                region=self.config.web_search.region,
                max_results_per_query=self.config.web_search.max_results_per_query,
                fetch_full_text=self.config.web_search.fetch_full_text,
            )
        else:
            searcher = NoOpSearcher()

        return AgentRuntime(
            config=self.config,
            llm=llm,
            retriever=retriever,
            searcher=searcher,
            balance_checker=SearchBalanceChecker(),
        )

    def create_initial_state(self, goal: str) -> GlobalState:
        return {
            "goal": goal,
            "criteria": self.config.workflow.criteria,
            "comparison_axes": self.config.workflow.comparison_axes,
            "corpus_manifest": self.manifest.sources,
            "query_plan": {},
            "market_context": {"summary": "", "bullet_points": [], "normalized_evidence": []},
            "company_results": {},
            "comparison_matrix": [],
            "swot": {},
            "bias_flags": [],
            "unresolved_conflicts": [],
            "retry_plan": None,
            "next_step": "start",
            "status": "initialized",
            "draft_sections": {},
            "references": [],
        }

    def run(self, goal: str) -> GlobalState:
        runtime = self.build_runtime()
        state = self.create_initial_state(goal)
        supervisor = Supervisor(runtime)
        return supervisor.run(state)
