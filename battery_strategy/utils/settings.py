from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from battery_strategy.utils.types import Axis, SourceManifestItem


@dataclass(slots=True)
class LLMSettings:
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_output_tokens: int = 4000


@dataclass(slots=True)
class RetrievalSettings:
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chunk_size_tokens: int = 900
    chunk_overlap_tokens: int = 120
    dense_top_k: int = 8
    sparse_top_k: int = 8
    final_top_k: int = 8
    use_reranker: bool = False
    index_dir: str = "outputs/index"


@dataclass(slots=True)
class WebSearchSettings:
    enabled: bool = True
    provider: str = "duckduckgo"
    region: str = "kr-kr"
    max_results_per_query: int = 5
    fetch_full_text: bool = True


@dataclass(slots=True)
class ExecutionSettings:
    max_bias_retries: int = 2
    max_subgraph_balance_retries: int = 1
    output_dir: str = "outputs"


@dataclass(slots=True)
class WorkflowSettings:
    companies: list[str] = field(default_factory=lambda: ["LGES", "CATL"])
    comparison_axes: list[Axis] = field(
        default_factory=lambda: [
            "portfolio",
            "commercialization",
            "manufacturing",
            "technology",
            "ecosystem",
            "risk",
            "strategy_horizon",
        ]
    )
    criteria: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str
    llm: LLMSettings
    retrieval: RetrievalSettings
    web_search: WebSearchSettings
    execution: ExecutionSettings
    workflow: WorkflowSettings
    data_manifest_path: str
    base_dir: Path

    @property
    def output_dir(self) -> Path:
        return (self.base_dir / self.execution.output_dir).resolve()

    @property
    def index_dir(self) -> Path:
        return (self.base_dir / self.retrieval.index_dir).resolve()

    @property
    def manifest_path(self) -> Path:
        return (self.base_dir / self.data_manifest_path).resolve()


@dataclass(slots=True)
class Manifest:
    sources: list[SourceManifestItem]


def _merge_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    return raw.get(key, {}) if isinstance(raw.get(key, {}), dict) else {}


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).resolve()
    base_dir = (
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    config = RuntimeConfig(
        project_name=raw.get("project_name", "battery_strategy_multi_agent"),
        llm=LLMSettings(**_merge_dict(raw, "llm")),
        retrieval=RetrievalSettings(**_merge_dict(raw, "retrieval")),
        web_search=WebSearchSettings(**_merge_dict(raw, "web_search")),
        execution=ExecutionSettings(**_merge_dict(raw, "execution")),
        workflow=WorkflowSettings(**_merge_dict(raw, "workflow")),
        data_manifest_path=raw.get("data_manifest_path", "configs/data_manifest.yaml"),
        base_dir=base_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.index_dir.mkdir(parents=True, exist_ok=True)
    return config


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    sources = raw.get("sources", [])
    for item in sources:
        local_path = manifest_path.parent.parent / item["local_path"]
        item["local_path"] = str(local_path.resolve())
    return Manifest(sources=sources)
