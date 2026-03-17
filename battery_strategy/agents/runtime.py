from __future__ import annotations

from dataclasses import dataclass

from battery_strategy.rag.retrieval import HybridRetriever
from battery_strategy.tools.balance import SearchBalanceChecker
from battery_strategy.tools.llm import BaseLLM
from battery_strategy.tools.web_search import BaseSearcher
from battery_strategy.utils.settings import RuntimeConfig


@dataclass(slots=True)
class AgentRuntime:
    config: RuntimeConfig
    llm: BaseLLM
    retriever: HybridRetriever
    searcher: BaseSearcher
    balance_checker: SearchBalanceChecker
