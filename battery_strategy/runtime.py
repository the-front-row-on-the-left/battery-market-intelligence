from __future__ import annotations

from dataclasses import dataclass

from battery_strategy.balance import SearchBalanceChecker
from battery_strategy.llm import BaseLLM
from battery_strategy.retrieval import HybridRetriever
from battery_strategy.settings import RuntimeConfig
from battery_strategy.web_search import BaseSearcher


@dataclass(slots=True)
class AgentRuntime:
    config: RuntimeConfig
    llm: BaseLLM
    retriever: HybridRetriever
    searcher: BaseSearcher
    balance_checker: SearchBalanceChecker
