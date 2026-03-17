from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from battery_strategy.chunking import chunk_page_range
from battery_strategy.embedding import LocalEmbedder
from battery_strategy.index_store import IndexBundle, load_index
from battery_strategy.types import ChunkRecord, EntityName, RetrievedChunk
from battery_strategy.utils import tokenize_for_bm25


class IdentityReranker:
    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return items


class OptionalFlagReranker:
    def __init__(self, model_name: str) -> None:
        self.enabled = False
        try:
            from FlagEmbedding import FlagReranker
        except Exception:  # noqa: BLE001
            self._model = None
            return
        self._model = FlagReranker(model_name, use_fp16=False)
        self.enabled = True

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self.enabled or not items:
            return items
        pairs = [(query, item["text"]) for item in items]
        scores = self._model.compute_score(pairs)
        enriched = []
        for item, score in zip(items, scores):
            updated = dict(item)
            updated["score"] = float(score)
            enriched.append(updated)
        return sorted(enriched, key=lambda row: row["score"], reverse=True)


@dataclass(slots=True)
class HybridRetriever:
    index_bundle: IndexBundle
    embedder: LocalEmbedder
    dense_top_k: int = 8
    sparse_top_k: int = 8
    final_top_k: int = 8
    reranker: IdentityReranker | OptionalFlagReranker | None = None

    @classmethod
    def from_dir(
        cls,
        index_dir: str,
        embedding_model: str,
        *,
        dense_top_k: int,
        sparse_top_k: int,
        final_top_k: int,
        use_reranker: bool,
        reranker_model: str,
    ) -> "HybridRetriever":
        bundle = load_index(index_dir)
        reranker = OptionalFlagReranker(reranker_model) if use_reranker else IdentityReranker()
        return cls(
            index_bundle=bundle,
            embedder=LocalEmbedder(embedding_model),
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            final_top_k=final_top_k,
            reranker=reranker,
        )

    def search(
        self,
        query: str,
        *,
        label: str,
        source_groups: Iterable[EntityName] | None = None,
    ) -> list[RetrievedChunk]:
        filters = set(source_groups or [])
        dense_results = self._dense_search(query, label, filters)
        sparse_results = self._sparse_search(query, label, filters)
        fused = self._rrf_fuse(dense_results, sparse_results)
        if self.reranker:
            fused = self.reranker.rerank(query, fused)
        return fused[: self.final_top_k]

    def _dense_search(
        self,
        query: str,
        label: str,
        filters: set[EntityName],
    ) -> list[RetrievedChunk]:
        query_vec = self.embedder.encode([query])
        n_search = min(max(self.dense_top_k * 4, 20), len(self.index_bundle.chunks))
        scores, indices = self.index_bundle.index.search(query_vec, n_search)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.index_bundle.chunks[int(idx)]
            if filters and chunk["source_group"] not in filters:
                continue
            results.append(self._to_hit(query, label, chunk, float(score), "dense"))
            if len(results) >= self.dense_top_k:
                break
        return results

    def _sparse_search(
        self,
        query: str,
        label: str,
        filters: set[EntityName],
    ) -> list[RetrievedChunk]:
        tokenized_query = tokenize_for_bm25(query)
        scores = self.index_bundle.bm25.get_scores(tokenized_query)
        ranked_idx = np.argsort(scores)[::-1]
        results: list[RetrievedChunk] = []
        for idx in ranked_idx:
            chunk = self.index_bundle.chunks[int(idx)]
            if filters and chunk["source_group"] not in filters:
                continue
            results.append(self._to_hit(query, label, chunk, float(scores[idx]), "sparse"))
            if len(results) >= self.sparse_top_k:
                break
        return results

    def _rrf_fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        *,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        ranked: dict[str, dict] = {}
        for rank, item in enumerate(dense_results, start=1):
            ranked.setdefault(item["chunk_id"], dict(item))
            ranked[item["chunk_id"]]["score"] = ranked[item["chunk_id"]].get("score", 0.0) + 1.0 / (k + rank)
        for rank, item in enumerate(sparse_results, start=1):
            ranked.setdefault(item["chunk_id"], dict(item))
            ranked[item["chunk_id"]]["score"] = ranked[item["chunk_id"]].get("score", 0.0) + 1.0 / (k + rank)
        return sorted(ranked.values(), key=lambda row: row["score"], reverse=True)

    @staticmethod
    def _to_hit(query: str, label: str, chunk: ChunkRecord, score: float, mode: str) -> RetrievedChunk:
        return {
            "query": query,
            "query_label": label,  # type: ignore[typeddict-item]
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "source_group": chunk["source_group"],
            "source_title": chunk["source_title"],
            "source_type": chunk["source_type"],
            "source_url": chunk["source_url"],
            "reference": chunk["reference"],
            "page_range": chunk_page_range(chunk),
            "text": chunk["text"],
            "score": score,
            "retrieval_mode": mode,  # type: ignore[typeddict-item]
        }
