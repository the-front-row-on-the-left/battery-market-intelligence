from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from battery_strategy.rag.chunking import TokenApproxChunker
from battery_strategy.rag.embedding import LocalEmbedder
from battery_strategy.rag.pdf_loader import extract_pdf_pages
from battery_strategy.utils.common import dump_json, ensure_dir, tokenize_for_bm25
from battery_strategy.utils.settings import RuntimeConfig
from battery_strategy.utils.types import ChunkRecord, SourceManifestItem


@dataclass(slots=True)
class IndexBundle:
    index: faiss.Index
    chunks: list[ChunkRecord]
    tokenized_corpus: list[list[str]]
    bm25: BM25Okapi


INDEX_FILE = "dense.faiss"
CHUNKS_FILE = "chunks.jsonl"
TOKENIZED_FILE = "bm25_corpus.json"
META_FILE = "index_meta.json"


def build_index(
    sources: Iterable[SourceManifestItem],
    config: RuntimeConfig,
) -> Path:
    index_dir = ensure_dir(config.index_dir)
    chunker = TokenApproxChunker(
        target_tokens=config.retrieval.chunk_size_tokens,
        overlap_tokens=config.retrieval.chunk_overlap_tokens,
    )
    embedder = LocalEmbedder(config.retrieval.embedding_model)

    all_chunks: list[ChunkRecord] = []
    for source in sources:
        pages = extract_pdf_pages(source["local_path"])
        all_chunks.extend(chunker.chunk_pages(source, pages))

    texts = [chunk["text"] for chunk in all_chunks]
    vectors = embedder.encode(texts)
    dense_index = faiss.IndexFlatIP(vectors.shape[1])
    dense_index.add(vectors)

    tokenized_corpus = [tokenize_for_bm25(text) for text in texts]

    faiss.write_index(dense_index, str(index_dir / INDEX_FILE))
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fp:
        for chunk in all_chunks:
            fp.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    dump_json(tokenized_corpus, index_dir / TOKENIZED_FILE)
    dump_json(
        {
            "embedding_model": config.retrieval.embedding_model,
            "reranker_model": config.retrieval.reranker_model,
            "chunk_count": len(all_chunks),
            "vector_dim": int(vectors.shape[1]),
        },
        index_dir / META_FILE,
    )
    return index_dir


def load_index(index_dir: str | Path) -> IndexBundle:
    resolved_dir = Path(index_dir).resolve()
    dense_index = faiss.read_index(str(resolved_dir / INDEX_FILE))

    chunks: list[ChunkRecord] = []
    with (resolved_dir / CHUNKS_FILE).open("r", encoding="utf-8") as fp:
        for line in fp:
            chunks.append(json.loads(line))

    with (resolved_dir / TOKENIZED_FILE).open("r", encoding="utf-8") as fp:
        tokenized_corpus = json.load(fp)

    bm25 = BM25Okapi(tokenized_corpus)
    return IndexBundle(
        index=dense_index, chunks=chunks, tokenized_corpus=tokenized_corpus, bm25=bm25
    )
