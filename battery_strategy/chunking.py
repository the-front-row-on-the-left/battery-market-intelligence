from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from battery_strategy.pdf_loader import PageText
from battery_strategy.types import ChunkRecord, SourceManifestItem
from battery_strategy.utils import format_page_range


@dataclass(slots=True)
class TokenApproxChunker:
    target_tokens: int = 900
    overlap_tokens: int = 120

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text.split()))

    def chunk_pages(self, source: SourceManifestItem, pages: Iterable[PageText]) -> list[ChunkRecord]:
        page_list = list(pages)
        chunks: list[ChunkRecord] = []
        buffer_text: list[str] = []
        buffer_pages: list[PageText] = []
        chunk_idx = 0

        for page in page_list:
            page_tokens = self._count_tokens(page.text)
            current_tokens = self._count_tokens("\n\n".join(buffer_text)) if buffer_text else 0
            if buffer_text and current_tokens + page_tokens > self.target_tokens:
                chunks.append(self._build_chunk(source, buffer_pages, buffer_text, chunk_idx))
                chunk_idx += 1
                buffer_pages, buffer_text = self._overlap(buffer_pages, buffer_text)
            buffer_pages.append(page)
            buffer_text.append(page.text)

        if buffer_text:
            chunks.append(self._build_chunk(source, buffer_pages, buffer_text, chunk_idx))
        return chunks

    def _overlap(self, pages: list[PageText], texts: list[str]) -> tuple[list[PageText], list[str]]:
        if not texts:
            return [], []
        kept_pages: list[PageText] = []
        kept_texts: list[str] = []
        token_count = 0
        for page, text in reversed(list(zip(pages, texts))):
            tokens = self._count_tokens(text)
            if token_count + tokens > self.overlap_tokens and kept_texts:
                break
            kept_pages.insert(0, page)
            kept_texts.insert(0, text)
            token_count += tokens
        return kept_pages, kept_texts

    @staticmethod
    def _build_chunk(
        source: SourceManifestItem,
        pages: list[PageText],
        texts: list[str],
        chunk_idx: int,
    ) -> ChunkRecord:
        first_page = pages[0]
        last_page = pages[-1]
        return {
            "chunk_id": f"{source['id']}-{chunk_idx:04d}",
            "source_id": source["id"],
            "source_group": source["group"],
            "source_title": source["title"],
            "source_type": source["source_type"],
            "source_url": source["url"],
            "reference": source["reference"],
            "page_start": first_page.page_num,
            "page_end": last_page.page_num,
            "display_page_start": first_page.display_page,
            "display_page_end": last_page.display_page,
            "text": "\n\n".join(texts),
            "language": source["language"],
        }



def chunk_page_range(chunk: ChunkRecord) -> str:
    return format_page_range(chunk["display_page_start"], chunk["display_page_end"])
