from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from battery_strategy.utils import guess_display_page


@dataclass(slots=True)
class PageText:
    page_num: int
    display_page: str
    text: str



def extract_pdf_pages(path: str | Path) -> list[PageText]:
    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(PageText(page_num=idx, display_page=guess_display_page(text, idx), text=text))
    return pages
