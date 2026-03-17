from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")



def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory



def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    match = CODE_FENCE_PATTERN.match(cleaned)
    return match.group(1).strip() if match else cleaned



def safe_json_loads(text: str, *, fallback: Any) -> Any:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return fallback
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return fallback
        return fallback



def truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."



def tokenize_for_bm25(text: str) -> list[str]:
    lowered = text.lower()
    return re.findall(r"[\w\-]+", lowered)



def domain_from_url(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower()



def dedupe_by_key(items: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    ordered: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in items:
        ordered[str(item.get(key, ""))] = item
    return list(ordered.values())



def parse_annotated_query(query: str) -> tuple[str, str]:
    parts = query.split("|", maxsplit=1)
    if len(parts) == 1:
        return "neutral", query
    label, raw_query = parts
    return label, raw_query



def format_page_range(start: str, end: str) -> str:
    return start if start == end else f"{start}-{end}"



def guess_display_page(text: str, fallback: int) -> str:
    if match := re.search(r"PAGE\s*\|\s*(\d+)", text):
        return f"p.{match.group(1)}"
    if match := re.match(r"\s*(\d{1,3})\s", text):
        return f"p.{match.group(1)}"
    if match := re.search(r"\n(\d{1,3})\s*$", text):
        return f"p.{match.group(1)}"
    return f"clip-p.{fallback}"



def dump_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
