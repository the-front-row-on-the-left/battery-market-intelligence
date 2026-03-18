from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class LocalEmbedder:
    model_name: str
    normalize: bool = True
    batch_size: int = 4
    max_seq_length: int = 512
    model: SentenceTransformer = field(init=False)

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
