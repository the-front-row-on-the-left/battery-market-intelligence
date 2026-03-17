from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class LocalEmbedder:
    model_name: str
    normalize: bool = True

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
