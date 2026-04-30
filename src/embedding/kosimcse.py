from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

MODEL_NAME = "jhgan/ko-sroberta-multitask"


class KoSimCSEEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, max_seq_length: int = 512) -> None:
        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = max_seq_length
        self._model.tokenizer.model_max_length = max_seq_length

    @property
    def tokenizer(self) -> Any:
        return self._model.tokenizer

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_tensor=False)
        return [e.tolist() for e in embeddings]
