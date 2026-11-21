"""Text embeddings, behind a small interface so tests can use a fake model."""

from collections.abc import Sequence
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, runs on CPU via ONNX

Vector = list[float]


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class FastEmbedEmbedder:
    """Local embeddings with fastembed (ONNX runtime, no PyTorch).

    BGE models are asymmetric: queries get an instruction prefix that documents
    don't. fastembed's `query_embed` / `passage_embed` handle that for us.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from fastembed import TextEmbedding  # heavy import, deferred until needed

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [vector.tolist() for vector in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> Vector:
        return next(iter(self._model.query_embed(text))).tolist()
