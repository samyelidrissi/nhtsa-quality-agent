"""Semantic search over the index, with optional metadata filters."""

from dataclasses import dataclass
from typing import Any, Literal

from chromadb.api.models.Collection import Collection

from nhtsa_quality_agent.rag.embeddings import Embedder

Source = Literal["complaint", "recall"]


@dataclass(frozen=True)
class Filters:
    make: str | None = None
    model: str | None = None
    model_year: int | None = None
    source: Source | None = None

    def to_where(self) -> dict[str, Any] | None:
        """Translate to a Chroma `where` clause (None when no filter is set)."""
        conditions: list[dict[str, Any]] = [
            {key: value.lower() if isinstance(value, str) else value}
            for key, value in vars(self).items()
            if value is not None
        ]
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity, 1 = identical

    @property
    def citation(self) -> str:
        """Short label the LLM must cite, e.g. `C-11777952` or `R-20V682000`."""
        prefix = "C" if self.metadata["source"] == "complaint" else "R"
        return f"{prefix}-{self.metadata['source_id']}"


class Retriever:
    def __init__(self, collection: Collection, embedder: Embedder) -> None:
        self._collection = collection
        self._embedder = embedder

    def search(self, query: str, k: int = 8, filters: Filters | None = None) -> list[Hit]:
        """Top-k most similar documents, one per complaint/recall campaign."""
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            # Over-fetch: a recall campaign is indexed once per covered vehicle.
            n_results=k * 3,
            where=(filters or Filters()).to_where(),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        seen: set[str] = set()
        for doc_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],  # type: ignore[index]
            result["metadatas"][0],  # type: ignore[index]
            result["distances"][0],  # type: ignore[index]
            strict=True,
        ):
            hit = Hit(id=doc_id, text=text, metadata=dict(metadata), score=1 - distance)
            if hit.citation in seen:
                continue
            seen.add(hit.citation)
            hits.append(hit)
            if len(hits) == k:
                break
        return hits
