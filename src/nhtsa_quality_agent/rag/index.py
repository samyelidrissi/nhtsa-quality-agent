"""Build the vector index (Chroma, persisted on disk) from the processed tables.

Run with:  uv run python -m nhtsa_quality_agent.rag.index
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection

from nhtsa_quality_agent.config import DEFAULT_DATA_DIR
from nhtsa_quality_agent.rag.documents import Document, complaint_documents, recall_documents
from nhtsa_quality_agent.rag.embeddings import Embedder, FastEmbedEmbedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "nhtsa-documents"
BATCH_SIZE = 256


def get_collection(client: chromadb.ClientAPI) -> Collection:
    """Cosine-distance collection; we always pass our own embeddings."""
    return client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=None,  # never let Chroma silently embed with its default model
        configuration={"hnsw": {"space": "cosine"}},
    )


def add_documents(
    collection: Collection,
    documents: Sequence[Document],
    embedder: Embedder,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Embed and upsert documents in batches (upsert keeps re-runs idempotent)."""
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        collection.upsert(
            ids=[doc.id for doc in batch],
            documents=[doc.text for doc in batch],
            metadatas=[doc.metadata for doc in batch],
            embeddings=embedder.embed_documents([doc.text for doc in batch]),
        )
        logger.info("indexed %d/%d", min(start + batch_size, len(documents)), len(documents))


def load_documents(processed_dir: Path) -> list[Document]:
    complaints = pd.read_parquet(processed_dir / "complaints.parquet")
    recalls = pd.read_parquet(processed_dir / "recalls.parquet")
    recall_vehicles = pd.read_parquet(processed_dir / "recall_vehicles.parquet")
    return complaint_documents(complaints) + recall_documents(recalls, recall_vehicles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    documents = load_documents(args.data_dir / "processed")
    logger.info("built %d documents", len(documents))
    client = chromadb.PersistentClient(path=str(args.data_dir / "index"))
    collection = get_collection(client)
    add_documents(collection, documents, FastEmbedEmbedder())
    logger.info("collection %s holds %d documents", COLLECTION_NAME, collection.count())


if __name__ == "__main__":
    main()
