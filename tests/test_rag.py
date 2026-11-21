from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import chromadb
import pandas as pd
import pytest

from nhtsa_quality_agent.rag.documents import complaint_documents, recall_documents
from nhtsa_quality_agent.rag.embeddings import Vector
from nhtsa_quality_agent.rag.generate import (
    build_user_message,
    extract_citations,
    generate_answer,
)
from nhtsa_quality_agent.rag.index import add_documents, get_collection
from nhtsa_quality_agent.rag.retrieve import Filters, Hit, Retriever

# --- fakes -------------------------------------------------------------------

KEYWORDS = ["brake", "engine", "airbag", "fuel"]


class KeywordEmbedder:
    """Deterministic fake: one dimension per keyword, so similarity is predictable."""

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        lowered = text.lower()
        return [float(lowered.count(word)) + 0.01 for word in KEYWORDS]


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"


@dataclass
class FakeMessages:
    reply: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse([FakeBlock(self.reply)])


@dataclass
class FakeClient:
    messages: FakeMessages


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def complaints() -> pd.DataFrame:
    rows = [
        ("1", "toyota", "camry", 2020, ["SERVICE BRAKES"], "The brake pedal went to the floor."),
        ("2", "toyota", "camry", 2021, ["ENGINE"], "Engine stalled, engine light on."),
        ("3", "jeep", "grand cherokee", 2021, ["AIR BAGS"], "Airbag light stays on."),
    ]
    return pd.DataFrame(
        {
            "odi_number": [r[0] for r in rows],
            "make": [r[1] for r in rows],
            "model": [r[2] for r in rows],
            "model_year": [r[3] for r in rows],
            "components": [r[4] for r in rows],
            "summary": [r[5] for r in rows],
            "crash": [False, True, False],
            "fire": [False, False, False],
            "date_filed": pd.to_datetime(["2021-01-01", "2022-01-01", "2022-06-01"]),
        }
    )


@pytest.fixture
def recalls() -> tuple[pd.DataFrame, pd.DataFrame]:
    recalls = pd.DataFrame(
        {
            "campaign_number": ["20V682000"],
            "report_date": pd.to_datetime(["2020-11-04"]),
            "component": ["FUEL SYSTEM, GASOLINE:DELIVERY:FUEL PUMP"],
            "summary": ["The fuel pump may fail."],
            "consequence": ["Engine stall."],
            "remedy": ["Replace the fuel pump."],
            "park_it": [False],
        }
    )
    links = pd.DataFrame(
        {
            "campaign_number": ["20V682000", "20V682000"],
            "make": ["toyota", "toyota"],
            "model": ["camry", "camry"],
            "model_year": [2020, 2021],
        }
    )
    return recalls, links


@pytest.fixture
def retriever(complaints: pd.DataFrame, recalls: tuple[pd.DataFrame, pd.DataFrame]) -> Retriever:
    documents = complaint_documents(complaints) + recall_documents(*recalls)
    collection = get_collection(chromadb.EphemeralClient())
    collection.delete(ids=collection.get()["ids"] or ["none"])  # isolate from other tests
    add_documents(collection, documents, KeywordEmbedder(), batch_size=2)
    return Retriever(collection, KeywordEmbedder())


def hit(source: str, source_id: str) -> Hit:
    return Hit(
        id=f"x-{source_id}",
        text=f"text {source_id}",
        metadata={"source": source, "source_id": source_id},
        score=0.9,
    )


# --- documents -----------------------------------------------------------------


def test_complaint_document_text_and_metadata(complaints: pd.DataFrame) -> None:
    doc = complaint_documents(complaints)[0]

    assert doc.id == "C-1"
    assert doc.text.startswith("Toyota Camry 2020 complaint. Components: SERVICE BRAKES.")
    assert doc.metadata["model_year"] == 2020
    assert doc.metadata["components"] == "SERVICE BRAKES"
    assert all(not isinstance(v, list) for v in doc.metadata.values())  # Chroma: scalars only


def test_recall_documents_one_per_vehicle(recalls: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    docs = recall_documents(*recalls)

    assert [d.id for d in docs] == ["R-20V682000-camry-2020", "R-20V682000-camry-2021"]
    assert {d.metadata["source_id"] for d in docs} == {"20V682000"}


# --- retrieval -----------------------------------------------------------------


def test_filters_to_where() -> None:
    assert Filters().to_where() is None
    assert Filters(make="Jeep").to_where() == {"make": "jeep"}
    assert Filters(make="toyota", model_year=2020).to_where() == {
        "$and": [{"make": "toyota"}, {"model_year": 2020}]
    }


def test_search_ranks_most_similar_first(retriever: Retriever) -> None:
    hits = retriever.search("brake failure", k=2)

    assert hits[0].citation == "C-1"


def test_search_applies_metadata_filters(retriever: Retriever) -> None:
    hits = retriever.search("airbag", k=5, filters=Filters(make="toyota"))

    assert hits
    assert all(h.metadata["make"] == "toyota" for h in hits)


def test_search_dedupes_recall_campaigns(retriever: Retriever) -> None:
    hits = retriever.search("fuel pump", k=5, filters=Filters(source="recall"))

    assert [h.citation for h in hits] == ["R-20V682000"]  # indexed twice, returned once


# --- generation ----------------------------------------------------------------


def test_extract_citations_keeps_order_without_duplicates() -> None:
    text = "Pumps fail [R-20V682000]. Owners report stalls [C-2][C-1]. Again [C-2]."

    assert extract_citations(text) == ["R-20V682000", "C-2", "C-1"]


def test_user_message_puts_sources_before_question() -> None:
    message = build_user_message("Why?", [hit("complaint", "1")])

    assert '<source id="C-1">' in message
    assert message.index("<sources>") < message.index("Question: Why?")


def test_generate_answer_maps_citations_and_flags_unknown_ids() -> None:
    client = FakeClient(FakeMessages("Brakes failed [C-1]. Also [C-999]."))
    hits = [hit("complaint", "1"), hit("complaint", "2")]

    answer = generate_answer(client, "claude-haiku-5-5", "Brakes?", hits)

    assert [s.citation for s in answer.sources] == ["C-1"]
    assert answer.unknown_citations == ["C-999"]
    request = client.messages.calls[0]
    assert request["model"] == "claude-haiku-5-5"
    assert "Question: Brakes?" in request["messages"][0]["content"]


def test_generate_answer_without_sources_skips_the_llm() -> None:
    client = FakeClient(FakeMessages("should not be called"))

    answer = generate_answer(client, "claude-haiku-5-5", "Anything?", [])

    assert answer.sources == []
    assert client.messages.calls == []
