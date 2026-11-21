"""Turn clean tables into retrievable documents (text + filterable metadata).

One complaint = one document: texts are short (median ~90 words), so no chunking
is needed. One recall = one document per covered vehicle, so that retrieval can be
filtered on make/model/year; duplicates of a campaign are merged at query time.
"""

from dataclasses import dataclass

import pandas as pd

Metadata = dict[str, str | int | bool]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Metadata


def complaint_id(odi_number: str) -> str:
    return f"C-{odi_number}"


def recall_id(campaign_number: str, model: str, model_year: int) -> str:
    return f"R-{campaign_number}-{model.replace(' ', '-')}-{model_year}"


def complaint_documents(complaints: pd.DataFrame) -> list[Document]:
    documents = []
    for row in complaints.itertuples(index=False):
        components = list(row.components)
        header = f"{row.make.title()} {row.model.title()} {row.model_year} complaint"
        text = f"{header}. Components: {', '.join(components)}.\n{row.summary}"
        documents.append(
            Document(
                id=complaint_id(row.odi_number),
                text=text,
                metadata={
                    "source": "complaint",
                    "source_id": row.odi_number,
                    "make": row.make,
                    "model": row.model,
                    "model_year": int(row.model_year),
                    # Chroma metadata must be scalar: store labels as one delimited string.
                    "components": "|".join(components),
                    "crash": bool(row.crash),
                    "fire": bool(row.fire),
                    "date": row.date_filed.date().isoformat(),
                },
            )
        )
    return documents


def recall_documents(recalls: pd.DataFrame, recall_vehicles: pd.DataFrame) -> list[Document]:
    linked = recall_vehicles.merge(recalls, on="campaign_number", how="inner")
    documents = []
    for row in linked.itertuples(index=False):
        text = (
            f"Recall {row.campaign_number} ({row.make.title()} {row.model.title()} "
            f"{row.model_year}). Component: {row.component}.\n"
            f"Summary: {row.summary}\nConsequence: {row.consequence}\nRemedy: {row.remedy}"
        )
        documents.append(
            Document(
                id=recall_id(row.campaign_number, row.model, int(row.model_year)),
                text=text,
                metadata={
                    "source": "recall",
                    "source_id": row.campaign_number,
                    "make": row.make,
                    "model": row.model,
                    "model_year": int(row.model_year),
                    "components": row.component,
                    "park_it": bool(row.park_it),
                    "date": row.report_date.date().isoformat(),
                },
            )
        )
    return documents
