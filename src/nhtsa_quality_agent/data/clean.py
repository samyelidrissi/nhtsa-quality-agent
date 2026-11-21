"""Turn raw NHTSA JSON files into clean, typed, deduplicated Parquet tables.

Outputs (in `<data_dir>/processed/`):
- complaints.parquet       one row per complaint (`odi_number`)
- recalls.parquet          one row per recall campaign (`campaign_number`)
- recall_vehicles.parquet  one row per (campaign, make, model, year): a campaign
                           covers several vehicles, so this is a 1-N link table.

Run with:  uv run python -m nhtsa_quality_agent.data.clean
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from nhtsa_quality_agent.config import DEFAULT_DATA_DIR
from nhtsa_quality_agent.data.ingest import Kind

logger = logging.getLogger(__name__)

# NHTSA joins labels with "," but some labels contain ", " themselves
# (e.g. "FUEL SYSTEM, GASOLINE,ENGINE" -> ["FUEL SYSTEM, GASOLINE", "ENGINE"]).
# So we only split on a comma that is NOT followed by a space.
_LABEL_SEPARATOR = re.compile(r",(?=\S)")
# From 2022 NHTSA lists the Ram 1500 per cab variant; we analyse it as one model.
_CAB_SUFFIX = re.compile(r"\s+(crew|quad|regular|mega)\s+cab$", re.IGNORECASE)

# Date formats differ between endpoints: US for complaints, European for recalls.
COMPLAINT_DATE_FORMAT = "%m/%d/%Y"
RECALL_DATE_FORMAT = "%d/%m/%Y"

QUERY_COLUMNS = {"query_make": "make", "query_model": "model", "query_year": "model_year"}


def split_components(raw: str) -> list[str]:
    """Split NHTSA's multi-label `components` string into individual labels."""
    return [label.strip() for label in _LABEL_SEPARATOR.split(raw) if label.strip()]


def canonical_model(model: str) -> str:
    """Map cab variants to their model family, e.g. "1500 crew cab" -> "1500"."""
    return _CAB_SUFFIX.sub("", model.strip())


def load_raw(raw_dir: Path, kind: Kind) -> list[dict[str, Any]]:
    """Load every raw file of `kind`, tagging each record with the vehicle it was queried for."""
    records: list[dict[str, Any]] = []
    for path in sorted((raw_dir / kind).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        vehicle = payload["vehicle"]
        query = {
            "query_make": vehicle["make"],
            "query_model": vehicle["model"],
            "query_year": vehicle["year"],
        }
        records += [{**record, **query} for record in payload["results"]]
    return records


_BOOL_VALUES = {True: True, False: False, "True": True, "False": False}


def _as_bool(df: pd.DataFrame, column: str) -> pd.Series:
    """Strict bool conversion: missing or unknown values become False.

    Never use `.astype(bool)` here: it turns NaN (and the string "False") into True.
    """
    if column not in df:
        return pd.Series(False, index=df.index)
    return df[column].map(_BOOL_VALUES).fillna(False).astype(bool)


def _vehicle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename query columns and normalise them (lower-case make, canonical model)."""
    df = df.rename(columns=QUERY_COLUMNS)
    df["make"] = df["make"].str.lower()
    df["model"] = df["model"].str.lower().map(canonical_model)
    df["model_year"] = df["model_year"].astype(int)
    return df


def clean_complaints(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One clean row per complaint, deduplicated on `odi_number`."""
    df = _vehicle_columns(pd.DataFrame(records))
    df = df.rename(
        columns={
            "odiNumber": "odi_number",
            "dateComplaintFiled": "date_filed",
            "dateOfIncident": "date_incident",
            "numberOfInjuries": "injuries",
            "numberOfDeaths": "deaths",
        }
    )
    df["odi_number"] = df["odi_number"].astype(str)
    for column in ("date_filed", "date_incident"):
        # errors="raise": an unexpected format must stop the pipeline, not become NaT.
        df[column] = pd.to_datetime(df[column], format=COMPLAINT_DATE_FORMAT)
    df["crash"] = _as_bool(df, "crash")
    df["fire"] = _as_bool(df, "fire")
    df["injuries"] = df["injuries"].astype(int)
    df["deaths"] = df["deaths"].astype(int)
    df["components"] = df["components"].map(split_components)
    df["summary"] = df["summary"].fillna("").str.strip()
    df["vin"] = df["vin"].replace("", None)

    df = df[df["summary"] != ""]
    df = df.drop_duplicates(subset="odi_number")
    columns = [
        "odi_number",
        "make",
        "model",
        "model_year",
        "date_filed",
        "date_incident",
        "crash",
        "fire",
        "injuries",
        "deaths",
        "components",
        "summary",
        "vin",
    ]
    return df[columns].sort_values("odi_number").reset_index(drop=True)


def clean_recalls(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (recalls: one row per campaign, recall_vehicles: campaign-vehicle links)."""
    df = _vehicle_columns(pd.DataFrame(records))
    df = df.rename(
        columns={
            "NHTSACampaignNumber": "campaign_number",
            "ReportReceivedDate": "report_date",
            "Component": "component",
            "Summary": "summary",
            "Consequence": "consequence",
            "Remedy": "remedy",
            "parkIt": "park_it",
            "parkOutSide": "park_outside",
            "overTheAirUpdate": "over_the_air",
        }
    )
    df["report_date"] = pd.to_datetime(df["report_date"], format=RECALL_DATE_FORMAT)
    for column in ("summary", "consequence", "remedy", "component"):
        df[column] = df[column].fillna("").str.strip()
    for column in ("park_it", "park_outside", "over_the_air"):
        df[column] = _as_bool(df, column)

    recall_columns = [
        "campaign_number",
        "report_date",
        "component",
        "summary",
        "consequence",
        "remedy",
        "park_it",
        "park_outside",
        "over_the_air",
    ]
    recalls = (
        df[recall_columns]
        .drop_duplicates(subset="campaign_number")
        .sort_values("campaign_number")
        .reset_index(drop=True)
    )
    links = (
        df[["campaign_number", "make", "model", "model_year"]]
        .drop_duplicates()
        .sort_values(["campaign_number", "make", "model", "model_year"])
        .reset_index(drop=True)
    )
    return recalls, links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_dir, out_dir = args.data_dir / "raw", args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    complaints = clean_complaints(load_raw(raw_dir, "complaints"))
    recalls, recall_vehicles = clean_recalls(load_raw(raw_dir, "recalls"))

    tables = {
        "complaints": complaints,
        "recalls": recalls,
        "recall_vehicles": recall_vehicles,
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
        logger.info("wrote %-16s %6d rows", name, len(table))


if __name__ == "__main__":
    main()
