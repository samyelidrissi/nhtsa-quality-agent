import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nhtsa_quality_agent.data.clean import (
    canonical_model,
    clean_complaints,
    clean_recalls,
    load_raw,
    split_components,
)


def complaint(odi: str = "1", model: str = "camry", **overrides: Any) -> dict[str, Any]:
    """A raw complaint record, as returned by the API and tagged by load_raw."""
    record = {
        "odiNumber": odi,
        "manufacturer": "Toyota Motor Corporation",
        "crash": False,
        "fire": False,
        "numberOfInjuries": 0,
        "numberOfDeaths": 0,
        "dateOfIncident": "10/30/2021",
        "dateComplaintFiled": "11/04/2021",
        "vin": "4T1G11AKXLU",
        "components": "ENGINE",
        "summary": "  Engine stalled on the highway.  ",
        "products": [],
        "query_make": "toyota",
        "query_model": model,
        "query_year": 2021,
    }
    return record | overrides


def recall(campaign: str = "21V001000", model: str = "camry", **overrides: Any) -> dict[str, Any]:
    record = {
        "NHTSACampaignNumber": campaign,
        "ReportReceivedDate": "04/11/2021",
        "Component": "FUEL SYSTEM, GASOLINE:DELIVERY:FUEL PUMP",
        "Summary": "Fuel pump may fail.",
        "Consequence": "Engine stall.",
        "Remedy": "Replace fuel pump.",
        "parkIt": False,
        "parkOutSide": False,
        "overTheAirUpdate": True,
        "query_make": "toyota",
        "query_model": model,
        "query_year": 2021,
    }
    return record | overrides


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ENGINE", ["ENGINE"]),
        ("ENGINE,POWER TRAIN", ["ENGINE", "POWER TRAIN"]),
        ("FUEL SYSTEM, GASOLINE,ENGINE", ["FUEL SYSTEM, GASOLINE", "ENGINE"]),
        ("STEERING,SERVICE BRAKES, HYDRAULIC", ["STEERING", "SERVICE BRAKES, HYDRAULIC"]),
        ("", []),
    ],
)
def test_split_components(raw: str, expected: list[str]) -> None:
    assert split_components(raw) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [("1500 crew cab", "1500"), ("1500 QUAD CAB", "1500"), ("camry", "camry"), ("1500", "1500")],
)
def test_canonical_model(model: str, expected: str) -> None:
    assert canonical_model(model) == expected


def test_load_raw_tags_records_with_queried_vehicle(tmp_path: Path) -> None:
    (tmp_path / "complaints").mkdir()
    payload = {
        "vehicle": {"make": "ram", "model": "1500 crew cab", "year": 2022},
        "results": [{"odiNumber": "1"}, {"odiNumber": "2"}],
    }
    (tmp_path / "complaints" / "ram_1500-crew-cab_2022.json").write_text(json.dumps(payload))

    records = load_raw(tmp_path, "complaints")

    assert len(records) == 2
    assert records[0]["query_model"] == "1500 crew cab"
    assert records[0]["query_year"] == 2022


def test_clean_complaints_types_and_values() -> None:
    row = clean_complaints([complaint(components="FUEL SYSTEM, GASOLINE,ENGINE")]).iloc[0]

    assert row["make"] == "toyota"
    assert row["model_year"] == 2021
    assert row["components"] == ["FUEL SYSTEM, GASOLINE", "ENGINE"]
    assert row["summary"] == "Engine stalled on the highway."


def test_clean_complaints_reads_us_dates() -> None:
    row = clean_complaints([complaint(dateComplaintFiled="11/04/2021")]).iloc[0]

    assert row["date_filed"] == pd.Timestamp("2021-11-04")  # 4 November, not 11 April


def test_clean_complaints_dedupes_cab_variants() -> None:
    records = [
        complaint(odi="42", model="1500 crew cab", query_make="ram"),
        complaint(odi="42", model="1500 quad cab", query_make="ram"),
    ]

    df = clean_complaints(records)

    assert len(df) == 1
    assert df.iloc[0]["model"] == "1500"


def test_clean_complaints_drops_empty_summaries() -> None:
    df = clean_complaints([complaint(odi="1"), complaint(odi="2", summary="   ")])

    assert df["odi_number"].tolist() == ["1"]


def test_clean_complaints_rejects_unexpected_date_format() -> None:
    with pytest.raises(ValueError):
        clean_complaints([complaint(dateComplaintFiled="2021-11-04")])


def test_clean_recalls_reads_european_dates() -> None:
    recalls, _ = clean_recalls([recall(ReportReceivedDate="04/11/2021")])

    assert recalls.iloc[0]["report_date"] == pd.Timestamp("2021-11-04")  # 4 November


def test_clean_recalls_splits_campaigns_and_vehicle_links() -> None:
    records = [recall(model="camry"), recall(model="camry hybrid"), recall(model="camry")]

    recalls, links = clean_recalls(records)

    assert len(recalls) == 1
    assert links["model"].tolist() == ["camry", "camry hybrid"]


def test_clean_recalls_missing_flags_are_false_not_true() -> None:
    record = recall()
    del record["parkIt"]

    recalls, _ = clean_recalls([record])

    assert not recalls.iloc[0]["park_it"]  # NaN.astype(bool) would give True
    assert recalls.iloc[0]["over_the_air"]
