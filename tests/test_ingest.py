import json
from pathlib import Path

import httpx
import pytest

from nhtsa_quality_agent.data.client import BASE_URL, COMPLAINTS_PATH
from nhtsa_quality_agent.data.ingest import (
    DEFAULT_SCOPE,
    Vehicle,
    ingest,
    ingest_one,
    raw_path,
    with_retries,
)

CAMRY = Vehicle("toyota", "camry", 2020)


def no_sleep(_: float) -> None:
    """Replaces time.sleep so retry tests run instantly."""


def fake_client(statuses: list[int], records: list[dict[str, str]] | None = None) -> httpx.Client:
    """Client answering successive requests with the given status codes (last one repeats)."""
    calls = iter(statuses)
    last = statuses[-1]

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(calls, last)
        body = {"results": records or []} if status == 200 else {"message": "error"}
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)


def test_vehicle_slug_is_filesystem_safe() -> None:
    assert Vehicle("Jeep", "Grand Cherokee", 2021).slug == "jeep_grand-cherokee_2021"


def test_default_scope_covers_each_model_year_on_both_endpoints() -> None:
    downloads = [(kind, v.slug) for v in DEFAULT_SCOPE for kind in v.kinds]
    covered = {
        (kind, v.make, v.model.split()[0], v.year) for v in DEFAULT_SCOPE for kind in v.kinds
    }

    assert len(downloads) == len(set(downloads))  # no file downloaded twice
    assert len(covered) == 6 * 5 * 2  # 6 models x 5 years x 2 endpoints


def test_ingest_only_queries_the_vehicle_kinds(tmp_path: Path) -> None:
    recalls_only = Vehicle("ram", "1500", 2022, kinds=("recalls",))

    ingest(fake_client([200]), [recalls_only], tmp_path, pause=0, sleep=no_sleep)

    assert raw_path(tmp_path, "recalls", recalls_only).exists()
    assert not raw_path(tmp_path, "complaints", recalls_only).exists()


def test_ingest_warns_about_empty_results(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ingest(fake_client([200]), [CAMRY], tmp_path, pause=0, sleep=no_sleep)

    assert "complaints/toyota_camry_2020" in caplog.text


def test_ingest_one_saves_raw_payload_with_metadata(tmp_path: Path) -> None:
    client = fake_client([200], records=[{"odiNumber": "1"}, {"odiNumber": "2"}])

    count = ingest_one(client, "complaints", CAMRY, tmp_path, sleep=no_sleep)

    assert count == 2
    saved = json.loads(raw_path(tmp_path, "complaints", CAMRY).read_text())
    assert saved["results"] == [{"odiNumber": "1"}, {"odiNumber": "2"}]
    assert saved["vehicle"] == {"make": "toyota", "model": "camry", "year": 2020}
    assert saved["source"] == "complaints"
    assert "fetched_at" in saved


def test_ingest_one_skips_existing_file(tmp_path: Path) -> None:
    path = raw_path(tmp_path, "complaints", CAMRY)
    path.parent.mkdir(parents=True)
    path.write_text("{}")

    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call the API")

    client = httpx.Client(transport=httpx.MockTransport(fail), base_url=BASE_URL)

    assert ingest_one(client, "complaints", CAMRY, tmp_path) is None
    assert path.read_text() == "{}"


def test_ingest_one_saves_empty_result_on_400(tmp_path: Path) -> None:
    count = ingest_one(fake_client([400]), "recalls", CAMRY, tmp_path, sleep=no_sleep)

    assert count == 0
    saved = json.loads(raw_path(tmp_path, "recalls", CAMRY).read_text())
    assert saved["results"] == []


def test_ingest_one_does_not_write_file_on_persistent_server_error(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        ingest_one(fake_client([503]), "complaints", CAMRY, tmp_path, sleep=no_sleep)

    assert not raw_path(tmp_path, "complaints", CAMRY).exists()


def test_with_retries_recovers_from_transient_errors() -> None:
    delays: list[float] = []
    client = fake_client([503, 502, 200], records=[{"odiNumber": "1"}])

    def call() -> httpx.Response:
        response = client.get(COMPLAINTS_PATH)
        response.raise_for_status()
        return response

    response = with_retries(call, attempts=3, base_delay=1.0, sleep=delays.append)

    assert response.status_code == 200
    assert delays == [1.0, 2.0]  # exponential backoff


def test_with_retries_does_not_retry_client_errors() -> None:
    delays: list[float] = []
    client = fake_client([404, 200])

    def call() -> httpx.Response:
        response = client.get(COMPLAINTS_PATH)
        response.raise_for_status()
        return response

    with pytest.raises(httpx.HTTPStatusError):
        with_retries(call, sleep=delays.append)
    assert delays == []


def test_ingest_counts_records_per_kind(tmp_path: Path) -> None:
    client = fake_client([200], records=[{"id": "a"}])
    vehicles = [CAMRY, Vehicle("ram", "1500", 2021)]

    totals = ingest(client, vehicles, tmp_path, pause=0, sleep=no_sleep)

    assert totals == {"complaints": 2, "recalls": 2}
    assert len(list(tmp_path.rglob("*.json"))) == 4
