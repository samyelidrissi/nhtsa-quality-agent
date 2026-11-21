"""Thin client for the public NHTSA complaints and recalls APIs.

The HTTP client is injected by the caller so that connection settings (timeout,
retries) live in one place and tests can swap in a fake transport.
"""

from typing import Any

import httpx

BASE_URL = "https://api.nhtsa.gov"
COMPLAINTS_PATH = "/complaints/complaintsByVehicle"
RECALLS_PATH = "/recalls/recallsByVehicle"
DEFAULT_TIMEOUT_SECONDS = 30.0

Record = dict[str, Any]


def make_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> httpx.Client:
    """Build an HTTP client preconfigured for the NHTSA API."""
    return httpx.Client(base_url=BASE_URL, timeout=timeout)


def _vehicle_params(make: str, model: str, year: int) -> dict[str, str | int]:
    return {"make": make, "model": model, "modelYear": year}


def _get_results(client: httpx.Client, path: str, params: dict[str, str | int]) -> list[Record]:
    """GET an NHTSA endpoint and return its `results` list (empty if absent)."""
    response = client.get(path, params=params)
    response.raise_for_status()
    results = response.json().get("results")
    return results if isinstance(results, list) else []


def fetch_complaints(client: httpx.Client, make: str, model: str, year: int) -> list[Record]:
    """Return raw complaint records for one make/model/year."""
    return _get_results(client, COMPLAINTS_PATH, _vehicle_params(make, model, year))


def fetch_recalls(client: httpx.Client, make: str, model: str, year: int) -> list[Record]:
    """Return raw recall records for one make/model/year."""
    return _get_results(client, RECALLS_PATH, _vehicle_params(make, model, year))
