"""Download raw NHTSA complaints and recalls for a fixed vehicle scope.

Raw API responses are saved untouched (plus provenance metadata) under
`<raw_dir>/<kind>/<make>_<model>_<year>.json`. Cleaning happens later, from
these files, so a cleaning bug never requires calling the API again.

Run with:  uv run python -m nhtsa_quality_agent.data.ingest
"""

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx

from nhtsa_quality_agent.config import DEFAULT_DATA_DIR
from nhtsa_quality_agent.data.client import Record, fetch_complaints, fetch_recalls, make_client

logger = logging.getLogger(__name__)

Kind = Literal["complaints", "recalls"]
FETCHERS: dict[Kind, Callable[[httpx.Client, str, str, int], list[Record]]] = {
    "complaints": fetch_complaints,
    "recalls": fetch_recalls,
}


ALL_KINDS: tuple[Kind, ...] = ("complaints", "recalls")


@dataclass(frozen=True)
class Vehicle:
    """One NHTSA product, queried on the endpoints listed in `kinds`."""

    make: str
    model: str
    year: int
    kinds: tuple[Kind, ...] = ALL_KINDS

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, e.g. `jeep_grand-cherokee_2021`."""
        model = self.model.lower().replace(" ", "-")
        return f"{self.make.lower()}_{model}_{self.year}"


YEARS = range(2019, 2024)
# Model names must match NHTSA's own naming exactly, which is neither stable over
# time nor shared between endpoints. From 2022 the Ram 1500 complaints are only
# listed per cab variant (the same complaint may appear under several variants, so
# cleaning must deduplicate on `odiNumber`), while its recalls stay under "1500".
# The Ford F-150 was left out: its naming is too fragmented across years.
STABLE_MODELS = [
    ("toyota", "camry"),
    ("honda", "cr-v"),
    ("ford", "explorer"),
    ("tesla", "model 3"),
    ("jeep", "grand cherokee"),
]
RAM_SPLIT_YEARS = (2022, 2023)
RAM_CAB_VARIANTS = ("1500 crew cab", "1500 quad cab")


def _ram_1500_scope() -> list[Vehicle]:
    vehicles = [Vehicle("ram", "1500", year) for year in YEARS if year not in RAM_SPLIT_YEARS]
    for year in RAM_SPLIT_YEARS:
        vehicles.append(Vehicle("ram", "1500", year, kinds=("recalls",)))
        vehicles += [Vehicle("ram", cab, year, kinds=("complaints",)) for cab in RAM_CAB_VARIANTS]
    return vehicles


DEFAULT_SCOPE = [
    Vehicle(make, model, year) for make, model in STABLE_MODELS for year in YEARS
] + _ram_1500_scope()


def raw_path(raw_dir: Path, kind: Kind, vehicle: Vehicle) -> Path:
    return raw_dir / kind / f"{vehicle.slug}.json"


def _is_retryable(error: Exception) -> bool:
    """Network failures and server-side (5xx) errors are worth retrying; 4xx are not."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)


def with_retries[T](
    call: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying transient errors with exponential backoff (1s, 2s, 4s...)."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            if not _is_retryable(error) or attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("Attempt %d failed (%s), retrying in %.1fs", attempt, error, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def ingest_one(
    client: httpx.Client,
    kind: Kind,
    vehicle: Vehicle,
    raw_dir: Path,
    *,
    overwrite: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Fetch and save one (kind, vehicle). Returns the record count, or None if skipped."""
    path = raw_path(raw_dir, kind, vehicle)
    if path.exists() and not overwrite:
        return None

    fetch = FETCHERS[kind]
    try:
        records = with_retries(
            lambda: fetch(client, vehicle.make, vehicle.model, vehicle.year), sleep=sleep
        )
    except httpx.HTTPStatusError as error:
        # NHTSA answers 400 (not an empty list) when it knows no such vehicle.
        if error.response.status_code != 400:
            raise
        logger.warning("No %s data for %s (HTTP 400), saving empty result", kind, vehicle.slug)
        records = []

    _write_json_atomic(
        path,
        {
            "source": kind,
            "vehicle": {"make": vehicle.make, "model": vehicle.model, "year": vehicle.year},
            "fetched_at": datetime.now(UTC).isoformat(),
            "results": records,
        },
    )
    return len(records)


def ingest(
    client: httpx.Client,
    vehicles: list[Vehicle],
    raw_dir: Path,
    *,
    overwrite: bool = False,
    pause: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[Kind, int]:
    """Ingest every (kind, vehicle) pair; returns the number of new records per kind."""
    totals: dict[Kind, int] = {"complaints": 0, "recalls": 0}
    empty: list[str] = []
    for vehicle in vehicles:
        for kind in vehicle.kinds:
            count = ingest_one(client, kind, vehicle, raw_dir, overwrite=overwrite, sleep=sleep)
            if count is None:
                logger.info("skip  %-9s %s (already downloaded)", kind, vehicle.slug)
                continue
            logger.info("saved %-9s %s: %d records", kind, vehicle.slug, count)
            totals[kind] += count
            if count == 0:
                empty.append(f"{kind}/{vehicle.slug}")
            sleep(pause)  # be polite to a free public API
    if empty:
        # An empty result is often a naming mismatch, not a defect-free vehicle.
        logger.warning("Empty results, check model naming: %s", ", ".join(empty))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DATA_DIR / "raw")
    parser.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    with make_client() as client:
        totals = ingest(client, DEFAULT_SCOPE, args.raw_dir, overwrite=args.overwrite)
    logger.info("Done. New records: %s", totals)


if __name__ == "__main__":
    main()
