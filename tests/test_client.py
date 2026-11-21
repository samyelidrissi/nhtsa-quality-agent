from collections.abc import Callable

import httpx
import pytest

from nhtsa_quality_agent.data.client import (
    BASE_URL,
    COMPLAINTS_PATH,
    RECALLS_PATH,
    fetch_complaints,
    fetch_recalls,
)

Handler = Callable[[httpx.Request], httpx.Response]


def fake_client(handler: Handler) -> httpx.Client:
    """HTTP client whose requests are answered by `handler` instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)


def test_fetch_complaints_sends_vehicle_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"count": 1, "results": [{"odiNumber": "1"}]})

    results = fetch_complaints(fake_client(handler), "jeep", "grand cherokee", 2021)

    assert results == [{"odiNumber": "1"}]
    request = seen[0]
    assert request.url.path == COMPLAINTS_PATH
    assert request.url.params["make"] == "jeep"
    assert request.url.params["model"] == "grand cherokee"
    assert request.url.params["modelYear"] == "2021"  # query params are always strings


def test_fetch_recalls_hits_recalls_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == RECALLS_PATH
        # Recalls API capitalises its keys differently; only `results` matters.
        return httpx.Response(200, json={"Count": 1, "results": [{"NHTSACampaignNumber": "X"}]})

    assert fetch_recalls(fake_client(handler), "ram", "1500", 2020) == [
        {"NHTSACampaignNumber": "X"}
    ]


@pytest.mark.parametrize("payload", [{}, {"results": None}, {"message": "No results"}])
def test_missing_results_returns_empty_list(payload: dict[str, object]) -> None:
    client = fake_client(lambda request: httpx.Response(200, json=payload))

    assert fetch_complaints(client, "toyota", "camry", 2020) == []


def test_http_error_is_raised() -> None:
    client = fake_client(lambda request: httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_complaints(client, "toyota", "camry", 2020)
