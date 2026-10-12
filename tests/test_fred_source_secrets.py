"""
Regression tests: a failed FRED request must never expose the API key.

FRED Version 1 endpoints authenticate with an ``api_key`` query parameter.
requests embeds the full request URL, query string included, in the
messages of HTTPError and ConnectionError. These tests build those errors the
way requests does, so the key really is present in the raw provider error, and
prove it does not survive into the exception FredDataSource raises: not in its
message, its __cause__ / __context__ chain, or its formatted traceback.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus

import pytest
import requests

from stock_agent.data.fred_source import FredDataSource

FAKE_API_KEY = "fake-fred-key-for-leak-tests-0123"

SERIES_ID = "GDPC1"


def _prepare(
    url: str,
    params: dict[str, object],
    headers: dict[str, str] | None,
) -> requests.PreparedRequest:
    return requests.Request(
        "GET",
        url,
        params=params,
        headers=headers,
    ).prepare()


class _HttpErrorSession:
    """Return a real requests.Response with an error status for every GET."""

    def __init__(self, status_code: int = 500) -> None:
        self._status_code = status_code
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> requests.Response:
        self.calls.append({"url": url, "params": params, "headers": headers})

        prepared = _prepare(url, params, headers)

        response = requests.Response()
        response.status_code = self._status_code
        response.reason = "Internal Server Error"
        response.url = prepared.url or url
        response.request = prepared

        return response


class _ConnectionErrorSession:
    """Raise ConnectionError with a urllib3-style message naming the URL."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> requests.Response:
        self.calls.append({"url": url, "params": params, "headers": headers})

        prepared = _prepare(url, params, headers)

        raise requests.ConnectionError(
            "HTTPSConnectionPool(host='api.stlouisfed.org', port=443): "
            f"Max retries exceeded with url: {prepared.path_url}",
            request=prepared,
        )


def _fetch_series_observations(source: FredDataSource) -> object:
    return source.get_series_observations(series_id=SERIES_ID)


def _fetch_vintage_dates(source: FredDataSource) -> object:
    return source.get_series_vintage_dates(series_id=SERIES_ID)


def _fetch_release_observations(source: FredDataSource) -> object:
    return source.get_observations(
        release_id=18,
        series_id=SERIES_ID,
        start="2026-01-01",
        end="2026-02-01",
    )


VERSION_1_ENDPOINTS = [
    pytest.param(_fetch_series_observations, id="series-observations"),
    pytest.param(_fetch_vintage_dates, id="vintage-dates"),
]

SESSIONS = [
    pytest.param(_HttpErrorSession, id="http-error"),
    pytest.param(_ConnectionErrorSession, id="connection-error"),
]


def _key_forms() -> tuple[str, ...]:
    return (FAKE_API_KEY, quote_plus(FAKE_API_KEY))


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    pending: list[BaseException | None] = [exc]

    while pending:
        current = pending.pop()

        if current is None or any(current is seen for seen in chain):
            continue

        chain.append(current)
        pending.extend([current.__cause__, current.__context__])

    return chain


def _assert_no_key(text: str) -> None:
    for key_form in _key_forms():
        assert key_form not in text


def _assert_exception_hides_key(exc: BaseException) -> None:
    for linked in _exception_chain(exc):
        _assert_no_key(str(linked))
        _assert_no_key(repr(linked))

    _assert_no_key("".join(traceback.format_exception(exc)))


@pytest.mark.parametrize("session_factory", SESSIONS)
def test_fake_provider_errors_really_contain_the_key(
    session_factory: Callable[[], Any],
) -> None:
    """Guard the guard: the raw errors the fakes produce do carry the key."""

    session = session_factory()

    try:
        response = session.get(
            FredDataSource.SERIES_OBSERVATIONS_URL,
            params={"series_id": SERIES_ID, "api_key": FAKE_API_KEY},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raw_error = exc
    else:
        pytest.fail("The fake session should have failed.")

    assert FAKE_API_KEY in str(raw_error)


@pytest.mark.parametrize("fetch", VERSION_1_ENDPOINTS)
@pytest.mark.parametrize("session_factory", SESSIONS)
def test_version_1_request_failure_never_exposes_api_key(
    fetch: Callable[[FredDataSource], object],
    session_factory: Callable[[], Any],
) -> None:
    session = session_factory()
    source = FredDataSource(api_key=FAKE_API_KEY, session=session)

    with pytest.raises(RuntimeError) as excinfo:
        fetch(source)

    # The key was sent: the request authenticated as before.
    assert session.calls[0]["params"]["api_key"] == FAKE_API_KEY

    _assert_exception_hides_key(excinfo.value)


@pytest.mark.parametrize("fetch", VERSION_1_ENDPOINTS)
@pytest.mark.parametrize("session_factory", SESSIONS)
def test_version_1_traceback_with_locals_never_exposes_api_key(
    fetch: Callable[[FredDataSource], object],
    session_factory: Callable[[], Any],
) -> None:
    """Tools such as pytest --showlocals render every frame's local variables."""

    source = FredDataSource(api_key=FAKE_API_KEY, session=session_factory())

    with pytest.raises(RuntimeError) as excinfo:
        fetch(source)

    rendered = traceback.TracebackException.from_exception(
        excinfo.value,
        capture_locals=True,
    )

    _assert_no_key("".join(rendered.format()))


@pytest.mark.parametrize("fetch", VERSION_1_ENDPOINTS)
def test_version_1_http_failure_keeps_status_and_series_for_diagnosis(
    fetch: Callable[[FredDataSource], object],
) -> None:
    source = FredDataSource(
        api_key=FAKE_API_KEY,
        session=_HttpErrorSession(status_code=503),
    )

    with pytest.raises(RuntimeError) as excinfo:
        fetch(source)

    message = str(excinfo.value)

    assert "HTTP 503" in message
    assert SERIES_ID in message


@pytest.mark.parametrize("fetch", VERSION_1_ENDPOINTS)
def test_version_1_connection_failure_names_the_error_type(
    fetch: Callable[[FredDataSource], object],
) -> None:
    source = FredDataSource(
        api_key=FAKE_API_KEY,
        session=_ConnectionErrorSession(),
    )

    with pytest.raises(RuntimeError) as excinfo:
        fetch(source)

    message = str(excinfo.value)

    assert "ConnectionError" in message
    assert SERIES_ID in message


@pytest.mark.parametrize("session_factory", SESSIONS)
def test_version_2_request_failure_never_exposes_api_key(
    session_factory: Callable[[], Any],
) -> None:
    """The release endpoint sends the key as a bearer header, not in the URL."""

    session = session_factory()
    source = FredDataSource(api_key=FAKE_API_KEY, session=session)

    with pytest.raises(TypeError) as excinfo:
        _fetch_release_observations(source)

    assert "api_key" not in session.calls[0]["params"]

    _assert_exception_hides_key(excinfo.value)
