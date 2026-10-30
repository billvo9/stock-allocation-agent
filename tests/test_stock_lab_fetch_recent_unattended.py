"""
The evening agent's price update after a wake (scripts/stock_lab_fetch_recent.py
--unattended, and the network check in stock_agent.stock_lab.tiingo).

On 2026-10-29 the Mac woke at 23:52 and launchd started the missed evening at
once: the first Tiingo request failed (transport_error; URLError) before Wi-Fi
had reconnected, although the evening could be recorded until 09:30. With
--unattended the run first waits a bounded time for the network (a DNS lookup
and a TCP connection, never a request or the key), retries a transport error
a bounded number of times, and never waits past the next open. A run on
battery warns and notifies, since caffeinate -s holds the Mac awake only on
AC power.

Injected clocks, a fake network, a fake transport, a fake notifier and fake
stores; no network, nothing shown.
"""

from __future__ import annotations

import io
import socket
import urllib.error

import pandas as pd
import pytest
import test_stock_lab_fetch_recent_script as base
from stock_lab_inference_helpers import Clock, ny
from test_stock_lab_tiingo import FakeTransport

from stock_agent.stock_lab import notify, tiingo
from stock_agent.stock_lab.inference.schema import DB_NAME
from stock_agent.stock_lab.inference.store import InferenceStore
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers
from stock_agent.stock_lab.tiingo import StockAdapter

fetch = base.fetch
WATCHED = base.WATCHED
KEY = "keychain-fixture-key-0789"
FLAGS = ["--authorized", "--store", "--keychain", "--unattended"]
# The evening of Friday 2026-10-16: the next open is Monday 2026-10-19 at 09:30 New York.
OPEN = ny("2026-10-19", "09:30")


@pytest.fixture
def research(tmp_path):
    store = StockLabStore(tmp_path / "research" / RESEARCH_DB_NAME, clock=lambda: base.NOW)
    store.add(parse_tickers(" ".join(WATCHED)))
    base.supported(store)
    return store


class Network:
    """A network that comes up at `up` (never when None); counts its checks."""

    def __init__(self, clock, events, up=None):
        self.clock, self.events, self.up, self.checks = clock, events, up, 0

    def __call__(self):
        self.checks += 1
        self.events.append("check")
        if self.up is not None and self.clock.now >= self.up:
            return None
        return "dns: gaierror"


class Evening:
    """One run of the script: its clock, network, sleeps, key reads, requests and notices."""

    def __init__(self, research, tmp_path, now=base.NOW, up=base.NOW, power="ac"):
        self.clock = Clock(now)
        self.events = []
        self.network = Network(self.clock, self.events, up)
        self.sleeps = []
        self.notices = []
        self.budgets = []
        self.power = power
        self.research = research
        self.inference_path = tmp_path / "inference" / DB_NAME

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock.now += pd.Timedelta(seconds=seconds)

    def key(self):
        self.events.append("key")
        return KEY

    def run(self, argv, responses=()):
        transport = FakeTransport(list(responses))
        events = self.events

        def sent(url, headers):
            events.append("request")
            return transport(url, headers)

        def factory(**kwargs):
            self.budgets.append(kwargs["max_requests"])
            return StockAdapter(transport=sent, sleep=lambda s: None, clock=self.clock, **kwargs)

        out = io.StringIO()
        code = fetch.main(
            argv,
            adapter_factory=factory,
            store_factory=lambda: self.research,
            inference_factory=lambda: InferenceStore(self.inference_path, clock=self.clock),
            clock=self.clock,
            out=out,
            key_reader=self.key,
            network=self.network,
            sleep=self.sleep,
            power=lambda: self.power,
            notifier=self.notices.append,
        )
        self.transport = transport
        return code, out.getvalue()

    def stored(self) -> list[int]:
        if not self.inference_path.exists():
            return [0] * len(WATCHED)
        return [len(InferenceStore(self.inference_path).batches(t)) for t in WATCHED]


def down(detail="URLError"):
    return urllib.error.URLError(detail)


# --- the network check: no request, no key ---------------------------------------------------


class Connection:
    def __init__(self):
        self.closed, self.sent = False, []

    def close(self):
        self.closed = True

    def sendall(self, data):  # never called: the check sends nothing
        self.sent.append(data)


def test_the_check_resolves_the_price_host_and_opens_and_closes_one_tcp_connection():
    resolved, connected, connection = [], [], Connection()

    def resolve(host, port, **kwargs):
        resolved.append((host, port, kwargs))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))]

    def connect(address, **kwargs):
        connected.append((address, kwargs))
        return connection

    assert tiingo.reach(resolve=resolve, connect=connect) is None
    assert tiingo.PRICES_HOST == "api.tiingo.com"  # the host the prices are requested from
    assert resolved == [("api.tiingo.com", 443, {"type": socket.SOCK_STREAM})]
    assert connected == [(("api.tiingo.com", 443), {"timeout": tiingo.PROBE_TIMEOUT})]
    assert connection.closed and connection.sent == []  # no TLS, no HTTP, no key, no header


@pytest.mark.parametrize(
    "resolve_error, connect_error, reason",
    [
        (socket.gaierror(8, "nodename nor servname provided"), None, "dns: gaierror"),
        (None, TimeoutError("timed out"), "connect: TimeoutError"),
        (None, ConnectionRefusedError(61, "refused"), "connect: ConnectionRefusedError"),
        (None, OSError(51, "Network is unreachable"), "connect: OSError"),
    ],
)
def test_an_unreachable_network_is_named_by_the_failures_type_only(
    resolve_error, connect_error, reason
):
    def resolve(host, port, **kwargs):
        if resolve_error is not None:
            raise resolve_error
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))]

    def connect(address, **kwargs):
        if connect_error is not None:
            raise connect_error
        return Connection()

    assert tiingo.reach(resolve=resolve, connect=connect) == reason


def test_the_wait_backs_off_and_stops_at_its_bound_or_its_deadline():
    clock = Clock(base.NOW)
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock.now += pd.Timedelta(seconds=seconds)

    never = tiingo.wait_for_network(
        lambda: "dns: gaierror", clock=clock, sleep=sleep, until=OPEN, seconds=300.0
    )
    assert not never.reachable and never.reason == "dns: gaierror"
    assert slept == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0]  # a fourth 60 s would pass 300 s
    assert never.checks == len(slept) + 1 and never.waited == sum(slept)
    # A deadline before the bound: no sleep may end at or after it.
    clock.now, slept[:] = OPEN - pd.Timedelta(seconds=30), []
    late = tiingo.wait_for_network(lambda: "dns: gaierror", clock=clock, sleep=sleep, until=OPEN)
    assert slept == [5.0, 10.0] and clock.now < OPEN and late.checks == 3
    # A sleep that would end exactly at the deadline is not started either.
    clock.now, slept[:] = OPEN - pd.Timedelta(seconds=15), []
    tiingo.wait_for_network(lambda: "dns: gaierror", clock=clock, sleep=sleep, until=OPEN)
    assert slept == [5.0] and clock.now == OPEN - pd.Timedelta(seconds=10)
    # A clock that never moves still ends: the bound counts the sleeps themselves.
    stuck = tiingo.wait_for_network(
        lambda: "dns: gaierror", clock=lambda: base.NOW, sleep=lambda s: None, until=OPEN
    )
    assert not stuck.reachable and stuck.waited <= tiingo.NETWORK_WAIT


# --- the unattended price update ---------------------------------------------------------------


def test_a_network_that_comes_up_after_two_minutes_is_waited_for_then_used(research, tmp_path):
    evening = Evening(research, tmp_path, up=base.NOW + pd.Timedelta(minutes=2))
    code, text = evening.run(FLAGS, [base.prices() for _ in WATCHED])
    assert code == 0, text
    assert evening.sleeps == [5.0, 10.0, 20.0, 40.0, 60.0]  # 135 s: the check at 2:15 passes
    assert "network: api.tiingo.com reachable after 135 s (6 checks; first dns: gaierror)" in text
    # Every check before the key is read, and the key before the first request; no extra request.
    assert evening.events == ["check"] * 6 + ["key"] + ["request"] * len(WATCHED)
    assert evening.stored() == [1, 1, 1, 1] and KEY not in text


def test_a_network_already_up_is_checked_once_and_nothing_waits(research, tmp_path):
    evening = Evening(research, tmp_path)
    code, text = evening.run(FLAGS, [base.prices() for _ in WATCHED])
    assert code == 0, text
    assert evening.network.checks == 1 and evening.sleeps == [] and "network:" not in text
    assert evening.events[:2] == ["check", "key"] and evening.stored() == [1, 1, 1, 1]


def test_a_network_that_stays_down_past_the_wait_stops_before_any_request(research, tmp_path):
    before = research.path.read_bytes()
    evening = Evening(research, tmp_path, up=None)
    code, text = evening.run(FLAGS, [base.prices() for _ in WATCHED])
    assert code == fetch.NO_NETWORK == 4
    assert sum(evening.sleeps) <= 300 and evening.clock.now - base.NOW <= pd.Timedelta(minutes=5)
    assert evening.events == ["check"] * evening.network.checks  # no key read, no request
    assert evening.budgets == [] and evening.transport.calls == []
    assert (
        "stopped: api.tiingo.com was not reachable after 255 s of waiting (8 checks; last dns: "
        "gaierror); nothing was requested"
    ) in text
    assert not evening.inference_path.exists() and research.path.read_bytes() == before


def test_the_wait_never_runs_past_the_next_open(research, tmp_path):
    # A Mac that woke at 09:29 on the Monday: the Friday evening closes at the 09:30 open.
    evening = Evening(research, tmp_path, now=ny("2026-10-19", "09:29"), up=None)
    code, _ = evening.run(FLAGS)
    assert code == fetch.NO_NETWORK and evening.clock.now < OPEN
    assert evening.transport.calls == []
    # After the open nothing waits at all: one check, then the stop.
    evening = Evening(research, tmp_path / "late", now=ny("2026-10-19", "10:00"), up=None)
    code, text = evening.run(FLAGS)
    assert code == fetch.NO_NETWORK and evening.sleeps == [] and evening.network.checks == 1
    assert "not reachable after 0 s of waiting (1 check; last dns: gaierror)" in text


def test_a_transport_error_is_retried_with_backoff_and_counted_in_the_budget(research, tmp_path):
    evening = Evening(research, tmp_path)
    responses = [down(), down(), base.prices(), base.prices(), base.prices(), base.prices()]
    code, text = evening.run(FLAGS, responses)
    assert code == 0, text
    assert evening.sleeps == [10.0, 30.0]  # the backoff, before each retry
    assert len(evening.transport.calls) == len(WATCHED) + 2
    assert evening.budgets == [len(WATCHED) + fetch.TRANSPORT_RETRIES]
    assert "AMD: transport_error; ticker AMD; URLError; retry 1 of 3 in 10 s" in text
    assert evening.stored() == [1, 1, 1, 1]


def test_transport_errors_beyond_the_retries_stop_the_run(research, tmp_path):
    evening = Evening(research, tmp_path)
    code, text = evening.run(FLAGS, [base.prices(), down(), down(), down(), down()])
    assert code == 1 and "stopped: transport_error; ticker MRVL" in text
    assert evening.sleeps == [10.0, 30.0, 90.0]
    assert len(evening.transport.calls) == 1 + 1 + fetch.TRANSPORT_RETRIES
    assert evening.stored() == [1, 0, 0, 0]  # what was stored stays; nothing else is written


def test_the_retries_are_counted_across_the_whole_run_not_per_ticker(research, tmp_path):
    evening = Evening(research, tmp_path)
    responses = [down(), base.prices(), down(), base.prices(), down(), base.prices(), down()]
    code, text = evening.run(FLAGS, responses)
    assert code == 1 and "stopped: transport_error; ticker NVDA" in text
    assert evening.sleeps == [10.0, 30.0, 90.0]  # one retry each for AMD, MRVL and MU
    assert len(evening.transport.calls) == len(responses) and evening.stored() == [1, 1, 1, 0]


@pytest.mark.parametrize(
    "response, reason",
    [
        ((401, b""), "invalid_token"),
        ((403, b""), "invalid_token"),
        ((404, b""), "http_error"),
        ((429, b""), "rate_limited"),
        ((500, b""), "server_error"),
        ((200, b"not json"), "not_json"),
    ],
    ids=["401", "403", "404", "429", "500", "not-json"],
)
def test_an_http_refusal_or_a_bad_response_is_never_retried(research, tmp_path, response, reason):
    evening = Evening(research, tmp_path)
    code, text = evening.run(FLAGS, [response, *[base.prices() for _ in WATCHED]])
    assert code == 1 and reason in text and "retry" not in text
    assert len(evening.transport.calls) == 1 and evening.sleeps == []
    assert evening.stored() == [0, 0, 0, 0]


def test_a_refused_response_or_an_audit_hold_is_never_retried(research, tmp_path):
    # A served session outside the request refuses the response and stops the run.
    stray = base.price_row("2026-10-17", close=100.0)
    status, body = base.prices()
    body = body[:-1] + b"," + base.json.dumps(stray).encode() + b"]"
    evening = Evening(research, tmp_path)
    code, text = evening.run(FLAGS, [(status, body), *[base.prices() for _ in WATCHED]])
    assert code == 1 and "outside_request" in text and "retry" not in text
    assert len(evening.transport.calls) == 1 and evening.sleeps == []
    # An audit that asks for review holds the ticker, unstored and never requested again.
    evening = Evening(research, tmp_path / "held")
    held = base.prices(**{"100": {"low": 500.0}})
    code, text = evening.run(FLAGS, [held, *[base.prices() for _ in WATCHED[1:]]])
    assert code == 0 and "retry" not in text and len(evening.transport.calls) == len(WATCHED)
    assert evening.stored() == [0, 1, 1, 1] and evening.sleeps == []


def test_a_retry_never_starts_at_or_after_the_next_open(research, tmp_path):
    evening = Evening(research, tmp_path, now=ny("2026-10-19", "09:29:55"), up=None)
    evening.network.up = evening.clock.now
    code, text = evening.run(FLAGS, [down(), *[base.prices() for _ in WATCHED]])
    assert code == 1 and "retry" not in text and evening.sleeps == []
    assert len(evening.transport.calls) == 1


def test_without_unattended_nothing_waits_or_retries(research, tmp_path):
    evening = Evening(research, tmp_path, up=None, power="battery")
    code, text = evening.run(FLAGS[:-1], [down(), *[base.prices() for _ in WATCHED]])
    assert code == 1 and "transport_error" in text
    assert evening.network.checks == 0 and evening.sleeps == [] and evening.notices == []
    assert evening.budgets == [len(WATCHED)] and len(evening.transport.calls) == 1


@pytest.mark.parametrize(
    "last, ends",
    [
        ("2026-10-16", ("2026-10-19", "09:30")),  # Friday: the Monday open
        ("2026-11-25", ("2026-11-27", "09:30")),  # Thanksgiving: the Friday after it
        ("2026-10-30", ("2026-11-02", "09:30")),  # across the change to standard time
    ],
    ids=["weekend", "holiday", "dst"],
)
def test_the_recording_window_ends_at_the_next_sessions_open_in_new_york(last, ends):
    end = fetch.recording_ends(pd.Timestamp(last, tz="UTC"))
    assert end == ny(*ends) and end.tz is not None
    assert end.tz_convert("America/New_York").strftime("%H:%M") == "09:30"


def test_the_evening_stays_within_the_keys_hourly_requests():
    assert fetch.TRANSPORT_RETRIES == len(fetch.RETRY_DELAYS) == 3
    assert fetch.MAX_TICKERS + fetch.TRANSPORT_RETRIES <= fetch.HOURLY_REQUESTS == 50


def test_on_battery_the_run_warns_and_notifies_then_carries_on(research, tmp_path):
    evening = Evening(research, tmp_path, power="battery")
    code, text = evening.run(FLAGS, [base.prices() for _ in WATCHED])
    assert code == 0, text
    assert evening.notices == ["on_battery"]
    assert notify.NOTICES["on_battery"] == (
        "running on battery - the Mac may sleep; plug in for reliable evenings"
    )
    assert f"warning: {notify.NOTICES['on_battery']}" in text
    assert evening.stored() == [1, 1, 1, 1]


@pytest.mark.parametrize("power", ["ac", "unknown"])
def test_on_ac_or_an_unknown_source_nothing_is_said(research, tmp_path, power):
    evening = Evening(research, tmp_path, power=power)
    code, text = evening.run(FLAGS, [base.prices() for _ in WATCHED])
    assert code == 0 and evening.notices == [] and "battery" not in text


def test_the_plan_describes_the_unattended_run_and_checks_nothing(research, tmp_path):
    evening = Evening(research, tmp_path, power="battery")
    code, text = evening.run(["--keychain", "--unattended"])
    assert code == 0 and evening.events == [] and evening.notices == []
    assert "unattended: waits up to 5 minutes for api.tiingo.com" in text
    assert "retries a transport error up to 3 times" in text
