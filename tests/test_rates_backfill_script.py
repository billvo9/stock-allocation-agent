"""The DGS3MO vintage backfill script on a fake FRED source: plan, gates, holdout, storage."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.data.vendors.keychain import KeychainFailure, KeychainRefused

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rates_backfill.py"
spec = importlib.util.spec_from_file_location("rates_backfill", SCRIPT)
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)

SECRET = "f" * 32


class FakeSource:
    def __init__(self, rows, listed=None):
        self.rows = rows
        self.listed = listed
        self.calls = []

    def get_series_observations(self, **kwargs):
        self.calls.append(("observations", kwargs))
        frame = pd.DataFrame(self.rows, columns=["date", "value", "realtime_start"])
        frame["realtime_end"] = "9999-12-31"
        return frame

    def get_series_vintage_dates(self, **kwargs):
        self.calls.append(("vintagedates", kwargs))
        if self.listed is not None:
            return list(self.listed)
        return sorted({start for _, _, start in self.rows})


GOOD = [
    ("2003-01-02", "1.20", "2005-06-28"),
    ("2024-10-10", "5.00", "2024-10-11"),
    ("2024-10-14", ".", "2024-10-15"),
    ("2024-10-15", "4.60", "2024-10-16"),
    ("2024-10-15", "4.61", "2024-11-01"),
]


def run(argv, rows=GOOD, *, tmp_path, key_reader=None, env_key=True, monkeypatch=None):
    source = FakeSource(rows)
    keys = []

    def factory(key):
        keys.append(key)
        return source

    if monkeypatch is not None:
        if env_key:
            monkeypatch.setenv("FRED_API_KEY", SECRET)
        else:
            monkeypatch.delenv("FRED_API_KEY", raising=False)
    out = io.StringIO()
    code = backfill.main(
        argv,
        source_factory=factory,
        key_reader=key_reader or (lambda: SECRET),
        clock=lambda: pd.Timestamp("2026-10-27T03:00:00Z"),
        home_factory=lambda: tmp_path / "rates",
        out=out,
    )
    return code, out.getvalue(), source, keys


def test_the_plan_makes_no_request_and_asks_for_nothing_after_2024(tmp_path, monkeypatch):
    code, text, source, keys = run([], tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 0 and source.calls == [] and keys == []
    assert "realtime_end=2024-12-31" in text and "observation_end=2024-12-30" in text
    assert "2025" not in text and SECRET not in text and "api_key=" not in text


def test_a_checking_run_requests_once_prints_diagnostics_and_writes_nothing(tmp_path, monkeypatch):
    code, text, source, keys = run(["--authorized"], tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 0, text
    assert keys == [SECRET]
    assert source.calls == [
        (
            "observations",
            {
                "series_id": "DGS3MO",
                "observation_start": "2002-12-02",
                "observation_end": "2024-12-30",
                "realtime_start": "1776-07-04",  # every vintage: without it, revisions are lost
                "realtime_end": "2024-12-31",
            },
        ),
        (
            "vintagedates",
            {"series_id": "DGS3MO", "realtime_start": "1776-07-04", "realtime_end": "2024-12-31"},
        ),
    ]
    assert "listed_vintage_dates: 5" in text and "listed_not_seen: 0" in text
    assert "revised_observations: 1" in text and "pre_alfred_first_vintages: 1" in text
    assert "first_release_missing: 1" in text  # the Columbus Day "."
    assert SECRET not in text and not (tmp_path / "rates").exists()


def test_a_storing_run_saves_a_new_private_file(tmp_path, monkeypatch):
    code, text, _, _ = run(["--authorized", "--store"], tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 0, text
    stored = list((tmp_path / "rates").glob("USD_TREASURY_3M-vintages-*.parquet"))
    assert len(stored) == 1  # the rate_id its callers already use (config/rates.yaml)
    code, text, _, _ = run(["--authorized", "--store"], tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 0 and "nothing new to store" in text  # a repeat run adds no file
    assert len(list((tmp_path / "rates").glob("*.parquet"))) == 1


def test_a_served_holdout_vintage_refuses_the_whole_response(tmp_path, monkeypatch):
    rows = [*GOOD, ("2024-12-30", "4.30", "2025-01-02")]
    code, text, _, _ = run(
        ["--authorized", "--store"], rows, tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    assert code == 1 and "holdout" in text and not (tmp_path / "rates").exists()


def test_without_a_key_nothing_is_requested(tmp_path, monkeypatch):
    code, text, source, keys = run(
        ["--authorized"], tmp_path=tmp_path, env_key=False, monkeypatch=monkeypatch
    )
    assert code == 2 and keys == [] and source.calls == [] and "FRED_API_KEY is not set" in text

    def refused():
        raise KeychainRefused(KeychainFailure.ITEM_NOT_FOUND, service="s", account="a")

    code, text, _, keys = run(
        ["--authorized", "--keychain"], tmp_path=tmp_path, key_reader=refused,
        monkeypatch=monkeypatch,
    )  # fmt: skip
    assert code == 2 and keys == [] and "nothing was requested" in text


def test_store_needs_authorization(tmp_path, monkeypatch):
    code, text, source, _ = run(["--store"], tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 2 and source.calls == [] and "--store needs --authorized" in text
    with pytest.raises(SystemExit):
        run(["--authorized", "--series", "DGS10"], tmp_path=tmp_path, monkeypatch=monkeypatch)


def test_keychain_reads_the_fred_item_and_hands_the_key_over_in_memory(tmp_path, monkeypatch):
    asked = []

    def read_key(service, account):
        asked.append((service, account))
        return SECRET

    monkeypatch.setattr(backfill, "read_key", read_key)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    source = FakeSource(GOOD)
    keys = []
    out = io.StringIO()
    code = backfill.main(
        ["--authorized", "--keychain"],
        source_factory=lambda key: keys.append(key) or source,
        clock=lambda: pd.Timestamp("2026-10-27T03:00:00Z"),
        home_factory=lambda: tmp_path / "rates",
        out=out,
    )
    assert code == 0, out.getvalue()
    assert asked == [("stock-agent-fred", "rates")] and keys == [SECRET]
    assert "FRED_API_KEY" not in __import__("os").environ  # never put in the environment
    assert SECRET not in out.getvalue()


def test_a_listed_holdout_vintage_date_refuses_the_run(tmp_path, monkeypatch):
    source = FakeSource(GOOD, listed=["2024-10-11", "2025-01-02"])
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    out = io.StringIO()
    code = backfill.main(
        ["--authorized", "--store"],
        source_factory=lambda key: source,
        clock=lambda: pd.Timestamp("2026-10-27T03:00:00Z"),
        home_factory=lambda: tmp_path / "rates",
        out=out,
    )
    assert code == 1 and "holdout" in out.getvalue() and not (tmp_path / "rates").exists()


def test_an_observation_outside_the_requested_dates_refuses_the_run(tmp_path, monkeypatch):
    rows = [*GOOD, ("2002-11-29", "1.21", "2005-06-28")]
    code, text, _, _ = run(["--authorized"], rows, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 1 and "outside the requested dates" in text


class FailingSource(FakeSource):
    def get_series_observations(self, **kwargs):
        raise RuntimeError("FRED series observations request failed for series DGS3MO: HTTP 500.")


class SurprisingSource(FakeSource):
    def get_series_observations(self, **kwargs):
        raise OSError(f"connection to https://api.stlouisfed.org/?api_key={SECRET} reset")


@pytest.mark.parametrize("source_type", [FailingSource, SurprisingSource])
def test_a_failed_request_stops_without_showing_the_key_or_the_url(
    tmp_path, monkeypatch, source_type
):
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    out = io.StringIO()
    code = backfill.main(
        ["--authorized", "--store"],
        source_factory=lambda key: source_type(GOOD),
        clock=lambda: pd.Timestamp("2026-10-27T03:00:00Z"),
        home_factory=lambda: tmp_path / "rates",
        out=out,
    )
    text = out.getvalue()
    assert code == 1 and text.startswith("stopped:")
    assert SECRET not in text and "api_key" not in text and "stlouisfed" not in text
    assert not (tmp_path / "rates").exists()
