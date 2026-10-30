"""
The daily job's real path (scripts/stock_lab_daily.py --dry-run / --record-real
--model ID): a registered frozen model, each watched stock's inference window,
the B1 comparator, scoring from the inference store, the job run and its Model
Health input. Real-classified fixtures on a weekday calendar with an injected
clock; nothing real, no network, and activation only through a test-only
monkeypatch of the owner's (empty) activation set.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    Clock,
    next_day,
    ny,
    open_store,
    record,
    steady_bars,
    weekdays,
    window_start,
)

from stock_agent.health.model import HealthState
from stock_agent.health.stock_lab import stock_lab_checks
from stock_agent.stock_lab import evening, monitoring, registry, synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.evaluation import EvaluationConfig, evaluate
from stock_agent.stock_lab.store import DB_NAME, StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_daily.py"
spec = importlib.util.spec_from_file_location("stock_lab_daily_real", SCRIPT)
daily = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)

SOURCES = {"AMD": "DEMO_CALM", "MU": "DEMO_SWING", "NVDA": "DEMO_TREND"}
CONFIG = EvaluationConfig(
    size_reps=8, sensitivity_bands=(), outcome_end=pd.Timestamp("2024-12-31", tz="UTC")
)
FROZEN_AT = pd.Timestamp("2026-10-01T22:00:00Z")


@pytest.fixture(scope="module")
def frozen():
    histories = {}
    for ticker, stock_id in SOURCES.items():
        bars = synthetic.demo_bars(stock_id).copy()
        bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit(
            "us"
        )
        histories[ticker] = bars
    run = evaluate(histories, CONFIG, data_class="real", run_id="R-daily", created_at=FROZEN_AT)
    return run, registry.freeze(histories, run, CONFIG, created_at=FROZEN_AT)


@pytest.fixture
def lab(tmp_path, frozen):
    evaluation, model = frozen
    clock = Clock(ny(AS_OF, "20:35"))
    store = StockLabStore(tmp_path / "research" / DB_NAME, clock=clock)
    store.add(parse_tickers(" ".join(SOURCES)))
    store.record_evaluation(evaluation)
    store.record_frozen_model(model)
    inference, inference_clock = open_store(tmp_path)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"),
               ticker=ticker, bars=steady_bars(window_start(), AS_OF, stock=stock_id))  # fmt: skip
    return store, inference, model, clock, tmp_path


def _main(lab, *flags):
    store, inference, model, clock, tmp_path = lab
    out = io.StringIO()
    code = daily.main(
        [*flags, "--model", model["model_id"][:16], "--health-inputs", str(tmp_path / "health"),
         "--ledger-head-backup", str(tmp_path / "kept_head")],
        store_factory=lambda: store, clock=lambda: clock.now, out=out,
        inference_path=inference.path, calendar=weekdays, calendar_id=CALENDAR_ID,
    )  # fmt: skip
    return code, out.getvalue()


def test_a_dry_run_prints_every_forecast_with_b1_and_writes_nothing(lab):
    store = lab[0]
    before = store.path.read_bytes()
    code, text = _main(lab, "--dry-run")
    assert code == 0, text
    assert "dry run: nothing recorded" in text and text.count(": dry_run; next session") == 3
    assert "B1 Incr." in text and f"expected session {AS_OF}" in text
    assert store.path.read_bytes() == before
    assert store.latest_job_run() is None and not (lab[4] / "health").exists()


def test_recording_is_refused_until_the_owner_activates_the_model(lab):
    code, text = _main(lab, "--record-real")
    assert code == 2 and "is not activated" in text and "decision A4" in text
    assert all(lab[0].forecasts(t) == () for t in SOURCES)


def test_an_activated_model_records_forecasts_scores_and_the_health_input(lab, monkeypatch):
    store, _, model, clock, tmp_path = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    for ticker in SOURCES:
        (forecast,) = store.forecasts(ticker)
        assert forecast["data_class"] == "real" and forecast["as_of_session"] == AS_OF
        assert forecast["comparator"]["estimator"] == "B1"
    run = store.latest_job_run()
    assert run["status"] == "ok" and run["expected_session"] == AS_OF and run["activation"] is None
    assert [r["status"] for r in run["results"]] == ["recorded"] * 3
    assert {r["ticker"]: r["sessions_behind"] for r in run["freshness"]} == dict.fromkeys(
        SOURCES, 0
    )
    (written,) = (tmp_path / "health" / "stock_lab_monitor").glob("*.json")
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["synthetic"] is False and document["ledger"]["verified"] is True
    assert document["outcomes"]["pending"] == 0 and document["job"]["results"] == run["results"]
    assert all(row["psi"] is not None for row in document["inputs_drift"])
    assert document["lineage"]["model_version"] == model["model_id"][:16]
    # The next evening: the new session's batch is stored first, then the job scores and forecasts.
    first = next_day(AS_OF)
    inference_clock = Clock(ny(first, "20:31"))
    inference = type(lab[1])(lab[1].path, clock=inference_clock)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(first), first,
               retrieved_at=ny(first, "20:30"), ticker=ticker,
               bars=steady_bars(window_start(first), first, stock=stock_id))  # fmt: skip
    clock.now = ny(first, "20:35")
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    assert "scores: 3 scored" in text
    assert [len(store.forecasts(t)) for t in SOURCES] == [2, 2, 2]
    assert {s["target"] for t in SOURCES for s in store.scores(t)} == {"1"}


def test_a_mac_that_sleeps_after_the_decision_still_records_with_honest_timestamps(
    lab, monkeypatch
):
    # 2026-10-27: the record step's decision, then a sleep before the writes, so the store
    # refused every forecast as backdated. Each is now made again at the time of its write.
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    clock.now = ny(AS_OF, "21:07")
    build, slept = evening.build_panel, []

    def sleeps(*args, **kwargs):  # the windows are read; the Mac sleeps 25 minutes, once
        if not slept:
            slept.append(True)
            clock.now += pd.Timedelta(minutes=25)
        return build(*args, **kwargs)

    monkeypatch.setattr(evening, "build_panel", sleeps)
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    assert store.latest_job_run()["status"] == "ok"
    for ticker in SOURCES:
        (forecast,) = store.forecasts(ticker)
        made = pd.Timestamp(forecast["created_at"])
        assert made == pd.Timestamp(forecast["data"]["decision_time"]) == ny(AS_OF, "21:32")
        assert pd.Timestamp(forecast["data"]["retrieved_at"]) <= made


def test_a_stale_close_is_reported_and_never_invented(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    clock.now = ny(next_day(AS_OF), "20:35")  # no batch for that session was stored
    code, text = _main(lab, "--record-real")
    run = store.latest_job_run()
    assert run["status"] == "partial" and sorted(run["skipped"]) == sorted(SOURCES)
    assert all(store.forecasts(t) == () for t in SOURCES) and "stale_prices" in text
    assert code == 0


def test_an_unknown_or_ambiguous_model_id_is_refused(lab):
    store, inference, _, clock, _ = lab
    out = io.StringIO()
    code = daily.main(
        ["--dry-run", "--model", "f" * 16], store_factory=lambda: store, clock=lambda: clock.now,
        out=out, inference_path=inference.path, calendar=weekdays, calendar_id=CALENDAR_ID,
    )  # fmt: skip
    assert code == 2 and "--model must name one registered real frozen model" in out.getvalue()
    assert store.frozen_models(DataClass.REAL)  # the registered one is still there


def test_a_run_after_the_next_open_records_a_failed_run_and_exits_1(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    clock.now = ny(next_day(AS_OF), "09:31")  # the evening job ran too late: no forecast
    code, text = _main(lab, "--record-real")
    run = store.latest_job_run()
    assert code == 1 and run["status"] == "failed"
    assert sorted(run["failed"]) == sorted(SOURCES) and "after_next_open" in text
    assert all(store.forecasts(t) == () for t in SOURCES)


def test_a_dry_run_after_a_recorded_evening_scores_nothing(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    assert _main(lab, "--record-real")[0] == 0
    first = next_day(AS_OF)
    inference_clock = Clock(ny(first, "20:31"))
    inference = type(lab[1])(lab[1].path, clock=inference_clock)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(first), first,
               retrieved_at=ny(first, "20:30"), ticker=ticker,
               bars=steady_bars(window_start(first), first, stock=stock_id))  # fmt: skip
    clock.now = ny(first, "20:35")
    code, text = _main(lab, "--dry-run")
    assert code == 0 and "scores: 3 scored" in text  # what recording would write ...
    assert all(store.scores(t) == () for t in SOURCES)  # ... and nothing was written
    assert [len(store.forecasts(t)) for t in SOURCES] == [1, 1, 1]


def test_a_model_id_shorter_than_16_characters_is_refused(lab):
    store, inference, model, clock, _ = lab
    out = io.StringIO()
    code = daily.main(
        ["--dry-run", "--model", model["model_id"][:15]], store_factory=lambda: store,
        clock=lambda: clock.now, out=out, inference_path=inference.path, calendar=weekdays,
        calendar_id=CALENDAR_ID,
    )  # fmt: skip
    assert code == 2 and "at least 16 characters" in out.getvalue()


def test_a_skipped_evening_stays_missed_on_time_and_is_reconstructed_apart(lab, monkeypatch):
    # Decision A4 never backfilled a missed evening; the owner's decision of 2026-10-30
    # replaces that for the activated model: it is reconstructed, labelled and kept apart,
    # and the evening stays missed on time.
    store, _, model, clock, tmp_path = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    assert f"evenings since {AS_OF}: 0 of 0 closed evenings recorded" in text
    # No run on the next evening (the Mac slept through it); the one after runs normally.
    skipped, third = next_day(AS_OF), next_day(AS_OF, 2)
    inference_clock = Clock(ny(third, "20:31"))
    inference = type(lab[1])(lab[1].path, clock=inference_clock)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(third), third,
               retrieved_at=ny(third, "20:30"), ticker=ticker,
               bars=steady_bars(window_start(third), third, stock=stock_id))  # fmt: skip
    clock.now = ny(third, "20:35")
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    line = f"evenings since {AS_OF}: 1 of 2 closed evenings recorded; missed (latest 5): {skipped}"
    assert line in text.splitlines()
    assert {f["as_of_session"] for t in SOURCES for f in store.forecasts(t)} == {AS_OF, third}
    assert {f["as_of_session"] for t in SOURCES
            for f in store.forecasts(t, reconstructed=True)} == {skipped}  # fmt: skip
    written = sorted((tmp_path / "health" / "stock_lab_monitor").glob("*.json"))
    document = json.loads(written[-1].read_text(encoding="utf-8"))
    assert document["schema_version"] == "1.0"  # `evenings` is additive: the format stays
    assert document["evenings"] == {"since": AS_OF, "due": 2, "recorded": 1, "missed": [skipped],
                                    "recent": 10, "recent_missed": 1,
                                    "reconstructed": [skipped]}  # fmt: skip
    checks = {c.check_id: c for c in stock_lab_checks(written[-1].parent)}
    job = checks["forecast.stock_lab_job"]
    assert (job.state, job.reason) == (HealthState.DEGRADED, "evening_missed")
    assert skipped in job.summary


def test_a_dry_run_reports_the_evenings_and_records_nothing(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    assert _main(lab, "--record-real")[0] == 0
    _store_batch(lab, next_day(AS_OF))  # its prices are in: a reconstruction could be made
    before = store.path.read_bytes()
    clock.now = ny(next_day(AS_OF, 2), "09:30")  # the next evening was never recorded
    code, text = _main(lab, "--dry-run")
    assert code == 0 and f"missed (latest 5): {next_day(AS_OF)}" in text
    assert store.path.read_bytes() == before


# --- the read-only environment check (--check-environment) ------------------------------


def _check_environment(lab, model_id: str) -> tuple[int, str]:
    store = lab[0]
    out = io.StringIO()
    code = daily.main(
        ["--check-environment", "--model", model_id], store_factory=lambda: store, out=out
    )
    return code, out.getvalue()


def _row(key: str, frozen: str, here: str, verdict: str) -> str:
    return (
        rf"\n  {re.escape(key)}\s+frozen {re.escape(frozen)}\s+here {re.escape(here)}\s+{verdict}\n"
    )


def test_the_environment_check_shows_the_frozen_versions_beside_this_pythons(
    lab, tmp_path, monkeypatch
):
    store, _, model, _, _ = lab
    # No override: a fall-through to the real job would write under this HOME.
    monkeypatch.delenv(monitoring.HOME_ENV)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runs = store.job_runs()
    code, text = _check_environment(lab, model["model_id"][:16])
    assert code == 0, text
    assert set(model["environment"]) == {"python", "numpy", "pandas", "scikit-learn"}
    for key, version in model["environment"].items():
        assert re.search(_row(key, version, version, "same"), text), key
    assert text.rstrip().endswith("same: the evening can load this model with this Python")
    # Read-only: no run recorded, no forecast, no health input anywhere.
    assert store.job_runs() == runs and all(store.forecasts(t) == () for t in SOURCES)
    assert not (tmp_path / "home").exists()


def test_a_different_environment_is_named_with_what_the_evening_needs(lab, monkeypatch):
    _, _, model, _, _ = lab
    frozen = model["environment"]
    here = {**frozen, "python": "3.13", "numpy": "0.0.1"}
    monkeypatch.setattr(registry, "environment", lambda: here)
    code, text = _check_environment(lab, model["model_id"])
    assert code == 1
    assert re.search(_row("numpy", frozen["numpy"], "0.0.1", "DIFFERS"), text)
    assert re.search(_row("python", frozen["python"], "3.13", "DIFFERS"), text)
    assert re.search(_row("pandas", frozen["pandas"], frozen["pandas"], "same"), text)
    assert "the evening would refuse this model (environment_mismatch)" in text
    assert (
        f"It needs a venv made from Python {frozen['python']} with exactly "
        f"numpy=={frozen['numpy']}, never newer versions" in text
    )


def test_a_package_that_cannot_load_is_reported_not_raised(lab, monkeypatch):
    def missing():
        raise registry.metadata.PackageNotFoundError("scikit-learn")

    monkeypatch.setattr(registry, "environment", missing)
    code, text = _check_environment(lab, lab[2]["model_id"])
    assert code == 1 and "cannot load the model" in text and "scikit-learn" in text


@pytest.mark.parametrize("model_id", ["f" * 16, None])
def test_the_environment_check_needs_one_registered_model(lab, model_id):
    store = lab[0]
    out = io.StringIO()
    argv = ["--check-environment", *(["--model", model_id] if model_id else [])]
    assert daily.main(argv, store_factory=lambda: store, out=out) == 2
    assert "--model must name one registered real frozen model" in out.getvalue()


def test_the_environment_check_is_its_own_mode(lab):
    with pytest.raises(SystemExit) as stopped:
        daily.main(["--check-environment", "--record-real", "--model", lab[2]["model_id"]])
    assert stopped.value.code == 2
    store, runs = lab[0], lab[0].job_runs()
    for other in (["--demo-replay", "3"], ["--authorized"], ["--inference-window"]):
        out = io.StringIO()
        argv = ["--check-environment", "--model", lab[2]["model_id"], *other]
        assert daily.main(argv, store_factory=lambda: store, out=out) == 2, other
        assert "--check-environment takes --model ID only" in out.getvalue()
    assert store.job_runs() == runs  # nothing ran


# --- reconstructing missed evenings (owner decision of 2026-10-30) ---------------------------


def _store_batch(lab, day: str) -> None:
    """Each stock's window ending `day`, retrieved at 20:30 New York that day."""

    clock = Clock(ny(day, "20:31"))
    inference = type(lab[1])(lab[1].path, clock=clock)
    for ticker, stock_id in SOURCES.items():
        record(inference, clock, window_start(day), day, retrieved_at=ny(day, "20:30"),
               ticker=ticker, bars=steady_bars(window_start(day), day, stock=stock_id))  # fmt: skip


def _missed_friday(lab, monkeypatch):
    """Thursday recorded by the job; Friday's prices stored and its evening missed; Monday's."""

    model = lab[2]
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    assert _main(lab, "--record-real")[0] == 0  # Thursday 20:35
    friday, monday = next_day(AS_OF), next_day(AS_OF, 2)
    _store_batch(lab, friday)
    _store_batch(lab, monday)
    return friday, monday


def test_the_next_evening_reconstructs_the_missed_one_and_it_stays_missed(lab, monkeypatch):
    store, _, _, clock, tmp_path = lab
    friday, monday = _missed_friday(lab, monkeypatch)
    clock.now = ny(monday, "20:35")
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    assert f"reconstructed {friday}: AMD, MU, NVDA (made after the fact, kept apart" in text
    for ticker in SOURCES:
        assert [f["as_of_session"] for f in store.forecasts(ticker)] == [AS_OF, monday]
        (rebuilt,) = store.forecasts(ticker, reconstructed=True)
        assert rebuilt["as_of_session"] == friday
        assert pd.Timestamp(rebuilt["created_at"]) >= ny(monday, "20:35")
    run = store.latest_job_run()
    assert run["status"] == "ok" and [r["status"] for r in run["results"]] == ["recorded"] * 3
    newest = max((tmp_path / "health" / "stock_lab_monitor").glob("*.json"))
    document = json.loads(newest.read_text(encoding="utf-8"))
    assert document["evenings"]["missed"] == [friday]  # missed on time
    assert document["evenings"]["reconstructed"] == [friday]  # and noted
    assert document["evenings"]["recorded"] + 1 == document["evenings"]["due"]


def test_the_manual_command_reconstructs_and_scores_then_finds_nothing_more(lab, monkeypatch):
    store, _, _, clock, _ = lab
    friday, monday = _missed_friday(lab, monkeypatch)
    runs = len(store.job_runs())
    clock.now = ny(next_day(AS_OF, 3), "12:00")  # Tuesday midday: Monday is missed too
    code, text = _main(lab, "--reconstruct-missed")
    assert code == 0, text
    assert f"reconstructed {friday}: AMD, MU, NVDA" in text
    assert f"reconstructed {monday}: AMD, MU, NVDA" in text
    for ticker in SOURCES:
        made = {f["as_of_session"]: f for f in store.forecasts(ticker, reconstructed=True)}
        assert sorted(made) == [friday, monday]
        scored = {s["forecast_id"] for s in store.scores(ticker, reconstructed=True)}
        assert scored == {made[friday]["forecast_id"]}  # Monday's outcome is stored; Tuesday's not
    assert len(store.job_runs()) == runs  # no evening run is recorded by the manual command
    assert all(store.scores(t) == () for t in SOURCES)  # live forecasts: the evening run's
    code, text = _main(lab, "--reconstruct-missed")
    assert code == 0 and "nothing to reconstruct" in text


def test_the_manual_command_needs_an_activated_model_and_is_its_own_mode(lab):
    code, text = _main(lab, "--reconstruct-missed")
    assert code == 2 and "is not activated" in text
    for flags in (("--reconstruct-missed", "--record-real"), ("--reconstruct-missed", "--dry-run")):
        with pytest.raises(SystemExit):
            _main(lab, *flags)


def test_the_manual_command_exits_1_when_a_window_cannot_be_read_yet(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    assert _main(lab, "--record-real")[0] == 0  # Thursday; Friday's prices never stored
    clock.now = ny(next_day(AS_OF, 2), "12:00")
    code, text = _main(lab, "--reconstruct-missed")
    assert code == 1
    for ticker in SOURCES:
        assert f"not reconstructed {ticker} as of {next_day(AS_OF)} (short_window" in text
        assert store.forecasts(ticker, reconstructed=True) == ()
