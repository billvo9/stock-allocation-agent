"""
The forecast record (PR 3): scoring as sessions close, the hash-chained
ledger, job runs, the synthetic replay, the Model Health input and its
checks, and the daily and schedule scripts. Synthetic data only.
"""

from __future__ import annotations

import importlib.util
import io
import json
import plistlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.health import fixtures as health_fixtures
from stock_agent.health.inputs import load_latest
from stock_agent.health.model import HealthState
from stock_agent.health.stock_lab import ARTIFACT, MIN_SCORED, stock_lab_checks
from stock_agent.stock_lab import monitoring, scoring, synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.scores import rps
from stock_agent.stock_lab.store import DB_NAME, StockLabStore

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


daily = load_script("stock_lab_daily")
schedule = load_script("stock_lab_schedule")


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    home = tmp_path_factory.mktemp("replay")
    store = StockLabStore(home / DB_NAME)
    inputs = home / "health_inputs"
    out = io.StringIO()
    assert daily.main(["--demo-replay", "6", "--health-inputs", str(inputs)],
                      store_factory=lambda: store, out=out) == 0  # fmt: skip
    return store, inputs


# --- scoring ---


def test_every_score_uses_an_outcome_retrieved_after_its_forecast(replayed):
    store, _ = replayed
    for stock_id in synthetic.DEMO_IDS:
        forecasts = {f["forecast_id"]: f for f in store.forecasts(stock_id)}
        retrieved = {b.batch_id: b.retrieved_at for b in store.batches(stock_id)}
        scores = store.scores(stock_id)
        assert scores
        for score in scores:
            assert retrieved[score["batch_id"]] > forecasts[score["forecast_id"]]["created_at"]


def test_scores_are_the_stored_probabilities_scored_never_recomputed(replayed):
    store, _ = replayed
    score = next(
        s for s in store.scores("DEMO_TREND") if s["status"] == "scored" and s["target"] == "1"
    )
    record = next(
        f for f in store.forecasts("DEMO_TREND") if f["forecast_id"] == score["forecast_id"]
    )
    row = record["daily"][0]
    p = np.array([[row["p_decrease"], row["p_stable"], row["p_increase"]]])
    assert score["rps"] == pytest.approx(rps(p, np.array([score["outcome"]]))[0])
    bars = store.bars("DEMO_TREND").bars
    session = bars[bars["session"] == pd.Timestamp(score["session"], tz="UTC")].index[0]
    expected = (
        bars["close"].iloc[session]
        * bars["split_factor"].iloc[session]
        / bars["close"].iloc[session - 1]
        - 1
    )
    assert score["realized_return"] == pytest.approx(expected)


def test_a_session_not_yet_closed_is_pending_and_nothing_is_written_for_it(replayed):
    store, _ = replayed
    latest = store.latest_forecast("DEMO_CALM")
    assert not [s for s in store.scores("DEMO_CALM") if s["forecast_id"] == latest["forecast_id"]]
    bars = store.bars("DEMO_CALM")
    batch_of = {
        row.session.strftime("%Y-%m-%d"): (row.batch_id, "x") for row in bars.bars.itertuples()
    }
    pending = scoring.score_forecast(latest, bars.bars.drop(columns=["batch_id"]), batch_of)
    assert {s.status for s in pending} == {scoring.PENDING}


def test_an_outcome_retrieved_before_the_forecast_is_never_scored(replayed):
    store, _ = replayed
    record = store.forecasts("DEMO_SWING")[0]
    bars = store.bars("DEMO_SWING")
    early = {row.session.strftime("%Y-%m-%d"): (row.batch_id, "2000-01-01T00:00:00.000000Z")
             for row in bars.bars.itertuples()}  # fmt: skip
    results = scoring.score_forecast(record, bars.bars.drop(columns=["batch_id"]), early)
    closed = [s for s in results if s.status != scoring.PENDING]
    assert closed and {s.reason for s in closed} == {"outcome_retrieved_before_forecast"}


def test_scores_are_write_once_and_scoring_again_adds_nothing(replayed):
    store, _ = replayed
    assert scoring.score_store(store, synthetic.DEMO_IDS) == 0
    first = store.scores("DEMO_CALM")[0]
    with pytest.raises(sqlite3.IntegrityError):
        store.record_scores([first])


# --- the ledger ---


def test_the_ledger_covers_every_record_and_verifies(replayed):
    store, _ = replayed
    entries, problem = store.verify_ledger()
    forecasts = sum(len(store.forecasts(s)) for s in synthetic.DEMO_IDS)
    assert problem is None
    assert entries == forecasts + len(store.scores()) + len(store.job_runs())


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("UPDATE forecasts SET record = replace(record, '\"B1\"', '\"B2\"')", "record"),
        ("UPDATE ledger_entries SET payload_sha256 = '0' WHERE seq = 2", "hash"),
        ("DELETE FROM ledger_entries WHERE seq = 3", "sequence"),
        ("UPDATE scores SET outcome = 2, rps = 0.0 WHERE rowid = 1", "score record"),
        ("DELETE FROM scores WHERE rowid = 1", "missing score"),
        ("UPDATE job_runs SET status = 'ok', record = replace(record, 'ok', 'failed')", "record"),
    ],
)
def test_editing_the_store_outside_the_lab_breaks_the_chain(replayed, tmp_path, tamper, message):
    store, _ = replayed
    copy_path = tmp_path / DB_NAME
    copy_path.write_bytes(store.path.read_bytes())
    tmp_path.chmod(0o700)
    connection = sqlite3.connect(copy_path)
    for trigger in [
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    ]:
        connection.execute(f"DROP TRIGGER {trigger}")
    connection.execute(tamper)
    connection.commit()
    connection.close()
    _, problem = StockLabStore(copy_path).verify_ledger()
    assert problem is not None and message in problem


# --- the replay ---


def test_the_replay_fits_only_on_sessions_before_it_starts(replayed):
    store, _ = replayed
    evaluation = store.latest_evaluation(DataClass.SYNTHETIC)
    start = store.forecasts("DEMO_CALM")[0]["as_of_session"]
    assert evaluation["config"]["outcome_end"] == start
    # Half-open windows end on the day after the start; outcomes after it are censored.
    day_after = (pd.Timestamp(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert all(fold["test_end"] <= day_after for fold in evaluation["folds"])
    for stock_id in synthetic.DEMO_IDS:
        history = store.forecasts(stock_id)
        assert len(history) == 7  # the start and six replay days
        assert [f["supersedes"] for f in history[1:]] == [f["forecast_id"] for f in history[:-1]]
        assert history[-1]["as_of_session"] == "2024-12-31"
        assert all(f["data_class"] == "synthetic" for f in history)


# --- monitoring and Model Health ---


def test_psi_is_near_zero_for_one_distribution_and_large_for_a_shift():
    rng = np.random.RandomState(0)
    development = rng.normal(size=5000)
    assert monitoring.psi(development, rng.normal(size=500)) < 0.05
    assert monitoring.psi(development, rng.normal(1.0, 1.0, size=500)) > 0.25
    assert monitoring.psi(development[:50], development[:30]) is None


def test_each_replay_day_writes_one_valid_health_input(replayed):
    _, inputs = replayed
    files = sorted((inputs / ARTIFACT).glob("*.json"))
    assert len(files) == 6
    latest = load_latest(inputs / ARTIFACT, ARTIFACT)
    assert latest.synthetic and latest.document["ledger"]["verified"]
    assert latest.document["lineage"]["feature_version"] == "stock_lab_features_v1"
    with pytest.raises(FileExistsError):
        monitoring.write_monitor(inputs, json.loads(files[-1].read_text()))
    checks = {c.check_id: c for c in stock_lab_checks(inputs / ARTIFACT, allow_synthetic=True)}
    assert checks["forecast.stock_lab_ledger"].state is HealthState.HEALTHY
    assert checks["forecast.stock_lab_scores"].reason == "too_few_scored"
    private = stock_lab_checks(inputs / ARTIFACT)  # the private scope: synthetic is never healthy
    assert all(c.state is not HealthState.HEALTHY for c in private)


@pytest.mark.parametrize(
    ("change", "check_id", "state", "reason"),
    [
        (
            lambda d: d["job"].update(status="partial", failed=["X"]),
            "forecast.stock_lab_job",
            "degraded",
            "job_partial",
        ),
        (
            lambda d: d["job"].update(status="skipped", reason="approvals"),
            "forecast.stock_lab_job",
            "degraded",
            "job_skipped",
        ),
        (
            lambda d: d["scores"].update(rps=0.3),
            "forecast.stock_lab_scores",
            "degraded",
            "worse_than_baseline",
        ),
        (
            lambda d: d["scores"].update(n=MIN_SCORED - 1),
            "forecast.stock_lab_scores",
            "insufficient_evidence",
            "too_few_scored",
        ),
        (
            lambda d: d.update(freshness=[]),
            "data.stock_lab_freshness",
            "insufficient_evidence",
            "no_watchlist",
        ),
        (
            lambda d: d["freshness"][0].update(sessions_behind=None),
            "data.stock_lab_freshness",
            "degraded",
            "stale_prices",
        ),
        (
            lambda d: d.update(inputs_drift=[]),
            "forecast.stock_lab_drift",
            "insufficient_evidence",
            "no_drift_measured",
        ),
        (
            lambda d: d.pop("ledger"),
            "forecast.stock_lab_ledger",
            "unavailable",
            "unrecognized_format",
        ),
    ],
)
def test_the_health_checks_judge_what_the_job_recorded(tmp_path, change, check_id, state, reason):
    document = health_fixtures.stock_lab_monitor()
    change(document)
    health_fixtures.write_input(tmp_path, document)
    checks = {c.check_id: c for c in stock_lab_checks(tmp_path / ARTIFACT, allow_synthetic=True)}
    assert (checks[check_id].state.value, checks[check_id].reason) == (state, reason)


# --- the daily and schedule scripts ---


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-10-16T23:59:00Z", "2026-10-15"),  # 19:59 in New York: today's close not yet
        ("2026-10-17T00:01:00Z", "2026-10-16"),  # 20:01 in New York
        ("2026-10-18T15:00:00Z", "2026-10-16"),  # a Sunday
        ("2026-11-27T03:00:00Z", "2026-11-25"),  # Thanksgiving evening: the holiday is skipped
    ],
)
def test_the_expected_session_follows_the_nyse_calendar(now, expected):
    assert daily.expected_session(pd.Timestamp(now)).strftime("%Y-%m-%d") == expected


def test_the_real_job_scores_records_itself_skipped_and_never_forecasts(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    out = io.StringIO()
    code = daily.main(["--health-inputs", str(tmp_path / "inputs")], store_factory=lambda: store,
                      clock=lambda: pd.Timestamp("2026-10-17T01:00:00Z"), out=out)  # fmt: skip
    assert code == 0, out.getvalue()
    run = store.latest_job_run()
    assert run["status"] == "skipped" and "A4" in run["reason"] and run["data_class"] == "real"
    assert run["expected_session"] == "2026-10-16"
    document = load_latest(tmp_path / "inputs" / ARTIFACT, ARTIFACT).document
    assert document["synthetic"] is False and document["schedule"]["next_expected_at"]
    assert store.verify_ledger() == (1, None)


@pytest.mark.parametrize("flag", ["--authorized", "--inference-window", "--record-real"])
def test_the_real_update_flags_are_refused_until_the_owners_decisions(tmp_path, flag):
    store = StockLabStore(tmp_path / DB_NAME)
    out = io.StringIO()
    assert (
        daily.main([flag, "--health-inputs", str(tmp_path)], store_factory=lambda: store, out=out)
        == 2
    )
    assert "A1, A2 and A4" in out.getvalue() and store.latest_job_run() is None


@pytest.mark.parametrize("sessions", ["0", "61"])
def test_the_replay_length_is_bounded(tmp_path, sessions):
    out = io.StringIO()
    store = StockLabStore(tmp_path / DB_NAME)
    assert daily.main(["--demo-replay", sessions], store_factory=lambda: store, out=out) == 2


def test_the_schedule_is_printed_never_installed(tmp_path):
    out = io.StringIO()
    assert schedule.main([], out=out) == 0
    plan = plistlib.loads(out.getvalue().encode())
    assert plan["Label"] == schedule.LABEL
    assert plan["ProgramArguments"][1].endswith("scripts/stock_lab_daily.py")
    times = plan["StartCalendarInterval"]
    assert [t["Weekday"] for t in times] == [1, 2, 3, 4, 5]
    assert {(t["Hour"], t["Minute"]) for t in times} == {(20, 30)}
    assert schedule.local_time("America/Los_Angeles") == (17, 30)
    assert "launchctl" not in Path(schedule.__file__).read_text().split('"""', 2)[2]  # never run
