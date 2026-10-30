"""
The sealed holdout run (stock_lab.holdout.evaluate): the frozen model's
forecasts from each decision's 252-session window, exactly as the evening job
computes them, scored against B0 on the outcomes after each decision.
Synthetic stocks under the protocol's tickers, a short test protocol on a
weekday calendar; no network, no real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stock_lab_holdout_helpers import (
    CALENDAR_ID,
    CREATED,
    frozen_model,
    holdout_bars,
    short_protocol,
    weekdays,
)

from stock_agent.model_diagnostics.inference import fold_block_t
from stock_agent.stock_lab import frozen_forecast, registry, scores
from stock_agent.stock_lab.evaluation import build_panel
from stock_agent.stock_lab.holdout import evaluate
from stock_agent.stock_lab.holdout.protocol import HoldoutRefused, Refusal, record_id
from stock_agent.stock_lab.size_simulation import one_sided_p

CODE = {"sha": "0" * 40, "clean": True}


@pytest.fixture(scope="module")
def setting():
    _, model_record = frozen_model()
    model = registry.load(model_record)
    protocol = short_protocol(model.model_id)
    histories = {ticker: holdout_bars(protocol, ticker) for ticker in protocol.tickers}
    return model, protocol, histories


def _run(setting, histories=None, protocol=None, model=None):
    model_, protocol_, histories_ = setting
    return evaluate.run(
        histories_ if histories is None else histories,
        model_ if model is None else model,
        protocol=protocol_ if protocol is None else protocol,
        sessions=weekdays,
        calendar_id=CALENDAR_ID,
        batches=[{"batch_id": "fixture"}],
        created_at=CREATED,
        code=CODE,
    )


@pytest.fixture(scope="module")
def record(setting):
    return _run(setting)


def _forecasts(setting, histories):
    model, protocol, _ = setting
    days = protocol.fetch_sessions(weekdays)
    rows, _ = evaluate._forecasts(histories, model, protocol, days, protocol.decisions(weekdays))
    return {(row["date"], row["ticker"]): row for row in rows}


def _window(bars: pd.DataFrame, decision: pd.Timestamp, sessions: int = 252) -> pd.DataFrame:
    return bars[bars["session"] <= decision].iloc[-sessions:].reset_index(drop=True)


def test_the_record_is_sealed_and_names_what_was_evaluated(setting, record):
    model, protocol, _ = setting
    assert record["record_id"] == record_id(record)
    assert record["protocol_sha256"] == protocol.sha256()
    assert record["protocol"] == protocol.as_dict()
    assert record["model"] == {
        "model_id": model.model_id,
        "version": model.version,
        "estimator": "B1",
        "state_rule": "frozen_long_run",
        "data_class": "real",
        "comparator": "B0",
    }
    assert record["counts"]["decisions"] == 29 and record["counts"]["forecasts"] == 116
    assert record["counts"]["first_decision"] == "2024-12-31"
    assert record["data"]["batches"] == [{"batch_id": "fixture"}]
    assert record["code"] == CODE and record["environment"] == registry.environment()
    assert len(record["limitations"]) == len(evaluate.LIMITATIONS)


def _evening_record(model, panel, ticker, decision):
    """The evening job's forecast record (frozen_forecast.forecast_record) for this window."""

    days = weekdays(decision, decision + pd.Timedelta(days=10))[1:6].strftime("%Y-%m-%d")
    data = {"source": "fixture", "batch_ids": ["fixture"], "retrieved_at": "fixture",
            "last_close_session": decision.strftime("%Y-%m-%d"), "decision_time": "fixture",
            "window": {"first_session": "fixture", "sessions": 252, "gaps": [],
                       "revised_sessions": [], "calendar": CALENDAR_ID}}  # fmt: skip
    return frozen_forecast.forecast_record(
        panel, ticker, model, target_sessions=list(days), created_at=CREATED, data=data,
        data_class="real",
    )  # fmt: skip


def _triple(entry) -> list[float]:
    return [entry["p_decrease"], entry["p_stable"], entry["p_increase"]]


def test_each_forecast_is_the_evening_jobs_from_the_252_sessions_ending_at_its_decision(setting):
    model, protocol, histories = setting
    found = _forecasts(setting, histories)
    for decision in protocol.decisions(weekdays)[[0, 17, -1]]:
        windows = {ticker: _window(bars, decision) for ticker, bars in histories.items()}
        assert all(len(window) == 252 for window in windows.values())
        panel = build_panel(windows, model.final.config)
        for ticker in protocol.tickers:
            evening = _evening_record(model, panel, ticker, decision)
            row = found[(decision, ticker)]
            assert evening["as_of_session"] == decision.strftime("%Y-%m-%d")
            np.testing.assert_allclose(
                row["daily"], [_triple(entry) for entry in evening["daily"]], rtol=0, atol=1e-15
            )
            np.testing.assert_allclose(
                row["total"], _triple(evening["cumulative"]), rtol=0, atol=1e-15
            )
            np.testing.assert_array_equal(row["b0_daily"], model.long_run["daily"][ticker])
            np.testing.assert_array_equal(row["b0_total"], model.long_run["cumulative"][ticker])


def test_every_scored_probability_is_finite_and_sums_to_one(setting):
    _, _, histories = setting
    for row in _forecasts(setting, histories).values():
        for values in (*row["daily"], row["total"], row["b0_daily"], row["b0_total"]):
            assert np.isfinite(values).all() and abs(values.sum() - 1.0) < 1e-12


def test_another_window_would_give_other_forecasts(setting):
    # The comparison above can tell windows apart: a shorter window, or one ending a
    # session earlier, changes B1 (so the run's own window is the one checked).
    model, protocol, histories = setting
    decision = protocol.decisions(weekdays)[20]
    found = _forecasts(setting, histories)[(decision, "MRVL")]["daily"]
    for windows in (
        {t: _window(b, decision, 200) for t, b in histories.items()},
        {t: _window(b, decision - pd.Timedelta(days=1)) for t, b in histories.items()},
    ):
        other = registry.probabilities(build_panel(windows, model.final.config), "MRVL", model)
        assert not np.allclose(other.daily, found)


def test_a_forecast_never_reads_a_session_after_its_decision_or_before_its_window(setting):
    _, protocol, histories = setting
    decisions = protocol.decisions(weekdays)
    decision = decisions[15]
    base = _forecasts(setting, histories)
    later = {t: b.copy() for t, b in histories.items()}
    for bars in later.values():  # every price after the decision doubles or halves
        after = bars["session"] > decision
        factor = np.where(np.arange(after.sum()) % 2, 2.0, 0.5)
        for column in ("open", "high", "low", "close"):
            bars.loc[after, column] = bars.loc[after, column].to_numpy() * factor
    changed = _forecasts(setting, later)
    for d in decisions[decisions <= decision]:
        for ticker in protocol.tickers:
            np.testing.assert_array_equal(base[(d, ticker)]["daily"], changed[(d, ticker)]["daily"])
    assert any(
        not np.allclose(base[(d, "AMD")]["daily"], changed[(d, "AMD")]["daily"])
        for d in decisions[decisions > decision]
    )
    # a session before the decision's window does not count either
    earlier = {t: b.copy() for t, b in histories.items()}
    first = earlier["AMD"]["session"] == earlier["AMD"]["session"].iloc[0]
    earlier["AMD"].loc[first, ["open", "high", "low", "close"]] *= 3.0
    moved = _forecasts(setting, earlier)
    window_start_moved = decisions[1]  # from here on, the first session left the window
    np.testing.assert_array_equal(
        base[(window_start_moved, "AMD")]["daily"], moved[(window_start_moved, "AMD")]["daily"]
    )
    # while it was in the window (the first decision), a change there did count: tripling the
    # first close makes the next session a Decrease, dividing it by three an Increase, so one
    # of the two changes that session's category whatever it was
    lower = {t: b.copy() for t, b in histories.items()}
    lower["AMD"].loc[first, ["open", "high", "low", "close"]] /= 3.0
    shifted = (moved, _forecasts(setting, lower))
    assert any(
        not np.allclose(base[(decisions[0], "AMD")]["daily"], other[(decisions[0], "AMD")]["daily"])
        for other in shifted
    )


def test_outcomes_are_the_categories_of_the_sessions_after_the_decision(setting):
    model, protocol, histories = setting
    outcomes = evaluate._outcomes(histories, model, protocol)
    days = protocol.fetch_sessions(weekdays)
    decision = protocol.decisions(weekdays)[-1]
    position = days.get_loc(decision)
    for h in range(1, 6):
        session = days[position + h]
        assert (
            outcomes.loc[("NVDA", decision), f"target_{h}"]
            == (outcomes.loc[("NVDA", session), "category_code"])
        )
    assert outcomes.loc[("NVDA", decision), "target_end_date"] == days[-1]


def test_the_deciding_test_recomputed_independently(setting, record):
    model, protocol, histories = setting
    forecasts = _forecasts(setting, histories)
    outcomes = evaluate._outcomes(histories, model, protocol)
    decisions = protocol.decisions(weekdays)
    per_date = []
    for decision in decisions:
        differences = []
        for h in range(5):
            row_diffs = []
            for ticker in protocol.tickers:
                y = np.array([int(outcomes.loc[(ticker, decision), f"target_{h + 1}"])])
                row = forecasts[(decision, ticker)]
                mine = scores.rps(row["daily"][h][None, :], y)[0]
                theirs = scores.rps(row["b0_daily"][None, :], y)[0]
                row_diffs.append(mine - theirs)
            differences.append(np.mean(row_diffs))
        per_date.append(np.mean(differences))
    # the test protocol's blocks: to 2025-01-14 (with 2024-12-31), to 2025-01-31, then February
    blocks = ["A" if d < pd.Timestamp("2025-01-15", tz="UTC") else
              "B" if d < pd.Timestamp("2025-02-01", tz="UTC") else "C" for d in decisions]  # fmt: skip
    expected = fold_block_t(np.array(per_date), np.array(blocks), ci_level=0.95)
    deciding = [row for row in record["comparisons"] if row["deciding"]]
    assert len(deciding) == 1 and deciding[0]["target"] == "daily_pooled"
    assert deciding[0]["score"] == "rps" and deciding[0]["vs"] == "B0"
    assert deciding[0]["block_mean_diff"] == pytest.approx(expected.estimate, abs=1e-15)
    assert deciding[0]["p_one_sided"] == pytest.approx(one_sided_p(expected), abs=1e-12)
    assert deciding[0]["df"] == 2 and record["decision"]["df"] == 2
    assert record["decision"]["p_one_sided"] == deciding[0]["p_one_sided"]
    assert [b["block"] for b in deciding[0]["by_block"]] == ["A", "B", "C"]


def test_the_model_beats_b0_only_below_alpha_with_a_negative_difference(setting, record):
    from dataclasses import replace

    _, protocol, _ = setting
    difference = record["decision"]["block_mean_diff"]
    p = record["decision"]["p_one_sided"]
    assert difference < 0 and 0.05 < p < 0.5  # the fixture's no-skill result
    assert record["decision"]["beats_comparator"] is False
    lenient = replace(protocol, alpha=0.6)
    assert _run(setting, protocol=lenient)["decision"]["beats_comparator"] is True


def test_every_score_and_the_total_are_reported_but_only_one_decides(record):
    keys = {(row["target"], row["score"]) for row in record["comparisons"]}
    assert keys == {
        (target, score)
        for target in ("daily_pooled", "target_cum")
        for score in ("rps", "brier", "log_loss")
    }
    assert sum(row["deciding"] for row in record["comparisons"]) == 1
    targets = {row["target"] for row in record["metrics"]}
    assert targets == {*(f"target_{h}" for h in range(1, 6)), "target_cum"}
    assert {row["stock"] for row in record["by_stock"]} == {"AMD", "MRVL", "MU", "NVDA"}
    assert {row["year"] for row in record["by_year"]} == {2025}  # 2024-12-31 counts in 2025


def test_calibration_uses_the_briefings_clustered_intervals(record):
    daily = [row for row in record["calibration"] if row["target"] == "target_1"]
    total = [row for row in record["calibration"] if row["target"] == "target_cum"]
    assert daily and total
    assert all(row["clusters"] == row["dates"] for row in daily)  # lag 0
    assert all(row["clusters"] == -(-row["dates"] // 5) for row in total)  # lag 4


def test_base_rates_are_each_sessions_category_in_the_holdout(setting, record):
    model, protocol, _ = setting
    row = next(r for r in record["base_rates"] if r["stock"] == "MU")
    holdout_days = weekdays(protocol.holdout_start, protocol.holdout_end)
    assert row["sessions"] == len(holdout_days)
    shares = [row[f"holdout_{c}"] for c in ("decrease", "stable", "increase")]
    assert sum(shares) == pytest.approx(1.0)
    assert row["b0_stable"] == pytest.approx(float(model.long_run["daily"]["MU"][1]))


def test_a_missing_close_gets_no_forecast_and_its_targets_are_not_scored(setting):
    _, protocol, histories = setting
    decisions = protocol.decisions(weekdays)
    gap = decisions[12]
    holed = {t: b.copy() for t, b in histories.items()}
    at = holed["MU"]["session"] == gap
    holed["MU"].loc[at, ["open", "high", "low", "close", "volume"]] = np.nan
    found = _run(setting, histories=holed)
    assert found["counts"]["skipped"] == {"as_of_close_missing": 1}
    assert found["counts"]["forecasts"] == 115
    # the gap breaks MU's returns on the gap session and the session after it: target_h of a
    # decision h or h - 1 sessions before the gap (the one at the gap is not forecast)
    assert found["counts"]["scored"]["target_1"] == 115 - 1
    for h in range(2, 6):
        assert found["counts"]["scored"][f"target_{h}"] == 115 - 2
    assert found["counts"]["scored"]["target_cum"] == 115 - 5  # every total spanning the gap
    deciding = next(row for row in found["comparisons"] if row["deciding"])
    assert deciding["dates"] == 29 and deciding["status"] == "ok"  # the other stocks remain


def test_a_run_of_another_model_or_on_other_histories_is_refused(setting):
    _, protocol, histories = setting
    from dataclasses import replace

    with pytest.raises(HoldoutRefused) as refused:
        _run(setting, protocol=replace(protocol, model_id="0" * 64))
    assert refused.value.reason is Refusal.MODEL
    with pytest.raises(HoldoutRefused) as refused:
        _run(setting, histories={t: b for t, b in histories.items() if t != "MU"})
    assert refused.value.reason is Refusal.INCOMPLETE
    short = {**histories, "MU": histories["MU"].iloc[:-1]}
    with pytest.raises(HoldoutRefused) as refused:
        _run(setting, histories=short)
    assert refused.value.reason is Refusal.INTEGRITY


def _deciding(**change) -> dict:
    row = {"p_one_sided": 0.01, "status": "ok", "blocks": 3, "block_mean_diff": -0.001,
           "df": 2, "reason": None}  # fmt: skip
    return {**row, **change}


def test_the_answer_needs_every_block_an_available_test_a_negative_mean_and_p_below_alpha(setting):
    _, protocol, _ = setting
    assert evaluate._decision(_deciding(), protocol, "B1")["beats_comparator"] is True
    for change, status in (
        ({"block_mean_diff": 0.001, "p_one_sided": 0.99}, "ok"),
        ({"block_mean_diff": 0.0}, "ok"),
        ({"p_one_sided": 0.025}, "ok"),  # not below alpha
        ({"p_one_sided": float("nan")}, "ok"),
        ({"status": "unavailable", "reason": "zero_variance"}, "unavailable"),
        ({"blocks": 2, "df": 1}, evaluate.INCOMPLETE_BLOCKS),  # a block lost its dates
    ):
        found = evaluate._decision(_deciding(**change), protocol, "B1")
        assert found["beats_comparator"] is False and found["status"] == status, change


def test_the_same_inputs_give_the_same_record(setting, record):
    assert _run(setting)["record_id"] == record["record_id"]


def test_a_model_compared_with_itself_is_refused(setting):
    from dataclasses import replace

    model, _, _ = setting
    with pytest.raises(HoldoutRefused) as refused:
        _run(setting, model=replace(model, estimator="B0"))
    assert refused.value.reason is Refusal.MODEL


def test_a_forecast_that_is_not_finite_stops_the_run(setting, monkeypatch):
    original = registry.probabilities

    def broken(panel, ticker, model_, estimator=None):
        found = original(panel, ticker, model_, estimator)
        return registry.Probabilities(
            found.daily * np.nan, found.cumulative, found.usual, found.row
        )

    monkeypatch.setattr(registry, "probabilities", broken)
    with pytest.raises(HoldoutRefused) as refused:
        _run(setting)
    assert refused.value.reason is Refusal.INTEGRITY


def test_the_block_means_are_reported(record):
    deciding = next(row for row in record["comparisons"] if row["deciding"])
    sizes = [block["dates"] for block in deciding["by_block"]]
    assert sum(sizes) == deciding["dates"] == 29
    means = [block["mean_diff"] for block in deciding["by_block"]]
    assert deciding["block_mean_diff"] == pytest.approx(np.mean(means), abs=1e-15)
