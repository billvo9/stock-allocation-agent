"""
Reconstructed forecasts: the record and the store's explicit path (owner
decision of 2026-10-30, replacing decision A4's "missed sessions are never
backfilled" for the activated model).

A forecast reconstructed for a missed evening is a schema 1.2 record with
reconstructed=true and its true created_at (the time it is made, after its
evening closed). The store accepts one only through record_reconstruction:
never backdated, never before the evening closed, never beside a live
forecast or another reconstruction of the same (stock, session). The live
path (record_forecast) refuses it, and keeps refusing every late forecast.

Real-classified fixtures on a weekday calendar and an injected clock; no
network, nothing real.
"""

from __future__ import annotations

import pandas as pd
import pytest
import test_stock_lab_evening as base
from stock_lab_inference_helpers import AS_OF, CALENDAR_ID, next_day, ny, weekdays

from stock_agent.stock_lab import evening, forecast
from stock_agent.stock_lab.frozen_forecast import forecast_record
from stock_agent.stock_lab.store import CLOCK_SKEW, BackdatingRefused, StoreRefused, stamp

frozen, lab = base.frozen, base.lab  # the evening tests' frozen model and stores
TICKERS, EXPECTED = base.TICKERS, base.EXPECTED
CLOSED = ny(next_day(AS_OF), "09:30")  # the evening of AS_OF closes at the next open
LATER = ny(next_day(AS_OF, 2), "21:15")  # two evenings later: when the reconstruction is made


def _artifact(lab, *, made: pd.Timestamp, reconstructed: bool, ticker: str = "AMD") -> dict:
    """The forecast of `ticker` as of AS_OF from its 20:40 window, made at `made` (no write)."""

    research, inference, model, _ = lab
    found = evening.forecast_watchlist(
        research, model, tickers=(ticker,), now=ny(AS_OF, "20:40"), expected=EXPECTED,
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path, record=False,
    )  # fmt: skip
    live = found.results[0].forecast
    return forecast_record(
        found.panel, ticker, model, target_sessions=live["target_sessions"], created_at=made,
        data=live["data"], data_class="real", reconstructed=reconstructed,
    )  # fmt: skip


# --- the record ----------------------------------------------------------------------------


def test_a_live_forecast_stays_schema_1_1_without_the_flag(lab):
    live = _artifact(lab, made=ny(AS_OF, "20:40"), reconstructed=False)
    assert live["schema_version"] == "1.1" and "reconstructed" not in live
    assert forecast.is_reconstructed(live) is False


def test_a_reconstruction_is_schema_1_2_and_flagged(lab):
    record = _artifact(lab, made=LATER, reconstructed=True)
    assert record["schema_version"] == "1.2" and record["reconstructed"] is True
    assert forecast.is_reconstructed(record) is True
    assert forecast.check(record) == record


def test_a_reconstruction_is_made_only_after_its_evening_closed(lab):
    with pytest.raises(forecast.ForecastRefused, match="after its evening closed"):
        _artifact(lab, made=CLOSED - pd.Timedelta(microseconds=1), reconstructed=True)
    assert forecast.check(_artifact(lab, made=CLOSED, reconstructed=True))


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_the_flag_of_a_1_2_record_is_exactly_true(lab, value):
    record = _artifact(lab, made=LATER, reconstructed=True)
    record["reconstructed"] = value
    record["forecast_id"] = forecast.forecast_id(record)
    with pytest.raises(forecast.ForecastRefused, match="reconstructed"):
        forecast.check(record)


def test_a_1_1_record_cannot_carry_the_flag(lab):
    record = _artifact(lab, made=ny(AS_OF, "20:40"), reconstructed=False)
    record["reconstructed"] = True
    record["forecast_id"] = forecast.forecast_id(record)
    with pytest.raises(forecast.ForecastRefused, match="unexpected or missing fields"):
        forecast.check(record)


def test_the_frozen_forecast_states_its_reconstruction_in_its_limitations(lab):
    live = _artifact(lab, made=ny(AS_OF, "20:40"), reconstructed=False)
    again = _artifact(lab, made=LATER, reconstructed=True)
    assert again["reconstructed"] is True and again["daily"] == live["daily"]
    assert again["limitations"][0].startswith("Reconstructed after the fact")
    assert again["limitations"][1:] == live["limitations"]


# --- the store -----------------------------------------------------------------------------


@pytest.fixture
def opened(lab):
    """The prospective record opened: NVDA's live forecast as of AS_OF, recorded at 20:40."""

    research, _, _, clock = lab
    live = _artifact(lab, made=ny(AS_OF, "20:40"), reconstructed=False, ticker="NVDA")
    clock.now = ny(AS_OF, "20:40")
    research.record_forecast(live, real_authorized=True)
    return lab


def test_the_store_records_a_reconstruction_with_its_true_time(opened):
    research, _, _, clock = lab = opened
    record = _artifact(lab, made=LATER, reconstructed=True)
    clock.now = LATER + pd.Timedelta(seconds=3)
    assert research.record_reconstruction(record, real_authorized=True) == record["forecast_id"]
    (stored,) = research.forecasts("AMD", reconstructed=True)
    assert stored == record and research.verify_ledger()[1] is None
    assert research.forecast_recorded_at("AMD")[record["forecast_id"]] == stamp(clock.now)


def test_a_reconstruction_is_never_live(opened):
    research, _, _, clock = lab = opened
    record = _artifact(lab, made=LATER, reconstructed=True)
    clock.now = LATER
    research.record_reconstruction(record, real_authorized=True)
    assert research.live_forecast("AMD", AS_OF) is None
    assert research.latest_forecast("AMD") is None
    assert research.forecasts("AMD") == ()  # the live track: no live reader ever sees it
    assert research.forecasts("AMD", reconstructed=True) == (record,)


def test_the_live_path_refuses_a_reconstruction_and_every_late_forecast(opened):
    research, _, _, clock = lab = opened
    clock.now = LATER
    with pytest.raises(StoreRefused, match="only through record_reconstruction"):
        research.record_forecast(
            _artifact(lab, made=LATER, reconstructed=True), real_authorized=True
        )
    late = _artifact(lab, made=LATER, reconstructed=False)  # a live record made too late
    with pytest.raises(StoreRefused, match="before its first target session opens"):
        research.record_forecast(late, real_authorized=True)
    with pytest.raises(StoreRefused, match="reconstructed"):
        research.record_reconstruction(late, real_authorized=True)
    assert research.forecasts("AMD") == ()


@pytest.mark.parametrize(
    "offset, refusal",
    [
        (CLOCK_SKEW + pd.Timedelta(seconds=1), BackdatingRefused),  # made, then written later
        (-CLOCK_SKEW - pd.Timedelta(seconds=1), StoreRefused),  # claims a time to come
    ],
    ids=["backdated", "future"],
)
def test_a_reconstruction_is_never_backdated_or_postdated(opened, offset, refusal):
    research, _, _, clock = lab = opened
    record = _artifact(lab, made=LATER, reconstructed=True)
    clock.now = LATER + offset
    with pytest.raises(refusal):
        research.record_reconstruction(record, real_authorized=True)
    assert research.forecasts("AMD") == ()


def test_one_reconstruction_per_stock_and_session_and_never_beside_a_live_forecast(opened):
    research, _, _, clock = lab = opened
    first = _artifact(lab, made=LATER, reconstructed=True)
    again = _artifact(lab, made=LATER + pd.Timedelta(minutes=1), reconstructed=True)
    mu_live = _artifact(lab, made=ny(AS_OF, "20:40"), reconstructed=False, ticker="MU")
    mu_rebuilt = _artifact(lab, made=LATER, reconstructed=True, ticker="MU")
    clock.now = LATER
    research.record_reconstruction(first, real_authorized=True)
    clock.now = LATER + pd.Timedelta(minutes=1)
    with pytest.raises(StoreRefused, match="already has a forecast as of"):
        research.record_reconstruction(again, real_authorized=True)
    # MU was recorded live that evening: its session is never reconstructed beside it.
    clock.now = ny(AS_OF, "20:40")
    research.record_forecast(mu_live, real_authorized=True)
    clock.now = LATER
    with pytest.raises(StoreRefused, match="already has a forecast as of"):
        research.record_reconstruction(mu_rebuilt, real_authorized=True)
    assert research.forecasts("AMD", reconstructed=True) == (first,)
    assert (
        research.forecasts("MU") == (mu_live,)
        and research.forecasts("MU", reconstructed=True) == ()
    )


def test_only_real_reconstructions_and_never_a_supersession(opened):
    research, _, _, clock = lab = opened
    clock.now = LATER
    record = _artifact(lab, made=LATER, reconstructed=True)
    record["supersedes"] = "0" * 64
    record["forecast_id"] = forecast.forecast_id(record)
    with pytest.raises(StoreRefused, match="supersedes nothing"):
        research.record_reconstruction(record, real_authorized=True)


def test_a_reconstruction_needs_the_callers_authorization(opened):
    research, _, _, clock = opened
    record = _artifact(opened, made=LATER, reconstructed=True)
    clock.now = LATER
    with pytest.raises(StoreRefused, match="decisions A2, A4 and 2026-10-30"):
        research.record_reconstruction(record)
    assert research.forecasts("AMD", reconstructed=True) == ()


def test_nothing_before_the_first_live_forecast_is_ever_reconstructed(lab):
    research, _, _, clock = lab
    record = _artifact(lab, made=LATER, reconstructed=True)
    clock.now = LATER
    with pytest.raises(StoreRefused, match="since the first live forecast only"):
        research.record_reconstruction(record, real_authorized=True)
    assert research.forecasts("AMD", reconstructed=True) == ()


def test_a_synthetic_record_is_never_a_reconstruction(lab):
    record = _artifact(lab, made=LATER, reconstructed=True)
    record["data_class"] = "synthetic"
    record["forecast_id"] = forecast.forecast_id(record)
    with pytest.raises(forecast.ForecastRefused, match="real forecast with reconstructed true"):
        forecast.check(record)


def test_a_live_forecast_never_supersedes_a_reconstruction(opened):
    # A reconstruction made at the open, then a "recalculation" stamped a moment before it and
    # written within the store's skew: refused, whatever the clock allows.
    research, _, _, clock = opened
    rebuilt = _artifact(opened, made=CLOSED, reconstructed=True)
    clock.now = CLOSED
    research.record_reconstruction(rebuilt, real_authorized=True)
    live = _artifact(opened, made=CLOSED - pd.Timedelta(seconds=1), reconstructed=False)
    live["supersedes"] = rebuilt["forecast_id"]
    live["forecast_id"] = forecast.forecast_id(live)
    clock.now = CLOSED + pd.Timedelta(seconds=30)
    with pytest.raises(StoreRefused, match="never supersedes a reconstruction"):
        research.record_forecast(live, real_authorized=True)
    assert research.forecasts("AMD") == ()


def test_the_stores_own_clock_must_be_past_the_open(opened):
    # created_at may run ahead of the store's clock by its skew; the evening must have closed
    # by the store's own clock too, or the on-time live forecast would be blocked.
    research, _, _, clock = opened
    record = _artifact(opened, made=CLOSED, reconstructed=True)
    clock.now = CLOSED - pd.Timedelta(minutes=1)
    with pytest.raises(StoreRefused, match="after its evening closed"):
        research.record_reconstruction(record, real_authorized=True)
    assert research.forecasts("AMD", reconstructed=True) == ()
