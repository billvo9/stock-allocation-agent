"""The coverage and corporate-action audit (stock_agent.stock_lab.audit) and its use in the fetch script."""

from __future__ import annotations

from stock_agent.stock_lab.audit import audit_bars, audit_lines
from stock_agent.stock_lab.prices import on_calendar
from stock_agent.stock_lab.synthetic import demo_bars


def test_clean_bars_with_a_real_split_are_ready():
    bars = demo_bars("DEMO_SWING")  # a 3-for-1 split the raw closes show
    report = audit_bars("DEMO_SWING", bars)
    assert report["ready"] and report["issues"] == []
    (split,) = report["splits"]
    assert split["factor"] == 3.0 and split["verdict"] == "consistent"
    assert (
        abs(split["raw_close_ratio"] - 1 / 3) < 0.05
        and abs(split["split_consistent_ratio"] - 1) < 0.05
    )
    assert (
        report["dividends"]["ex_dividend_sessions"] > 0
        and report["dividends"]["largest_yield"] < 0.01
    )
    lines = audit_lines(report)
    assert "-> consistent" in lines[1] and lines[-1].endswith("ready to store")
    assert not any(f"{bars['close'].iloc[-1]:.4f}" in line for line in lines)  # no prices printed


def test_a_split_the_closes_do_not_show_and_bad_bars_need_review():
    bars = demo_bars("DEMO_CALM").iloc[:300].copy()
    bars.loc[100, "split_factor"] = 2.0  # reported, but the closes did not halve
    bars.loc[150, "low"] = bars.loc[150, "close"] * 1.01  # a low above the close
    bars.loc[160, "volume"] = 0.0
    report = audit_bars("X", bars)
    assert not report["ready"]
    assert report["splits"][0]["verdict"] == "mismatch"
    assert report["sanity"] == {
        "bracket_violations": 1,
        "bracket_first": [bars.loc[150, "session"].strftime("%Y-%m-%d")],
        "zero_volume_sessions": 1,
    }
    assert "split_price_mismatch" in report["returns"]["unavailable_by_reason"]
    assert any(line.startswith("  audit: review") for line in audit_lines(report))


def test_missing_sessions_are_counted_on_the_calendar():
    bars = demo_bars("DEMO_CALM").iloc[:40]
    holed = on_calendar(bars.drop(index=[10, 11]), bars["session"])
    report = audit_bars("X", holed)
    assert report["coverage"]["sessions_missing"] == 2
    assert report["coverage"]["missing_first"] == [
        d.strftime("%Y-%m-%d") for d in bars["session"].iloc[[10, 11]]
    ]
    assert report["ready"]  # a gap is reported, not a reason to refuse


def test_a_large_move_without_a_split_is_flagged_for_review():
    bars = demo_bars("DEMO_CALM").iloc[:60].copy()
    bars.loc[30:, ["open", "high", "low", "close"]] *= 0.4
    report = audit_bars("X", bars)
    assert report["moves"]["large_moves"] == 1 and not report["ready"]


def test_no_close_at_all_is_not_ready():
    bars = demo_bars("DEMO_CALM").iloc[:5].copy()
    bars["close"] = float("nan")
    assert audit_bars("X", bars) == {"ticker": "X", "ready": False, "issues": ["no closes served"]}
