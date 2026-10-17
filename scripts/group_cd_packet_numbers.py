"""
Worked examples for the group C and group D financial decision packets.

Documentation only. This script computes every number quoted in
`docs/research/group_c_decisions.md` and `docs/research/group_d_decisions.md`
from stated synthetic inputs, so the packets can be re-checked by anyone.
It is not an implementation: no module under `src/` imports it, nothing in it
is approved, and it reads no stored data and makes no network request. It
reuses two existing pieces of the repository: the frozen NYSE calendar
(session counts) and the T2 circular block bootstrap (the interval example).

`tests/test_group_cd_packet_numbers.py` checks that every value below appears
in its packet and recomputes the key ones independently.

Run from the repository root:

    PYTHONPATH=src .venv/bin/python scripts/group_cd_packet_numbers.py
"""

from __future__ import annotations

import calendar
import math
from datetime import date
from itertools import pairwise

import numpy as np

from stock_agent.calendars import nyse
from stock_agent.model_diagnostics.inference import (
    bootstrap_mean,
    bootstrap_statistic,
    draw_circular_blocks,
    normal_cdf,
    normal_ppf,
    student_t_ppf,
)

SESSIONS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25
EULER_GAMMA = 0.5772156649015329
Z_TWO_SIDED_5 = normal_ppf(0.975)  # also one-sided 2.5%, the PRD's confirmatory alpha
Z_POWER_80 = normal_ppf(0.80)


def pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def bp(value: float, decimals: int = 2) -> str:
    return f"{value * 1e4:+.{decimals}f} bp"


def fixed(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def act365_accrual(annual_yield: float, start: date, end: date) -> float:
    """y x days / 365 (366 in a leap year), split at January 1 (proposed F3.1)."""

    accrual = 0.0
    current = start
    while current < end:
        segment_end = min(end, date(current.year + 1, 1, 1))
        days_in_year = 366 if calendar.isleap(current.year) else 365
        accrual += annual_yield * (segment_end - current).days / days_in_year
        current = segment_end
    return accrual


def session_dates(start: str, end: str) -> list[date]:
    return [ts.date() for ts in nyse.sessions(start, end)]


def per_session_cash_returns(annual_yield: float, start: str, end: str) -> np.ndarray:
    days = session_dates(start, end)
    return np.array([act365_accrual(annual_yield, a, b) for a, b in pairwise(days)])


def sharpe_from_sums(sums: dict[str, np.ndarray]) -> np.ndarray:
    """Annualized Sharpe of a series from its replicate sums (n, sum x, sum x^2)."""

    n = sums["n"]
    mean = sums["x"] / n
    variance = (sums["x2"] - n * mean**2) / (n - 1)
    return mean / np.sqrt(variance) * math.sqrt(SESSIONS_PER_YEAR)


def expected_max_of_normals(k: int) -> float:
    """E[max of k independent standard normals], by numerical integration."""

    grid = np.linspace(-10.0, 10.0, 200_001)
    pdf = np.exp(-0.5 * grid**2) / math.sqrt(2.0 * math.pi)
    cdf = np.array([normal_cdf(float(x)) for x in grid[::100]])
    cdf = np.interp(grid, grid[::100], cdf)
    integrand = grid * k * pdf * cdf ** (k - 1)
    return float(np.trapezoid(integrand, grid))


def binomial_upper_tail(k0: int, n: int, p: float) -> float:
    """P(X >= k0) for X ~ Binomial(n, p), summed exactly in log space."""

    log_terms = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log(1.0 - p)
        for k in range(k0, n + 1)
    )
    return sum(math.exp(term) for term in log_terms)


def deflated_sharpe(
    sr: float, n: int, skew: float, kurtosis: float, trials: int, trial_variance: float
) -> tuple[float, float]:
    """
    Bailey and Lopez de Prado (2014), per-session units. `kurtosis` is the
    ordinary (not excess) kurtosis. Returns (SR_0, DSR).
    """

    if trials == 1:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(trial_variance) * (
            (1.0 - EULER_GAMMA) * normal_ppf(1.0 - 1.0 / trials)
            + EULER_GAMMA * normal_ppf(1.0 - 1.0 / (trials * math.e))
        )
    denominator = math.sqrt(1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2)
    return sr0, normal_cdf((sr - sr0) * math.sqrt(n - 1) / denominator)


def group_c_numbers() -> dict[str, str]:
    out: dict[str, str] = {}

    # C1: the 91-day bill convention check.
    price, days = 98.75, 91
    hpr = 100.0 / price - 1.0
    investment_yield = (100.0 - price) / price * 365.0 / days
    discount_yield = (100.0 - price) / 100.0 * 360.0 / days
    bey_from_discount = 365.0 * discount_yield / (360.0 - discount_yield * days)
    out["C1.bill_hpr"] = pct(hpr, 4)
    out["C1.investment_yield"] = pct(investment_yield, 4)
    out["C1.discount_yield"] = pct(discount_yield, 4)
    out["C1.discount_as_investment"] = pct(discount_yield * days / 365.0, 4)
    out["C1.discount_as_investment_error"] = bp(discount_yield * days / 365.0 - hpr)
    out["C1.discount_as_investment_error_year"] = bp(4 * (discount_yield * days / 365.0 - hpr), 1)
    out["C1.investment_on_360_error"] = bp(investment_yield * days / 360.0 - hpr)
    out["C1.bey_from_discount"] = pct(bey_from_discount, 4)

    # C1: day counts at 5%.
    friday, monday = date(2023, 12, 15), date(2023, 12, 18)
    out["C1.fri_mon"] = fixed(act365_accrual(0.05, friday, monday), 9)
    out["C1.fri_mon_per_session_252"] = fixed(0.05 / SESSIONS_PER_YEAR, 9)
    out["C1.fri_mon_act360"] = fixed(0.05 * 3 / 360.0, 9)
    out["C1.year_crossing"] = fixed(act365_accrual(0.05, date(2023, 12, 29), date(2024, 1, 2)), 9)

    # C1: compounding at a constant 5% investment yield.
    daily = (1.0 + 0.05 / 365.0) ** 365 - 1.0
    quarterly_roll = (1.0 + 0.05 * 91.25 / 365.0) ** 4 - 1.0
    cash_2023 = per_session_cash_returns(0.05, "2022-12-30", "2023-12-29")
    out["C1.daily_compounding"] = pct(daily, 4)
    out["C1.quarterly_roll"] = pct(quarterly_roll, 4)
    out["C1.roll_91_day"] = pct((1.0 + 0.05 * 91 / 365.0) ** (365.0 / 91) - 1.0, 4)
    out["C1.daily_over_quarterly"] = bp(daily - quarterly_roll, 1)
    out["C1.sessions_2023"] = str(len(cash_2023))
    out["C1.cash_2023_growth"] = pct(float(np.prod(1.0 + cash_2023) - 1.0), 4)

    # C1: a short-Treasury fund instead of the account.
    out["C1.duration_loss_10bp"] = bp(-0.25 * 0.0010, 1)
    fund_year = (1.0 + 0.05 / 365.0) ** 365 * (1.0 - 0.0015 / 365.0) ** 365 - 1.0
    out["C1.fund_year_after_expense_ratio"] = pct(fund_year, 2)

    # C2: Columbus Day 2024 (NYSE open; bond market and Federal Reserve closed).
    if not nyse.is_session("2024-10-14"):
        raise AssertionError("2024-10-14 must be an NYSE session for the C2 example.")
    out["C2.columbus_fri_mon"] = fixed(
        act365_accrual(0.051, date(2024, 10, 11), date(2024, 10, 14)), 9
    )
    out["C2.columbus_mon_tue_carried"] = fixed(
        act365_accrual(0.051, date(2024, 10, 14), date(2024, 10, 15)), 9
    )
    all_sessions = session_dates("2003-01-02", "2024-12-31")
    out["C2.sessions_2003_2024"] = f"{len(all_sessions):,}"
    out["C2.sessions_before_alfred"] = f"{len(session_dates('2003-01-02', '2005-06-27')):,}"

    # C3: a 100% cash account in 2023 under rf = 0 versus rf = the same series.
    sharpe_rf0 = cash_2023.mean() / cash_2023.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)
    out["C3.cash_sharpe_rf0"] = fixed(sharpe_rf0, 1)
    # An all-cash portfolio's net return goes through (1 - C)(1 + w.r) - 1 with C = 0, so its
    # excess over the same rate is a floating-point residual, not an exact zero.
    residual = np.abs(((1.0 + cash_2023) - 1.0) - cash_2023).max()
    out["C3.cash_excess_residual"] = f"{residual:.1e}"

    # C4: half cash, half a short-Treasury fund, for one year at a constant 5%.
    half_d4a = 0.5 * fund_year
    half_c1 = 0.5 * fund_year + 0.5 * daily
    out["C4.half_cash_d4a"] = pct(half_d4a, 2)
    out["C4.half_cash_c1"] = pct(half_c1, 2)
    out["C4.half_cash_gap"] = f"{(half_c1 - half_d4a) * 100:.2f} percentage points"
    out["C4.shv_switch_saving"] = bp(0.20 * 0.0002, 1)

    # C5: total-return benchmarks.
    build_cost = 0.0005
    out["C5.benchmark_after_build"] = fixed(1.0 - build_cost, 4)
    out["C5.benchmark_year_with_build"] = pct((1.0 - build_cost) * 1.10 - 1.0, 3)
    total_return = 1.005**4 - 1.0
    out["C5.dividend_total_return"] = pct(total_return, 4)
    out["C5.strategy_vs_total_return"] = f"{(0.01 - total_return) * 100:.2f} pt"
    out["C5.fee_example"] = pct(1.10 * (1.0 - 0.0009) - 1.0, 3)

    # C7: 60/40 drift and rebalance; a benchmark under a 0.5 cap.
    drifted = 0.6 * 1.10 / (0.6 * 1.10 + 0.4 * 1.00)
    trade = drifted - 0.6
    out["C7.drifted_equity_weight"] = fixed(drifted, 4)
    out["C7.rebalance_turnover"] = fixed(trade, 4)
    out["C7.rebalance_cost"] = bp(0.0005 * 2 * trade, 3)
    capped = (1.0 - 0.5 * build_cost) * (0.5 * 1.10 + 0.5 * (1.0 + daily)) - 1.0
    out["C7.capped_benchmark"] = pct(capped, 2)

    # C8: the common window start (return sessions: one fewer than closes).
    n_2007 = len(session_dates("2007-01-05", "2024-12-31")) - 1
    n_2003_agg = len(session_dates("2003-09-22", "2024-12-31")) - 1
    out["C8.sessions_from_shv"] = f"{n_2007:,}"
    out["C8.sessions_from_agg"] = f"{n_2003_agg:,}"
    out["C8.mde_ratio"] = fixed(math.sqrt(n_2007 / n_2003_agg), 3)
    return out


def group_d_numbers() -> dict[str, str]:
    out: dict[str, str] = {}

    # M1: CAGR on calendar days.
    out["M1.cagr_calendar"] = fixed(1.10 ** (DAYS_PER_YEAR / 365) - 1.0, 7)
    out["M1.cagr_sessions"] = fixed(1.10 ** (SESSIONS_PER_YEAR / 248) - 1.0, 7)
    first, last = date(2021, 1, 4), date(2022, 12, 30)
    span = (last - first).days
    with_build = (0.9995 * 1.21) ** (DAYS_PER_YEAR / span) - 1.0
    without_build = 1.21 ** (DAYS_PER_YEAR / span) - 1.0
    out["M1.span_days"] = str(span)
    out["M1.cagr_with_build"] = pct(with_build, 3)
    out["M1.cagr_without_build"] = pct(without_build, 3)
    out["M1.half_year_annualized"] = pct(1.05 ** (DAYS_PER_YEAR / 182) - 1.0, 2)

    # M2: volatility and Sharpe.
    sr_session, n = 0.05, 2520
    sr_annual = sr_session * math.sqrt(SESSIONS_PER_YEAR)
    lo_se = math.sqrt((1.0 + 0.5 * sr_session**2) / n) * math.sqrt(SESSIONS_PER_YEAR)
    out["M2.sharpe_annual"] = fixed(sr_annual, 3)
    out["M2.lo_se"] = fixed(lo_se, 3)
    out["M2.iid_low"] = fixed(sr_annual - Z_TWO_SIDED_5 * lo_se, 2)
    out["M2.iid_high"] = fixed(sr_annual + Z_TWO_SIDED_5 * lo_se, 2)
    out["M2.iid_half_width"] = fixed(Z_TWO_SIDED_5 * lo_se, 2)
    weekend_rf = act365_accrual(0.05, date(2023, 12, 15), date(2023, 12, 18))
    out["M2.weekend_rf"] = pct(weekend_rf, 4)
    out["M2.weekend_excess"] = pct(0.006 - weekend_rf, 4)
    out["M2.root_250_vs_252"] = pct(math.sqrt(250 / 252) - 1.0, 2)

    # M3: the interval method on a seeded synthetic series (i.i.d. normal log returns).
    seed, n, block = 20261017, 1260, 63
    rng = np.random.RandomState(seed)
    log_x = 0.0003 + 0.01 * rng.standard_normal(n)
    common = rng.standard_normal(n)
    log_y = 0.0002 + 0.01 * (0.9 * (log_x - 0.0003) / 0.01 + math.sqrt(1 - 0.81) * common)
    days = session_dates("2015-01-02", "2024-12-31")[: n + 1]
    span_days = (days[-1] - days[0]).days
    draws = draw_circular_blocks(n, block_length=block, reps=999, seed=seed)
    k = draws.blocks_per_series
    t_crit = student_t_ppf(0.975, k - 1)
    growth = bootstrap_mean(log_x, draws)
    scale = n * DAYS_PER_YEAR / span_days

    def to_cagr(mean_log: float) -> float:
        return math.exp(mean_log * scale) - 1.0

    out["M3.blocks"] = str(k)
    out["M3.t_critical"] = fixed(t_crit, 3)
    out["M3.se_relative_noise"] = pct(1.0 / math.sqrt(2.0 * (k - 1)), 0)
    out["M3.span_days"] = f"{span_days:,}"
    out["M3.cagr_point"] = pct(to_cagr(growth.estimate), 2)
    out["M3.cagr_low"] = pct(to_cagr(growth.ci_low), 2)
    out["M3.cagr_high"] = pct(to_cagr(growth.ci_high), 2)
    out["M3.cagr_true"] = pct(to_cagr(0.0003), 2)

    simple_x = np.expm1(log_x)
    sums = {"n": np.ones(n), "x": simple_x, "x2": simple_x**2}
    sharpe = bootstrap_statistic(sums, sharpe_from_sums, draws, n=n)
    sharpe_session = sharpe.estimate / math.sqrt(SESSIONS_PER_YEAR)
    lo = math.sqrt((1.0 + 0.5 * sharpe_session**2) / n) * math.sqrt(SESSIONS_PER_YEAR)
    out["M3.sharpe_point"] = fixed(sharpe.estimate, 2)
    out["M3.sharpe_boot_se"] = fixed(sharpe.se, 3)
    out["M3.sharpe_t_low"] = fixed(sharpe.estimate - t_crit * sharpe.se, 2)
    out["M3.sharpe_t_high"] = fixed(sharpe.estimate + t_crit * sharpe.se, 2)
    out["M3.sharpe_pct_low"] = fixed(sharpe.ci_low, 2)
    out["M3.sharpe_pct_high"] = fixed(sharpe.ci_high, 2)
    out["M3.sharpe_lo_se"] = fixed(lo, 3)
    true_mean = math.exp(0.0003 + 0.5 * 0.01**2) - 1.0
    true_sd = math.sqrt((math.exp(0.01**2) - 1.0) * math.exp(2 * 0.0003 + 0.01**2))
    out["M3.sharpe_true"] = fixed(true_mean / true_sd * math.sqrt(SESSIONS_PER_YEAR), 2)

    paired = bootstrap_mean(log_x - log_y, draws)
    alone_y = bootstrap_mean(log_y, draws)
    # Annualized log growth on the CAGR's calendar scale: mean log growth x n x 365.25 / D.
    out["M3.sessions_per_calendar_year"] = fixed(scale, 1)
    out["M3.paired_diff"] = pct(paired.estimate * scale, 2)
    out["M3.paired_se"] = pct(paired.se * scale, 2)
    unpaired_se = math.sqrt(growth.se**2 + alone_y.se**2)
    out["M3.unpaired_se"] = pct(unpaired_se * scale, 2)
    out["M3.paired_low"] = pct((paired.estimate - t_crit * paired.se) * scale, 2)
    out["M3.paired_high"] = pct((paired.estimate + t_crit * paired.se) * scale, 2)
    out["M3.paired_true"] = pct((0.0003 - 0.0002) * scale, 2)
    out["M3.t_8_blocks"] = fixed(student_t_ppf(0.975, 7), 3)
    out["M3.t_72_blocks"] = fixed(student_t_ppf(0.975, 71), 3)

    # M4: Monte Carlo precision of a coverage estimate.
    for reps in (1000, 2000, 5000):
        half_width = Z_TWO_SIDED_5 * math.sqrt(0.95 * 0.05 / reps)
        out[f"M4.half_width_{reps}"] = f"{half_width * 100:.2f} pp"
    # Acceptance at 93% per cell, alone and jointly over the design's 120 cells per method and
    # block length (5 processes x 4 sample sizes x 6 statistics). Per cell: the exact binomial
    # tail. Jointly: the product, which assumes independent cells (they share simulated paths).
    cells = 5 * 4 * 6
    out["M4.cells"] = str(cells)
    for reps in (2000, 5000):
        threshold = -(-93 * reps // 100)
        for true_coverage in (0.95, 0.94, 0.92):
            per_cell = binomial_upper_tail(threshold, reps, true_coverage)
            label = f"{reps}_{round(true_coverage * 100)}"
            out[f"M4.cell_pass_{label}"] = pct(per_cell, 1)
            out[f"M4.all_pass_{label}"] = pct(per_cell**cells, 1)
            se = math.sqrt(true_coverage * (1 - true_coverage) / reps)
            normal = 1.0 - normal_cdf((0.93 - true_coverage) / se)
            out[f"M4.all_pass_normal_{label}"] = pct(normal**cells, 1)
    out["M4.bootstrap_statistics"] = f"{cells * 5000 * 999 * 3:,}"

    # M5: registered stress windows (peak and trough sessions of the S&P 500).
    windows = {
        "gfc": ("2007-10-09", "2009-03-09"),
        "covid": ("2020-02-19", "2020-03-23"),
        "rates_2022": ("2022-01-03", "2022-10-12"),
    }
    for name, (start, end) in windows.items():
        out[f"M5.sessions_{name}"] = str(len(session_dates(start, end)) - 1)

    # M6: turnover per calendar year (construction separate).
    rebalances = {2021: [0.10, 0.15], 2022: [0.20, 0.05, 0.10, 0.12], 2023: [0.08, 0.30]}
    rebalances[2024] = [0.10]
    per_year = {year: sum(values) for year, values in rebalances.items()}
    for year, total in per_year.items():
        out[f"M6.year_{year}"] = fixed(total, 2)
    out["M6.full_year_mean"] = fixed((per_year[2022] + per_year[2023]) / 2, 3)
    turnover_span = (date(2024, 6, 28) - date(2021, 3, 1)).days
    rebalance_total = sum(per_year.values())
    out["M6.span_days"] = f"{turnover_span:,}"
    out["M6.scaled_rebalance"] = fixed(rebalance_total * DAYS_PER_YEAR / turnover_span, 3)
    out["M6.scaled_with_build"] = fixed((rebalance_total + 1.0) * DAYS_PER_YEAR / turnover_span, 3)

    # M7: participation on traded notional.
    wealth, mdv = 1_000_000, 20_000_000
    out["M7.rebalance_q"] = pct(0.15 * wealth / mdv, 2)
    out["M7.holdings_q"] = pct(0.30 * wealth / mdv, 2)
    out["M7.build_q"] = pct(0.30 * wealth / mdv, 2)
    out["M7.q_at_100k"] = pct(0.15 * 100_000 / mdv, 3)
    out["M7.q_at_10m"] = pct(0.15 * 10_000_000 / mdv, 1)

    # M8: deflated Sharpe ratio (per-session units; trials assumed pure noise).
    n, sr_hat = 1260, 0.80 / math.sqrt(SESSIONS_PER_YEAR)
    for trials in (1, 2, 10, 100):
        sr0, dsr = deflated_sharpe(sr_hat, n, -0.5, 6.0, trials, 1.0 / n)
        out[f"M8.sr0_{trials}"] = fixed(sr0 * math.sqrt(SESSIONS_PER_YEAR), 2)
        out[f"M8.dsr_{trials}"] = fixed(dsr, 3)
    p_values = [0.010, 0.020, 0.030, 0.200]
    holm = [p <= 0.025 / (len(p_values) - i) for i, p in enumerate(sorted(p_values))]
    out["M8.holm_first_threshold"] = fixed(0.025 / len(p_values), 5)
    alpha = 0.025
    out["M8.dsr_threshold"] = fixed(1.0 - alpha, 3)
    out["M8.error_union_bound"] = fixed(2 * alpha, 3)
    out["M8.error_independent_bound"] = fixed(alpha + alpha**2, 6)
    out["M8.error_negative_dependence"] = fixed(1.5 * alpha, 4)
    out["M8.holm_rejections"] = str(next((i for i, ok in enumerate(holm) if not ok), len(holm)))
    out["M8.unadjusted_rejections"] = str(sum(p <= 0.025 for p in p_values))

    # M9: power and minimum sample size (i.i.d. planning approximations).
    z = Z_TWO_SIDED_5 + Z_POWER_80
    out["M9.z_sum"] = fixed(z, 4)
    for sr in (0.3, 0.5, 1.0):
        out[f"M9.years_sr_{sr}"] = fixed((z / sr) ** 2, 1)
    for rho in (0.5, 0.8, 0.9, 0.95):
        for years in (3, 5, 10, 18):
            blocks = years * SESSIONS_PER_YEAR // 63
            zt = student_t_ppf(0.975, blocks - 1) + student_t_ppf(0.80, blocks - 1)
            mde = zt * math.sqrt(2 * (1 - rho) / years)
            out[f"M9.mde_sr_rho{rho}_t{years}"] = fixed(mde, 2)
    for years in (3, 10):
        normal = z * math.sqrt(0.2 / years)
        out[f"M9.mde_sr_normal_rho0.9_t{years}"] = fixed(normal, 3)
        # The PRD's planning figures match 260 sessions a year annualized with sqrt(252).
        prd = z * math.sqrt(0.2 / (260 * years)) * math.sqrt(SESSIONS_PER_YEAR)
        out[f"M9.mde_sr_prd_260_rho0.9_t{years}"] = fixed(prd, 3)
    for te in (0.02, 0.05, 0.10):
        for years in (3, 5, 10, 18):
            blocks = years * SESSIONS_PER_YEAR // 63
            zt = student_t_ppf(0.975, blocks - 1) + student_t_ppf(0.80, blocks - 1)
            out[f"M9.mde_growth_te{te}_t{years}"] = pct(zt * te / math.sqrt(years), 1)
    out["M9.years_ir_0.5"] = fixed((z / 0.5) ** 2, 1)
    for k in (2, 5, 10, 20):
        out[f"M9.best_of_{k}"] = fixed(expected_max_of_normals(k), 3)
    test_window = session_dates("2020-01-02", "2024-12-31")
    out["M9.test_sessions"] = f"{len(test_window) - 1:,}"
    out["M9.test_blocks"] = str(-(-(len(test_window) - 1) // 63))
    return out


def main() -> None:
    for title, numbers in (("Group C", group_c_numbers()), ("Group D", group_d_numbers())):
        print(title)
        for key, value in numbers.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
