"""
Simulation study: zero realized volatility under tick discretization (SHV-like).

SIMULATION ONLY: never development-data evidence and never evidence about any
fund. It is the study behind docs/research/zero_vol_floor.md, section 5, and
the acceptance thresholds there (run by the lead session on 2026-10-17 and
committed unchanged except for formatting, this header and the optional
`seeds` and `names` arguments of `main`). It reads no data and contacts no
network.

    .venv/bin/python scripts/research/zero_vol_simulation.py <output directory>
    .venv/bin/python scripts/research/zero_vol_simulation_tables.py <output directory>

The defaults (10 seeds, every scenario) reproduce sim_results_by_seed.csv,
sim_summary.csv and sim_run.json (about 15 s).

Latent process (per session t):
    g_lat_t = mu_d + sigma * eps_t,  eps ~ N(0,1) iid              latent daily log total return
    P_t     = P_{t-1} exp(g_lat_t) - D_t                            latent (unrounded) price
    D_t     = round(base * (exp(mu_d * s) - 1), 4) on every 21st session in distribution regimes
              (s = sessions since the last ex-date, base = latent price after the last ex-date)
Observed (F5.1-style, one-session returns only):
    C_t = round(P_t, 2)                                             $0.01 tick
    g_t = log((C_t + D_t) / C_{t-1})                                D_t as paid, not tick-rounded
Target window of as-of row t: returns t+L+1 .. t+L+h (L = entry lag = 1).
    sigma_raw = sqrt(mean g^2);  y_lat = 0.5 log(mean g_lat^2) (same functional, unrounded)
Rules:
    A   train/calibrate on y_raw finite rows; score every test row, zero windows = miss
    C   train/calibrate on y_raw finite rows; score only non-zero test rows
    B1  y = log(max(sigma_raw, f1)), f1 = 0.5 log(1 + tau/P~)        per-session floor
    B2  y = log(max(sigma_raw, f1 / sqrt(h)))                        sqrt(h)-adjusted floor
    B1_F1    B1 with the same floor on the log_rv / sigma_60 features (sensitivity)
    B1_asof  B1 with P~ = the as-of close of the row instead of the training median
    B2_asof  B2 with P~ = the as-of close of the row
P~ = median observed close over the training segment (never calibration or test closes).
Forecaster (identical spec under every rule): ridge on the code's five features (tsmom_20d,
tsmom_60d, log_rv_5d, log_rv_20d, log_rv_60d), training-median imputation, training
standardization, alpha = 0.1 * n_fit (fixed, no tuning). Bands: split conformal on a separate
calibration segment, the code's finite-sample ranks, q = 0.1 / 0.9. Purge: a row is in a
segment only if its label ends before the next segment starts.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

TAU = 0.01
L = 1
HS = (5, 20, 30)
PENALTY = 0.1
QLO, QHI = 0.1, 0.9
SEEDS = tuple(range(10))
N = 6000
CAL_START, TEST_START = 3000, 4500
BURN = 60
RULES = ("A", "C", "B1", "B2", "B1_F1", "B1_asof", "B2_asof")


def wsum(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= k:
        out[k - 1 :] = swv(a, k).sum(axis=1)
    return out


def regime_arrays(n, segments):
    """segments: list of (length, sigma_bp, mu_annual, distributions)."""
    sigma = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    dist = np.zeros(n + 1, dtype=bool)
    t = 1
    for length, s_bp, mu_ann, d in segments:
        sl = slice(t, min(t + length, n + 1))
        sigma[sl] = s_bp * 1e-4
        mu[sl] = math.log1p(mu_ann) / 252
        dist[sl] = d
        t += length
    assert t >= n + 1, (t, n)
    return sigma, mu, dist


def simulate(seed, n, segments, p0, kappa=0.0, sv_sd=0.0):
    """kappa > 0: anchored (mean-reverting) latent log price around a centre drawn uniformly
    within +-half a tick of p0 (T-bill-fund-like); no drift or distributions then."""
    rng = np.random.default_rng(seed)
    sigma, mu, dist = regime_arrays(n, segments)
    if sv_sd > 0:  # persistent stochastic volatility: log sigma AR(1), phi = 0.99
        phi = 0.99
        z = rng.standard_normal(n + 1) * sv_sd * math.sqrt(1 - phi**2)
        state = np.zeros(n + 1)
        state[0] = rng.standard_normal() * sv_sd
        for t in range(1, n + 1):
            state[t] = phi * state[t - 1] + z[t]
        sigma = sigma * np.exp(state - 0.5 * sv_sd**2)
    shocks = sigma * rng.standard_normal(n + 1)
    if kappa > 0:
        assert not dist.any() and not mu.any()
        centre = p0 + rng.uniform(-0.005, 0.005)
        x = np.zeros(n + 1)
        for t in range(1, n + 1):
            x[t] = (1.0 - kappa) * x[t - 1] + shocks[t]
        price = centre * np.exp(x)
        g_lat = np.full(n + 1, np.nan)
        g_lat[1:] = np.diff(x)
        close = np.rint(price * 100.0) / 100.0
        cash = np.zeros(n + 1)
        g = np.full(n + 1, np.nan)
        g[1:] = np.log1p(close[1:] / close[:-1] - 1.0)
        return g, g_lat, close, cash
    g_lat = mu + shocks
    g_lat[0] = np.nan
    price = np.empty(n + 1)
    price[0] = p0
    cash = np.zeros(n + 1)
    since, base = 0, p0
    for t in range(1, n + 1):
        cum = price[t - 1] * math.exp(g_lat[t])
        since += 1
        if dist[t] and since >= 21:
            d = round(base * (math.exp(mu[t] * since) - 1.0), 4)
            cash[t] = d
            price[t] = cum - d
            base, since = price[t], 0
        else:
            price[t] = cum
    close = np.rint(price * 100.0) / 100.0
    g = np.full(n + 1, np.nan)
    g[1:] = np.log1p((close[1:] + cash[1:]) / close[:-1] - 1.0)
    return g, g_lat, close, cash


def features(g, floor=None):
    g2 = g**2
    ms60 = wsum(g2, 60) / 60
    vol60 = np.sqrt(ms60)
    if floor is not None:
        vol60 = np.maximum(vol60, floor)
    cols = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in (20, 60):
            cols.append(np.where(vol60 > 0, wsum(g, k) / (vol60 * np.sqrt(k)), np.nan))
        for k in (5, 20, 60):
            ms = wsum(g2, k) / k
            if floor is None:
                cols.append(np.where(ms > 0, 0.5 * np.log(ms), np.nan))
            else:
                cols.append(np.log(np.maximum(np.sqrt(ms), floor)))
    return np.column_stack(cols)


def fit_ridge(X, y):
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(X), med, X)
    mean = filled.mean(axis=0)
    sd = filled.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (filled - mean) / sd
    alpha = PENALTY * len(y)
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"med": med, "mean": mean, "sd": sd, "beta": beta, "b0": float(y.mean())}


def predict(m, X):
    filled = np.where(np.isnan(X), m["med"], X)
    return m["b0"] + (((filled - m["mean"]) / m["sd"]) * m["beta"]).sum(axis=1)


def conformal(u):
    u = np.sort(u)
    n = len(u)
    lo, hi = math.floor((n + 1) * QLO), math.ceil((n + 1) * QHI)
    if n == 0 or lo < 1 or hi > n:
        return None
    return float(u[lo - 1]), float(u[hi - 1])


def qlike(label, forecast):
    d = label - forecast
    return np.exp(2.0 * d) - 2.0 * d - 1.0


def run_path(scenario, seed, n, segments, p0, cal_start, test_start, kappa=0.0, sv_sd=0.0):
    g, g_lat, close, _ = simulate(seed, n, segments, p0, kappa, sv_sd)
    p_tilde = float(np.median(close[:cal_start]))  # training-segment closes only
    f1 = 0.5 * math.log1p(TAU / p_tilde)
    X0 = features(g)
    X1 = features(g, floor=f1)
    nonzero_flag = (g != 0) & np.isfinite(g)
    out = []
    for h in HS:
        ms_end = wsum(g**2, h) / h
        nz_end = wsum(nonzero_flag.astype(float), h)
        lat_end = wsum(g_lat**2, h) / h
        rows = np.arange(BURN, n - L - h + 1)
        x = rows + L + h
        ms = ms_end[x]
        raw = np.sqrt(ms)
        zero = nz_end[x] == 0
        with np.errstate(divide="ignore"):
            y_raw = 0.5 * np.log(ms)
        y_lat = 0.5 * np.log(lat_end[x])
        f1_asof = 0.5 * np.log1p(TAU / close[rows])
        y = {
            "raw": y_raw,
            "B1": np.log(np.maximum(raw, f1)),
            "B2": np.log(np.maximum(raw, f1 / math.sqrt(h))),
            "B1_asof": np.log(np.maximum(raw, f1_asof)),
            "B2_asof": np.log(np.maximum(raw, f1_asof / math.sqrt(h))),
        }
        floored1 = raw < f1
        floored2 = raw < f1 / math.sqrt(h)
        train = (rows + L + h) < cal_start
        cal = (rows >= cal_start) & ((rows + L + h) < test_start)
        test = rows >= test_start
        nzr = ~zero
        base = {
            "scenario": scenario,
            "seed": seed,
            "h": h,
            "p_tilde": p_tilde,
            "floor_bp": f1 * 1e4,
            "zero_rate_all": zero.mean(),
            "zero_rate_test": zero[test].mean(),
            "floored_B1_rate_all": floored1.mean(),
            "floored_B2_rate_all": floored2.mean(),
            "floored_B1_not_zero_all": (floored1 & nzr).mean(),
            "floored_B2_not_zero_all": (floored2 & nzr).mean(),
            # target bias vs the latent window log volatility (whole path)
            "bias_raw_nonzero": np.mean(y_raw[nzr] - y_lat[nzr]) if nzr.any() else np.nan,
            "bias_B1_all": np.mean(y["B1"] - y_lat),
            "bias_B2_all": np.mean(y["B2"] - y_lat),
            "bias_B1_zero": np.mean(y["B1"][zero] - y_lat[zero]) if zero.any() else np.nan,
            "bias_B2_zero": np.mean(y["B2"][zero] - y_lat[zero]) if zero.any() else np.nan,
            "bias_B1_floored_nonzero": (
                np.mean(y["B1"][floored1 & nzr] - y_lat[floored1 & nzr])
                if (floored1 & nzr).any()
                else np.nan
            ),
            "selection_lat_nonzero_minus_all": (
                np.mean(y_lat[nzr]) - np.mean(y_lat) if nzr.any() else np.nan
            ),
            "labelmean_minus_latmean_C": (
                np.mean(y_raw[nzr]) - np.mean(y_lat) if nzr.any() else np.nan
            ),
            "labelmean_minus_latmean_B1": np.mean(y["B1"]) - np.mean(y_lat),
            "labelmean_minus_latmean_B2": np.mean(y["B2"]) - np.mean(y_lat),
            "y_raw_min_nonzero": np.min(y_raw[nzr]) if nzr.any() else np.nan,
            "min_raw_minus_lat_nonzero": (np.min(y_raw[nzr] - y_lat[nzr]) if nzr.any() else np.nan),
        }
        Xr0 = X0[rows]
        Xr1 = X1[rows]
        for rule in RULES:
            if rule in ("A", "C"):
                label, X, keep = y_raw, Xr0, nzr
            elif rule == "B1_F1":
                label, X, keep = y["B1"], Xr1, np.ones_like(nzr)
            else:
                label, X, keep = y[rule], Xr0, np.ones_like(nzr)
            tr, ca = train & keep, cal & keep
            rec = {**base, "rule": rule, "n_train": int(tr.sum()), "n_cal": int(ca.sum())}
            if tr.sum() < 50 or ca.sum() < 50:
                rec["status"] = "too_few_rows"
                out.append(rec)
                continue
            model = fit_ridge(X[tr], label[tr])
            null = float(label[tr].mean())
            pred = predict(model, X)
            q = conformal(label[ca] - pred[ca])
            if q is None:
                rec["status"] = "no_quantiles"
                out.append(rec)
                continue
            lo, hi = pred + q[0], pred + q[1]
            inside = (label >= lo) & (label <= hi)  # -inf labels are outside
            scored = test & (nzr if rule == "C" else np.ones_like(nzr))
            own = test & keep  # finite labels in the rule's own population
            rec.update(
                status="ok",
                n_test_scored=int(scored.sum()),
                coverage=inside[scored].mean(),
                coverage_zero_rows=inside[test & zero].mean() if (test & zero).any() else np.nan,
                coverage_nonzero_rows=inside[test & nzr].mean(),
                coverage_latent=((y_lat >= lo) & (y_lat <= hi))[test].mean(),
                band_width=q[1] - q[0],
                band_low_q=q[0],
                band_high_q=q[1],
                forecast_bias_vs_latent=np.mean(pred[test] - y_lat[test]),
                qlike_vs_latent=np.mean(qlike(y_lat[test], pred[test])),
                qlike_null_vs_latent=np.mean(qlike(y_lat[test], np.full(test.sum(), null))),
                qlike_own=np.mean(qlike(label[own], pred[own])),
                qlike_own_null=np.mean(qlike(label[own], np.full(own.sum(), null))),
                mse_vs_latent=np.mean((pred[test] - y_lat[test]) ** 2),
            )
            out.append(rec)
    return out


def scenarios():
    specs = {}
    for p0 in (110.30, 50.00):
        for mu in (0.0, 0.005, 0.04):
            for s in (0.25, 0.5, 1.0, 2.0, 5.0):
                name = f"P{p0:.2f}_mu{mu * 100:.1f}pct_s{s:g}bp"
                specs[name] = {
                    "p0": p0,
                    "segments": [(N, s, mu, mu > 0)],
                    "n": N,
                    "cal_start": CAL_START,
                    "test_start": TEST_START,
                    "mu": mu,
                    "sigma_bp": s,
                    "dynamics": "rw",
                    "kappa": 0.0,
                    "sv_sd": 0.0,
                }
        # Anchored zero-rate variants (robustness): mean-reverting latent log price.
        for kappa in (0.05, 0.2):
            for s in (0.25, 0.5, 1.0):
                name = f"P{p0:.2f}_mu0.0pct_s{s:g}bp_ou{kappa:g}"
                specs[name] = {
                    "p0": p0,
                    "segments": [(N, s, 0.0, False)],
                    "n": N,
                    "cal_start": CAL_START,
                    "test_start": TEST_START,
                    "mu": 0.0,
                    "sigma_bp": s,
                    "dynamics": f"ou{kappa:g}",
                    "kappa": kappa,
                    "sv_sd": 0.0,
                }
        # Stochastic-volatility variants: zero windows cluster in predictable low-vol states.
        for mu in (0.0, 0.04):
            for s in (0.25, 0.5, 1.0):
                name = f"P{p0:.2f}_mu{mu * 100:.1f}pct_s{s:g}bp_sv"
                specs[name] = {
                    "p0": p0,
                    "segments": [(N, s, mu, mu > 0)],
                    "n": N,
                    "cal_start": CAL_START,
                    "test_start": TEST_START,
                    "mu": mu,
                    "sigma_bp": s,
                    "dynamics": "sv0.5",
                    "kappa": 0.0,
                    "sv_sd": 0.5,
                }
    # SHV-history-like regime path (illustrative): ZIRP 2010-15, hikes 2016-19, ZIRP 2020-21,
    # 4.5% 2022-24. Train ~2010-17, calibrate ~2018-21, test ~2022-24.
    hist = [
        (1500, 0.5, 0.0, False),
        (1000, 1.0, 0.015, True),
        (500, 1.0, 0.0, False),
        (760, 2.0, 0.045, True),
    ]
    specs["P110.30_history_like"] = {
        "p0": 110.30,
        "segments": hist,
        "n": 3750,
        "cal_start": 2000,
        "test_start": 3000,
        "mu": -1.0,
        "sigma_bp": -1.0,
        "dynamics": "history_like",
        "kappa": 0.0,
        "sv_sd": 0.0,
    }
    return specs


def main(outdir, seeds=SEEDS, names=None):
    """Every scenario (or those in `names`) for every seed; writes three files to `outdir`."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    specs = {k: v for k, v in scenarios().items() if names is None or k in names}
    for name, spec in specs.items():
        for seed in seeds:
            records += run_path(
                name,
                seed,
                spec["n"],
                spec["segments"],
                spec["p0"],
                spec["cal_start"],
                spec["test_start"],
                spec["kappa"],
                spec["sv_sd"],
            )
    frame = pd.DataFrame(records)
    meta = pd.DataFrame(
        [
            {
                "scenario": k,
                **{x: v for x, v in s.items() if x != "segments"},
                "segments": json.dumps(s["segments"]),
            }
            for k, s in specs.items()
        ]
    )
    frame = frame.merge(meta, on="scenario", how="left")
    frame.to_csv(out / "sim_results_by_seed.csv", index=False)
    numeric = [c for c in frame.columns if frame[c].dtype.kind in "fi" and c not in ("seed",)]
    group = ["scenario", "h", "rule"]
    agg = frame.groupby(group)[numeric].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "sim_summary.csv", index=False)
    (out / "sim_run.json").write_text(
        json.dumps(
            {
                "seeds": list(seeds),
                "scenarios": sorted(specs),
                "n_sessions": N,
                "cal_start": CAL_START,
                "test_start": TEST_START,
                "penalty": PENALTY,
                "entry_lag": L,
                "horizons": list(HS),
                "rules": list(RULES),
                "elapsed_seconds": round(time.time() - started, 1),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            indent=1,
        )
    )
    print("rows", len(frame), "elapsed", round(time.time() - started, 1))


if __name__ == "__main__":
    main(sys.argv[1])
