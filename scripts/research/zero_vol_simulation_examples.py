"""
Worked examples of the tick-based realized-volatility floor (SIMULATION ONLY research code
from the lead session's 2026-10-17 study; the repository's own worked examples come from
`scripts/check_zero_vol_floor.py examples`, on the production arithmetic).

    .venv/bin/python scripts/research/zero_vol_simulation_examples.py <output .json>

    sigma_floor        = 0.5 * log(1 + tau / P)           per-session floor (B1)
    sigma_floor_sqrt_h = 0.5 * log(1 + tau / P) / sqrt(h) sqrt(h)-adjusted floor (B2)
    sigma_raw          = sqrt(mean g^2)                   measured daily RMS of the h returns
    y                  = log(max(sigma_raw, floor))       = 0.5 log(max(mean g^2, floor^2))
    measured_zero      = every g == 0 exactly
    floored            = sigma_raw < floor
"""

import json
import math
import sys

TAU = 0.01


def g_close(prev, cur, dist=0.0):
    """Daily log total return from unadjusted closes and an as-paid distribution."""
    return math.log1p((cur + dist) / prev - 1.0)


def describe(name, p, closes, dists, h):
    g = [g_close(closes[i - 1], closes[i], dists[i]) for i in range(1, len(closes))]
    assert len(g) == h
    ms = sum(x * x for x in g) / h
    raw = math.sqrt(ms)
    floor1 = 0.5 * math.log1p(TAU / p)
    floor2 = floor1 / math.sqrt(h)
    out = {
        "example": name,
        "P_tilde": p,
        "h": h,
        "closes": closes,
        "distributions": dists[1:],
        "g": g,
        "mean_g2": ms,
        "sigma_raw_bp": raw * 1e4,
        "measured_zero": all(x == 0.0 for x in g),
        "y_raw": (math.log(raw) if raw > 0 else None),
    }
    for tag, fl in (("B1_per_session", floor1), ("B2_sqrt_h", floor2)):
        out[tag] = {
            "floor_bp": fl * 1e4,
            "log_floor": math.log(fl),
            "floored": raw < fl,
            "y": math.log(max(raw, fl)),
        }
    return out


def main(path):
    rows = []
    for p in (110.30, 50.00):
        up = round(p + TAU, 2)
        rows.append(describe("all_zero_5", p, [p] * 6, [0.0] * 6, 5))
        rows.append(describe("one_tick_5", p, [p, up, up, up, up, up], [0.0] * 6, 5))
        # One tick up on day 1, then a sub-tick cash distribution of $0.0030 on day 3 whose
        # ex-date price drop rounds to zero ticks (zero-rate regime).
        rows.append(
            describe(
                "one_tick_plus_subtick_distribution_5",
                p,
                [p, up, up, up, up, up],
                [0.0, 0.0, 0.0, 0.0030, 0.0, 0.0],
                5,
            )
        )
        # 4% regime: a monthly distribution of 110.30*0.04/12 ~= $0.3677 (scaled with P);
        # the ex-date close falls by the distribution rounded to the tick.
        d = round(p * 0.04 / 12, 4)
        ex_close = round(up - d, 2)
        rows.append(
            describe(
                "one_tick_plus_4pct_distribution_5",
                p,
                [p, up, up, ex_close, ex_close, ex_close],
                [0.0, 0.0, 0.0, d, 0.0, 0.0],
                5,
            )
        )
        # Smallest non-zero tick window at h = 20 and h = 30 (one tick move).
        for h in (20, 30):
            rows.append(describe(f"one_tick_{h}", p, [p] + [up] * h, [0.0] * (h + 1), h))
            rows.append(describe(f"all_zero_{h}", p, [p] * (h + 1), [0.0] * (h + 1), h))
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)
    for r in rows:
        print(
            f"{r['example']:40s} P={r['P_tilde']:7.2f} h={r['h']:2d} "
            f"sig_raw={r['sigma_raw_bp']:.6f}bp y_raw={r['y_raw']} zero={r['measured_zero']} | "
            f"B1 floor={r['B1_per_session']['floor_bp']:.6f}bp "
            f"floored={r['B1_per_session']['floored']} y={r['B1_per_session']['y']:.6f} | "
            f"B2 floor={r['B2_sqrt_h']['floor_bp']:.6f}bp "
            f"floored={r['B2_sqrt_h']['floored']} y={r['B2_sqrt_h']['y']:.6f}"
        )
        print("    g =", [f"{x:.6e}" for x in r["g"]], "dist", r["distributions"])


if __name__ == "__main__":
    main(sys.argv[1])
