"""
Key tables from sim_results_by_seed.csv (SIMULATION ONLY; see
scripts/research/zero_vol_simulation.py). Writes sim_tables.txt next to it.

    .venv/bin/python scripts/research/zero_vol_simulation_tables.py <output directory>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def mse(x):
    return f"{np.mean(x):.3f}±{np.std(x, ddof=1) / np.sqrt(len(x)):.3f}"


def main(d):
    d = Path(d)
    f = pd.read_csv(d / "sim_results_by_seed.csv")
    lines = []

    def emit(title, table):
        lines.append(f"\n## {title}\n")
        lines.append(table.to_string())
        print(f"\n## {title}\n")
        print(table.to_string())

    one = f[f["rule"] == "B1"]
    # 1. zero-window frequency
    z = one.groupby(["p0", "dynamics", "mu", "sigma_bp", "h"])["zero_rate_all"].mean().unstack("h")
    emit("Zero-window frequency (share of all windows), mean of 10 seeds", z.round(3))
    fl = (
        one.groupby(["p0", "dynamics", "mu", "sigma_bp", "h"])[
            ["floored_B1_not_zero_all", "floored_B2_not_zero_all"]
        ]
        .mean()
        .unstack("h")
    )
    emit("Floored but NOT measured zero (share of windows)", fl.round(3))
    # 2. target bias vs latent
    cols = [
        "bias_raw_nonzero",
        "selection_lat_nonzero_minus_all",
        "labelmean_minus_latmean_C",
        "labelmean_minus_latmean_B1",
        "labelmean_minus_latmean_B2",
        "bias_B1_zero",
        "bias_B2_zero",
        "bias_B1_floored_nonzero",
        "min_raw_minus_lat_nonzero",
    ]
    b = one.groupby(["p0", "dynamics", "mu", "sigma_bp", "h"])[cols].mean()
    emit("Target bias vs latent window log-vol (log units), mean of seeds", b.round(3))
    # 3. coverage
    ok = f[f["status"] == "ok"]
    cov = (
        ok.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])["coverage"]
        .mean()
        .unstack("rule")
    )
    emit("80% conformal band coverage on held-out test segment (rule's own scoring)", cov.round(3))
    cov_se = (
        ok.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])["coverage"]
        .agg(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
        .unstack("rule")
    )
    emit("Coverage standard error across seeds", cov_se.round(3))
    cz = (
        ok.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])["coverage_zero_rows"]
        .mean()
        .unstack("rule")
    )
    emit("Coverage on zero-window test rows only", cz.round(3))
    w = (
        ok.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])["band_width"]
        .mean()
        .unstack("rule")
    )
    emit("Band width (log units)", w.round(3))
    # 4. QLIKE vs latent and forecast bias
    q = (
        ok.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])["qlike_vs_latent"]
        .mean()
        .unstack("rule")
    )
    emit("Mean QLIKE of the ridge forecast vs LATENT log-vol (all test rows)", q.round(4))
    piv = ok.pivot_table(
        index=["p0", "dynamics", "mu", "sigma_bp", "h", "seed"],
        columns="rule",
        values="qlike_vs_latent",
    )
    diff = pd.DataFrame(
        {
            "B1_minus_A": piv["B1"] - piv["A"],
            "B2_minus_A": piv["B2"] - piv["A"],
            "B1F1_minus_A": piv["B1_F1"] - piv["A"],
            "B1asof_minus_B1": piv["B1_asof"] - piv["B1"],
            "B2asof_minus_B2": piv["B2_asof"] - piv["B2"],
            "B2_minus_B1": piv["B2"] - piv["B1"],
        }
    ).groupby(["p0", "dynamics", "mu", "sigma_bp", "h"])
    dd = pd.concat(
        [
            diff.mean().add_suffix("_mean"),
            diff.agg(lambda x: np.std(x, ddof=1) / np.sqrt(len(x))).add_suffix("_se"),
        ],
        axis=1,
    )
    emit("Paired QLIKE difference vs latent (negative = better than A/C)", dd.round(4))
    fb = ok.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])[
        "forecast_bias_vs_latent"
    ].mean()
    emit("Forecast bias vs latent: mean(f - y_lat) on test rows", fb.unstack("rule").round(3))
    qo = ok.assign(gain=ok["qlike_own_null"] - ok["qlike_own"])
    emit(
        "QLIKE improvement vs null on each rule's own labels/population",
        qo.groupby(["p0", "dynamics", "mu", "sigma_bp", "h", "rule"])["gain"]
        .mean()
        .unstack("rule")
        .round(4),
    )
    status = f.groupby(["scenario", "h", "rule"])["status"].agg(lambda s: dict(s.value_counts()))
    bad = status[status.map(lambda s: set(s) != {"ok"})]
    emit("Paths without a fit (too few rows)", bad.to_frame())
    (d / "sim_tables.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
