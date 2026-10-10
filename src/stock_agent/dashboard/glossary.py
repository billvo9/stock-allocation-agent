"""
Short definitions shown next to technical terms in the dashboard.

Plain text only, so the same definitions can appear in tooltips, captions
and the glossary page. Definitions describe what T2 computed; they do not
introduce new statistics.
"""

from __future__ import annotations

TERMS: dict[str, str] = {
    "Rank IC": (
        "Spearman correlation between predictions and realized labels across the names "
        "on ONE date, then averaged over dates. With few names it is coarse (4 names: 11 "
        "possible values in steps of 0.2), so a single date says little."
    ),
    "Hit rate": (
        "Share of rows where the prediction and the realized label have the same sign "
        "(rows with a zero prediction or label are excluded and counted). Compared with "
        "the rate expected if signs were independent, p*q + (1-p)*(1-q), not with 50%."
    ),
    "OOS R²": (
        "Out-of-sample R² = 1 - SSE(model) / SSE(baseline), summed over identical rows. "
        "Positive means smaller squared errors than the named baseline forecast."
    ),
    "Cross-sectional R²": (
        "1 - SSE / SST computed on date-demeaned forecasts and labels: squared-error fit "
        "within each date, with market timing removed. It depends on the forecast's scale "
        "(unlike Rank IC) and can be strongly negative for a badly scaled forecast."
    ),
    "Mincer-Zarnowitz": (
        "Regression of realized labels on predictions, y = a + b*prediction. A calibrated "
        "forecast has a = 0 and b = 1. b < 1 can mean overconfidence or a noisy forecast."
    ),
    "Calibration": (
        "Whether forecast LEVELS match realized outcomes. Ranking (Rank IC) only asks "
        "whether the ORDER is right: a forecast can rank well and be badly calibrated, "
        "or be calibrated on average and rank no better than chance."
    ),
    "HAC / Newey-West": (
        "Standard error for a mean of dependent observations: adds Bartlett-weighted "
        "autocovariances up to a chosen lag (recorded with the run). Overlapping labels "
        "make adjacent dates dependent, so the naive standard error is far too small."
    ),
    "Fold-block t": (
        "t-test on the mean of per-fold averages with t(B-1) critical values (B = number "
        "of folds). Fixed in advance as T2's decision method because it came closest to "
        "its nominal 5% false-positive rate in T2's design simulations."
    ),
    "Block bootstrap": (
        "Resamples contiguous blocks of whole dates (block length recorded with the run, "
        "wrapping at the end), so dependence inside a block and each date's cross-section "
        "are kept."
    ),
    "Effective sample size": (
        "n * gamma0 / long-run variance: the number of independent dates carrying the "
        "same information about a mean (Newey-West only). It can exceed n under "
        "negative dependence."
    ),
    "Minimum detectable effect": (
        "Smallest true mean the decision test would detect with 80% power at its 5% "
        "level. 'No evidence' below this size is not evidence of 'no effect'."
    ),
    "Simulated false-positive rate": (
        "Share of simulated no-skill series that a method declared significant at 5%, "
        "recorded with every run. It carries Monte Carlo error (about 0.5 percentage "
        "points at 2,000 replicates); T2's stored threshold sits about 3 such errors above "
        "5%. Rates clearly above it mean the method over-rejects."
    ),
    "VIF": (
        "Variance inflation factor, 1 / (1 - R²) of a feature regressed on the others, on "
        "one fold's standardized training rows. Large values mean collinearity."
    ),
    "Condition number": (
        "sqrt(largest / smallest eigenvalue) of the training correlation matrix (centered "
        "design, no intercept). Large values mean near-linear dependence among features."
    ),
    "Date-variance share": (
        "Share of a feature's variance explained by date means. 1.0 means the feature is "
        "the same for every name on a date (e.g. macro), so it cannot rank names alone."
    ),
    "PSI": (
        "Population stability index of test-window values against the fold's training "
        "deciles. The 0.10 and 0.25 guide lines are conventions, not calibrated tests."
    ),
    "KS distance": ("Largest gap between the training and test-window empirical distributions."),
    "Standardized mean difference": (
        "Test-window mean minus training mean, in training standard deviations."
    ),
    "Purge": (
        "Training rows whose labels end on or after the test window starts are removed "
        "for every symbol, so no training label overlaps the test period."
    ),
    "Lockbox": (
        "Data from the recorded holdout start onward is a final holdout. Development runs "
        "never use its values: rows are cut before labels are built, although the feature "
        "build reads the whole price file (recorded as holdout_rows_loaded). Holdout runs "
        "come only in ticket T6."
    ),
    "Null model": (
        "A predictor with no ranking or timing information by construction (zero, the "
        "training average, noise). It can still beat another baseline on levels (the "
        "average beats zero when markets drift up) but must show no ranking skill; if it "
        "does, suspect a bug or leakage."
    ),
    "Control": (
        "A reference with a known, non-zero source of score (a benchmark or a selection "
        "effect). Its score is context, not evidence of skill."
    ),
    "Canary": (
        "A memorizer whose only possible skill is leakage. The purged run must show no "
        "skill; the deliberately unpurged reference proves leakage would be visible. The "
        "unpurged reference is never a strategy."
    ),
    "Block-permutation null": (
        "Each date's feature vectors are reassigned among that date's names, with the "
        "mapping held for blocks of sessions (length recorded with the run). Breaks the "
        "link to each name's own future returns while keeping each date's cross-section."
    ),
    "Static ordering": (
        "One fixed ranking of the names used on every date. Enumerating every ordering "
        "(24 for 4 names) shows how much Rank IC a ranking that never changes can produce."
    ),
    "Stale-feature control": (
        "The same model fed each name's own features from many sessions earlier (lags "
        "recorded with the run). IC(model) - IC(stale) asks whether the signal is timely."
    ),
    "Pooled deciles": (
        "Prediction buckets formed over ALL out-of-sample rows at once. Descriptive only: "
        "they mix dates and folds and are not same-date portfolios."
    ),
    "Rank position": (
        "Same-date position 1..N (1 = highest prediction). Tied names share their "
        "group's mean outcome. With a handful of names there are no same-date quintiles "
        "or deciles."
    ),
    "Goyal-Welch curve": (
        "Cumulative sum over dates of SSE(baseline) - SSE(model). Rising means the model "
        "is beating the baseline in that period."
    ),
}


def define(term: str) -> str:
    """Definition of a known term (KeyError for unknown terms, to catch typos)."""

    return TERMS[term]
