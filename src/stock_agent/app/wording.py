"""
The copy of the Portfolio Lab's destinations other than Explore (whose copy
is `labels`): Overview, Portfolio Builder, Experiments, Model Health and
Evidence. One data module, so tests assert the exact wording and scan it.

Wording rules (as for `labels`): describe the past only; no forecast, no
advice, no trading verbs; nothing is ranked or preferred; every pending
capability is named as pending, with the date it was recorded. The one
exception is the Experiments page's prospective forecasts, shown only as
recorded research records, each labelled unvalidated with no demonstrated
skill, beside the historical average; still no advice and no trading verbs.
"""

from __future__ import annotations

LAB_NAME = "Portfolio Lab"

# Navigation: (title, url path), in display order.
NAV = {
    "overview": ("Overview", "overview"),
    "explore": ("Explore", "explore"),
    "builder": ("Portfolio Builder", "builder"),
    "experiments": ("Experiments", "experiments"),
    "health": ("Model Health", "model-health"),
    "evidence": ("Evidence", "evidence"),
}

CANNOT_TELL_TITLE = "What this lab cannot tell you"
CANNOT_TELL = (
    "The lab describes the past. Its only forecasts are the unvalidated research records on the "
    "Experiments page, which have no demonstrated skill. It does not tell anyone what to hold. "
    "Past returns, real or synthetic, do not predict future returns."
)
DETAILS = "Exact measurements"

# --- Overview ---

OVERVIEW_EYEBROW = LAB_NAME
OVERVIEW_TITLE = "Overview"
OVERVIEW_LEDE = (
    "What the lab can show today, what needs attention, and what is still waiting on a "
    "decision. Everything here describes the past; nothing is a forecast."
)
HAPPENING_TITLE = "What is happening"
ATTENTION_TITLE = "Needs attention"
NOTHING_NEEDS_ATTENTION = "Nothing needs attention: every fund has current, complete data."
PENDING_TITLE = "Waiting on a decision"
NEXT_TITLE = "Where to go next"
NEXT_LINKS = {
    "explore": "Explore fund history",
    "builder": "Build a mix",
    "evidence": "See where the numbers come from",
}
# What is waiting on a decision, recorded from the project's state on
# PENDING_AS_OF: (word, title, sentence, decision). `decision` is never shown;
# it names the pending entry of docs/PROJECT_STATE.md ("Pending:") the item
# waits for, and a test fails once that entry is no longer pending, so this
# list is updated when the decision is taken. None: a fact about the lab itself.
PENDING_AS_OF = "2026-10-15"
PENDING = (
    (
        "Awaiting approval",
        "How a mix would have grown",
        (
            "A fixed mix held without rebalancing or costs needs one more formula approval, "
            "so the Builder shows weights and concentration only."
        ),
        "D9",
    ),
    (
        "Being built",
        "Simulator formulas",
        (
            "Rebalancing, transaction costs and execution timing were approved on 2026-10-15 "
            "and are being built. Cash returns, benchmarks and the reporting statistics still "
            "await approval, so the lab shows no simulated portfolio."
        ),
        "OD8 groups C-D",
    ),
    (
        "Not loaded",
        "Real fund data",
        (
            "Real fund histories need an approved data source and publication first. Until "
            "then every fund is synthetic demo data."
        ),
        "OD9",
    ),
    (
        "None",
        "Forecasts",
        (
            "No forecast exists and no model is validated, so no expected return, probability "
            "or confidence level is shown anywhere."
        ),
        None,
    ),
    (
        "Not yet",
        "Experiments and model health",
        (
            "No machine-learning or reinforcement-learning result is ready to show. Model "
            "Health shows what the local monitor last measured, once it has run."
        ),
        None,
    ),
)

# --- Portfolio Builder ---

BUILDER_EYEBROW = "Build"
BUILDER_TITLE = "Portfolio Builder"
BUILDER_LEDE = (
    "Combine synthetic funds with weights you choose and see how those weights are spread. "
    "The weights are illustrative, from the rule you chose."
)
ILLUSTRATIVE_BADGE = "Illustrative weights from the rule you chose"
NO_HISTORY_BADGE = "No mix history yet"
INPUTS_TITLE = "Your mix"
INPUTS_SCOPE = "Choose funds and a rule for their weights. Nothing is ranked or preferred."
FUNDS_LABEL = "Funds"
FUNDS_HELP = "Listed in a fixed order. No fund is ranked or preferred."
RULE_LABEL = "Weights"
RULES = {"equal": "Equal weights", "entered": "Weights I enter"}
RULE_HELP = "Equal weights split the mix evenly. Entered weights are whole percentage points."
WEIGHT_HELP = "Whole percentage points in steps of 5. A fund at 0% is left out of the mix."
NO_DATA_FUNDS = "Not listed, because they have no available data: {names}."
EMPTY = "No fund chosen. Choose one or more funds above to build a mix."
POINTS_SHORT = "The weights add to {total}%. Add {gap} points so they reach 100%."
POINTS_OVER = "The weights add to {total}%. Remove {gap} points so they add to 100%."
MIX_NAME = "This mix"
CONCENTRATION_TITLE = "Concentration"
CONCENTRATION_SCOPE = (
    "How the weights are spread across funds and their roles. Holdings inside each fund are "
    "not known for synthetic funds, so concentration inside a fund is not shown."
)
WEIGHTS_SUMMARY = "Weights: {weights}."
CONCENTRATED = "More than half of the mix sits in one fund: {fund}, {weight}."
SINGLE_FUND = "The whole mix sits in one fund: {fund}."
FUNDS_HELD_LABEL = "Funds held"
LARGEST_LABEL = "Largest weight"
ROLES_LABEL = "Roles held"
WHAT_IF_TITLE = "What if the weights were different?"
WHAT_IF_SCOPE = (
    "Another rule for the same funds, and how its weights would be spread. It shows weights "
    "only: no past or future result."
)
WHAT_IF_LABEL = "Compare with"
NO_VARIANT = "No alternative applies: choose two or more funds to compare other weights."
HISTORY_PENDING_TITLE = "How this mix would have grown"
HISTORY_PENDING = (
    "How a fixed mix would have grown needs one more formula approval: a mix held from its "
    "start without rebalancing or costs. Until it is approved, the Builder shows weights and "
    "concentration only. Each fund's own history is on the Explore page."
)
NOT_MODELLED_BADGE = "Awaiting approval"
NOT_MODELLED = (
    "Rebalancing, transaction costs and execution timing were approved for the simulator on "
    "2026-10-15 and are being built, not shown here. Taxes are not modelled, and the Builder "
    "makes no forecast."
)
EXPLORE_LINK = "See each fund's history"

# --- Experiments ---

EXPERIMENTS_EYEBROW = "Research"
EXPERIMENTS_TITLE = "Experiments"
EXPERIMENTS_LEDE = (
    "Machine-learning and reinforcement-learning experiments will appear here once they "
    "produce results this lab can show, each with its evidence status."
)
EXPERIMENTS_EMPTY_BADGE = "No experiments yet"
EXPERIMENTS_EMPTY_TITLE = "No experiment to show yet"
EXPERIMENTS_EMPTY = (
    "No run has produced a result for this lab. Nothing is estimated or filled in while that is so."
)
EXPERIMENTS_LATER = (
    "When runs exist, each will show its data window, validation status and limitations, as "
    "research evidence only: never as a portfolio to hold."
)
PROSPECTIVE_TITLE = "Prospective forecasts (unvalidated)"
PROSPECTIVE_INTRO = (
    "Each forecast was recorded once, before its outcome existed, by models frozen on data up to "
    "2024. It is a research record with no demonstrated skill, shown beside the historical "
    "average, the simple forecast it is compared with, and it is scored only when the real "
    "outcome arrives."
)
PROSPECTIVE_EMPTY_BADGE = "No forecast recorded yet"
PROSPECTIVE_EMPTY_TITLE = "No forecast recorded yet"
PROSPECTIVE_EMPTY = (
    "The forecast ledger is empty or not created yet. Nothing is estimated or filled in while "
    "that is so."
)
PROSPECTIVE_LATER = (
    "When a forecast is recorded, it will show its forecast, its 80% band and the historical "
    "average beside it, and its outcome once that is known."
)
PROSPECTIVE_BROKEN = (
    "The forecast ledger failed its integrity check ({reason}), so none of its records is shown."
)
PROSPECTIVE_BADGE = "Unvalidated"
PROSPECTIVE_SYNTHETIC = "Synthetic test data"
PROSPECTIVE_RECORDED = (
    "As of the {as_of} close; recorded {created_at}. Each horizon starts {lag} session after "
    "that close."
)
PROSPECTIVE_UNITS = (
    "Total return is the log return over the horizon in percent, distributions reinvested. "
    "Volatility is the daily realized volatility over the horizon in percent. For volatility, "
    "the historical average is a geometric mean: the exponential of the mean log volatility."
)
PROSPECTIVE_BAND = (
    "The 10% and 90% columns are the ends of an 80% band calibrated on development data to 2024 "
    "only. Development coverage was measured out of sample on bands built fold by fold, each "
    "from earlier folds only, not on this frozen band; how often real outcomes fall inside "
    "this band is unknown until many are scored."
)
PROSPECTIVE_SCORES_TITLE = "Scored so far"
PROSPECTIVE_NO_SCORES = (
    "No forecast has matured yet: an outcome is shown only once the last session of its horizon "
    "has closed."
)
PROSPECTIVE_PENDING = "Not yet known"
PROSPECTIVE_REFUSED = (
    "Not forecast, because a close was missing from its data for the window: {assets}. Missing "
    "closes are never filled in."
)
PROSPECTIVE_NO_CLOSE = "No close on the as-of session"
PROSPECTIVE_MEASURES = {
    "total_log_return": "Total return (log)",
    "log_realized_volatility": "Daily volatility",
}
PROSPECTIVE_STAGES = {
    "engineering_only": "Engineering only",
    "development_evaluated": "Development evaluated",
}
PROSPECTIVE_COLUMNS = {
    "fund": "Fund",
    "horizon": "Horizon",
    "measure": "Measure",
    "forecast": "Forecast",
    "low": "10%",
    "high": "90%",
    "null": "Historical average",
    "development": "Development coverage",
    "stage": "Evidence stage",
    "outcome": "Outcome",
    "inside": "Inside the band",
}
DASHBOARD_TITLE = "Research diagnostics dashboard"
DASHBOARD_TEXT = (
    "Saved model-diagnostics runs are read by a separate local research dashboard. Start it "
    "from the repository root, then open it on this machine:"
)
DASHBOARD_COMMAND = ".venv/bin/python scripts/run_dashboard.py"
DASHBOARD_URL = "http://127.0.0.1:8501"
DASHBOARD_LINK = "Open the research dashboard (once it is running)"

# --- Model Health ---

HEALTH_EYEBROW = "Operations"
HEALTH_TITLE = "Model Health"
HEALTH_LEDE = (
    "Whether the code, the data, the forecast records and the models can be trusted today: "
    "the status and the next step first, exact measurements underneath. Read-only: nothing "
    "here acts on what it finds."
)
HEALTH_EMPTY_BADGE = "Not yet collected"
HEALTH_EMPTY_TITLE = "Not yet collected"
HEALTH_EMPTY = (
    "No local health summary was found. Nothing here is estimated or assumed while that is so, "
    "and a check that was never measured is never shown as healthy."
)
HEALTH_LOOKED = "Looked for: {path}."
HEALTH_NOT_SET = "No summary location was given to the lab."
HEALTH_CREATE = (
    "Create one on this machine. It reads private data, so it runs only here, never on GitHub:"
)
HEALTH_CREATE_COMMAND = ".venv/bin/python scripts/health_check.py local --run-verify --github"
HEALTH_LATER = (
    "Then start the lab with scripts/run_portfolio_lab.py, which finds the newest summary. "
    "The scheduled run is described in docs/operations/model_health.md."
)
HEALTH_REFUSED = "The {label} at {path} was refused ({reason}): {detail}. Nothing from it is shown."
HEALTH_LOCAL_LABEL = "local health summary"
HEALTH_CI_LABEL = "daily workflow summary"
HEALTH_NEXT = "Next step: {action}"
HEALTH_GLANCE_TITLE = "At a glance"
HEALTH_GLANCE_CAPTION = "Each area takes the state of its worst check. Open an area for details."
HEALTH_MEASUREMENTS = "Exact measurements: {title}"
HEALTH_MEASURED_FIELDS = {
    "value": "Value",
    "threshold": "Threshold",
    "measured": "Measured",
    "reason": "Reason code",
    "check_id": "Check id",
    "evidence": "Evidence",
    "runbook": "Runbook",
}
HEALTH_ALERTS_TITLE = "Alerts"
HEALTH_ALERTS_CAPTION = "Informational only: nothing acts on an alert."
HEALTH_NO_ALERTS = "No alerts."
HEALTH_NOT_APPLICABLE_TITLE = "Not applicable"
HEALTH_NOT_APPLICABLE_CAPTION = (
    "Checks that do not apply to what was measured. They never count towards the state."
)
HEALTH_PROVENANCE_TITLE = "Provenance"
HEALTH_CI_TITLE = "Daily GitHub check"
HEALTH_CI_SCOPE = (
    "The daily workflow on GitHub checks repository code and synthetic fixtures only; it cannot "
    "see private data or local models. When the local monitor records GitHub runs, its latest "
    "result also appears under Code and tests."
)
HEALTH_CI_MISSING = "No downloaded summary of the daily workflow was found. Download the latest:"
HEALTH_CI_DOWNLOAD = (
    "RUN=$(gh run list --repo billvo9/stock-allocation-agent --workflow health.yml "
    "--limit 1 --json databaseId --jq '.[0].databaseId')\n"
    'gh run download "$RUN" --repo billvo9/stock-allocation-agent --name model-health '
    '--dir "reports/health/ci/$RUN"'
)
HEALTH_CI_GENERATED = "Generated {generated}. {summary}"

# --- Evidence ---

EVIDENCE_EYEBROW = "Provenance"
EVIDENCE_TITLE = "Evidence"
EVIDENCE_LEDE = (
    "Where every number in the lab comes from, what it assumes, and why none of it is ready "
    "for investment decisions."
)
NOT_READY_TITLE = "Why nothing here is investment-ready"
# (status level, word, title, sentence)
NOT_READY = (
    (
        "NOTICE",
        "Descriptive",
        "Facts about the past only",
        (
            "Every number outside the Experiments page describes the past. The forecasts there are "
            "unvalidated research records with no demonstrated skill."
        ),
    ),
    (
        "FUTURE",
        "Being built",
        "No simulator",
        (
            "Transaction costs, rebalancing and execution timing are approved but not yet built "
            "into the lab, so no result reflects what holding a portfolio would have cost."
        ),
    ),
    (
        "FUTURE",
        "None",
        "No validated model",
        (
            "No model or rule has passed a check on data it has not seen, so nothing here has "
            "evidence that it would hold up later."
        ),
    ),
)
SYNTHETIC_REASON = (
    "WARNING",
    "Synthetic",
    "Synthetic demo data",
    "Every fund is generated by a seeded random generator and is not real fund performance.",
)
DATA_TITLE = "Data versions"
DATA_SCOPE = "Every fund the lab loads, with its source, coverage, vintage and limitations."
CALCULATIONS_TITLE = "Calculations"
CALCULATIONS = (
    "Total returns from raw closes, distributions and share-count splits.",
    "Hypothetical growth of $10,000, distributions reinvested on the ex-date.",
    (
        "Maximum drawdown and recovery, or the largest observed decline when a history is "
        "incomplete."
    ),
    "Calendar-year returns, partial years labelled and never annualized.",
)
CALCULATIONS_SOURCE = (
    "Approved by the project owner on 2026-10-14 for descriptive use only, and implemented in "
    "`stock_agent.portfolio.descriptive`."
)
BUILDER_STEP = (
    "The Builder restates weights only (their spread by fund and by role). How a mix would "
    "have grown awaits approval of one more formula."
)
MODELS_TITLE = "Models"
MODELS = (
    "No model is validated. The only model outputs in the lab are the unvalidated research "
    "forecasts on the Experiments page."
)
PUBLICATION_TITLE = "Publication provenance"
PUBLICATION = (
    "No publication is read yet. The funds are generated in memory by the synthetic demo "
    "generator, version {version}, seed {seed}, on weekday sessions from {start} to {end}."
)
SEALED = (
    "Data from {sealed} on is reserved for one final test and is never shown; the demo data "
    "ends {end}."
)
ASSUMPTIONS_TITLE = "Assumptions"
SOFTWARE_TITLE = "Software"
SOFTWARE_SCOPE = "The versions this page is running with."
