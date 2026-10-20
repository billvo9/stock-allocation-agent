"""
The briefing as plain text and as a self-contained HTML page (standard
library only; the palette is passed in, so the Stock Lab never imports the
design system).

Both renderings show only what the briefing holds: a stock without a stored
forecast reads "Unavailable" (or "Stale") with its reason, never a number.
The order serves a reader at 08:30: notices, then every stock at a glance,
then the header facts, then each stock in full, then reliability.

The HTML page loads nothing from anywhere (no scripts, links, images or
fetched fonts), so it can be opened from disk; it follows the Lab's warm
editorial palette in light and dark (prefers-color-scheme), wraps wide
tables so a phone can scroll them, and writes every category in words beside
its marker, never carried by colour alone (bars and markers are hidden from
screen readers, which read the words and numbers). No link to a local
dashboard is included: a recipient may not be able to reach it.

The theme mapping (`Theme`) holds, per scheme ("light", "dark"): the Lab's
colours (canvas, surface, surface_muted, border, border_strong, text,
text_muted, text_subtle, accent, accent_soft), the status treatments (fg,
bg and border for VALID, WARNING, INVALID/BLOCKED, NOTICE, UNAVAILABLE),
the direction colours (increase, stable, decrease), and the font stacks
(serif, sans, mono).
"""

from __future__ import annotations

import html
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from datetime import date

from stock_agent.stock_lab.briefing.report import (
    FRESH,
    STALE,
    Briefing,
    Probabilities,
    StockBriefing,
    Trap,
    generated_local,
    new_york_time,
    subject,
)
from stock_agent.stock_lab.briefing.timing import long_date
from stock_agent.stock_lab.contracts import EVIDENCE_WORDS, EvidenceStatus, StableBand

Theme = Mapping[str, Mapping]
DASH = "—"
MINUS = "−"
WIDTH = 88
CATEGORIES = ("increase", "stable", "decrease")
WORDS = {"increase": "Increase", "stable": "Stable", "decrease": "Decrease"}
MARKERS = {"increase": "▲", "stable": "●", "decrease": "▼"}  # as the Lab's chart markers
STATUS_WORDS = {FRESH: "Fresh", STALE: "Stale", "unavailable": "Unavailable"}
STATUS_TONE = {FRESH: "valid", STALE: "warning", "unavailable": "unavailable"}
REASON_SHORT = {
    "no_recorded_forecast": "no recorded forecast",
    "stale": "stale",
    "refused": "no probability given",
    "ledger_unverified": "withheld: ledger check failed",
    "target_mismatch": "withheld: calendar mismatch",
    "as_of_not_prior_session": "withheld: calendar mismatch",
    "record_check_failed": "withheld: record check failed",
}
TARGET_WORDS = {"1": "Next session", "cum": "Five-session total"}
OUTCOME_VERBS = {"continuation": "continued", "stabilization": "stabilized", "reversal": "reversed"}
REGIME_WORDS = {
    "calm": "calm (lowest third of its own history)",
    "middle": "middle third of its own history",
    "volatile": "volatile (highest third of its own history)",
}
REGIME_WINDOW_WORDS = {
    "calm": "calm (lowest third of the forecast's own inference window)",
    "middle": "middle third of the forecast's own inference window",
    "volatile": "volatile (highest third of the forecast's own inference window)",
}
WINDOW_REGIME_CAVEAT = " Not comparable with the development tables' regimes."
DEVELOPMENT_METHOD = "Jeffreys, with the distinct dates as the count"
PROSPECTIVE_METHOD = "Jeffreys at the date-clustered effective count"


# --- formatting ---


def pct(value: float | None, digits: int = 0, *, signed: bool = False) -> str:
    if value is None:
        return DASH
    text = f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"
    return text.replace("-", MINUS)


def points(value: float | None) -> str:
    """A change in probability, in percentage points: '+10 pts'."""

    if value is None:
        return DASH
    rounded = round(value * 100)
    return (f"{rounded:+d} pts" if rounded else "0 pts").replace("-", MINUS)


def band(tau: float) -> str:
    return StableBand(tau).words()


def evidence_words(status: str) -> str:
    try:
        return EVIDENCE_WORDS[EvidenceStatus(status)]
    except ValueError:
        return status


def when(text: str | None) -> str:
    return DASH if not text else new_york_time(text)


def _target_words(target: str) -> str:
    return TARGET_WORDS.get(target, f"Session {target} ahead")


def _range(low: float, high: float) -> str:
    return f"{low * 100:.0f}–{high * 100:.0f}%"


def _prepared(briefing: Briefing) -> str:
    local = generated_local(briefing)
    utc = briefing.generated_at[11:16]
    return f"{local:%H:%M} New York ({utc} UTC), {long_date(local.date())}"


def _market(briefing: Briefing) -> str:
    return long_date(date.fromisoformat(briefing.market_date))


def _prior(briefing: Briefing) -> str:
    return long_date(date.fromisoformat(briefing.prior_session))


def _data_words(briefing: Briefing) -> str:
    tail = "" if briefing.data_as_of else " (no stock has a forecast as of that close)"
    return f"Close of {_prior(briefing)}{tail}"


def _freshness(briefing: Briefing) -> str:
    counts = briefing.freshness
    total = sum(counts.values())
    return (
        f"{counts['fresh']} fresh · {counts['stale']} stale · {counts['unavailable']} "
        f"unavailable (of {total})"
    )


def _ledger(briefing: Briefing) -> str:
    check = briefing.ledger
    if check.verified:
        return f"Verified: the hash chain holds ({check.entries:,} entries)."
    return f"Check failed: {check.problem}. Stored probabilities are withheld."


def _three(p: Probabilities) -> str:
    return " · ".join(f"{WORDS[c]} {pct(getattr(p, c))}" for c in CATEGORIES)


def _changes(p: Probabilities) -> str:
    return ", ".join(f"{WORDS[c]} {points(getattr(p, c))}" for c in CATEGORIES)


def _unchanged(p: Probabilities | None) -> bool:
    return p is not None and all(round(getattr(p, c) * 100) == 0 for c in CATEGORIES)


def _change_text(stock: StockBriefing) -> str:
    change = stock.change
    if not change.comparable:
        return change.words
    if _unchanged(change.next_session) and (
        change.cumulative is None or _unchanged(change.cumulative)
    ):
        text = (
            f"No change from the forecast as of {change.prior_as_of} (same band, to the "
            "nearest point)."
        )
        return text + (" The model changed between the two." if change.model_changed else "")
    text = f"{change.words} Next session: {_changes(change.next_session)}."
    if change.cumulative is not None:
        text += f" Five-session total: {_changes(change.cumulative)}."
    return text


def _track_text(stock: StockBriefing, category: str) -> str | None:
    found = next((t for t in stock.track if t.category == category), None)
    if found is None:
        return None
    if found.status == "insufficient":
        return (
            f"development tests: too few forecasts of {_range(found.bin_low, found.bin_high)} "
            f"to say ({found.n:,} on {found.dates:,} dates)"
        )
    return (
        f"development tests: forecasts of {_range(found.bin_low, found.bin_high)} came true "
        f"{_range(found.low, found.high)} of the time ({found.n:,} cases on {found.dates:,} dates)"
    )


def _close(stock: StockBriefing) -> str:
    close = stock.close
    if not close.available:
        return f"Unavailable. {close.words}"
    if close.source == "observed":
        return (
            f"{close.value:,.2f} at the close of {close.session} (retrieved "
            f"{when(close.retrieved_at)}). {close.words}"
        )
    return (
        f"{close.session}: {close.value:,.2f} (stored batch retrieved {when(close.retrieved_at)})"
    )


def _momentum(stock: StockBriefing) -> str:
    context = stock.context
    if not context.available:
        return f"Unavailable. {context.words}"
    recorded = " Recorded with the forecast." if context.regime_basis == "window" else ""
    return (
        f"{pct(context.ret_5, 1, signed=True)} over 5 sessions · "
        f"{pct(context.ret_20, 1, signed=True)} over 20 sessions (price returns).{recorded}"
    )


def _volatility(stock: StockBriefing) -> str:
    context = stock.context
    if not context.available:
        return "Unavailable (see momentum)."
    words = REGIME_WINDOW_WORDS if context.regime_basis == "window" else REGIME_WORDS
    regime = words.get(context.regime or "", "not enough history for a regime")
    source = (
        WINDOW_REGIME_CAVEAT + " Recorded with the forecast."
        if context.regime_basis == "window"
        else ""
    )
    return (
        f"{pct(context.vol_20, 1)} a day over 20 sessions (EWMA {pct(context.ewma_vol, 1)}); "
        f"regime: {regime}. Descriptive only.{source}"
    )


def _reversal(stock: StockBriefing) -> str:
    reversal = stock.reversal
    if not reversal.available:
        return reversal.words
    parts = []
    for outcome in reversal.outcomes:
        verb = OUTCOME_VERBS[outcome.outcome]
        if outcome.estimate is None:
            parts.append(f"{verb}: insufficient observations")
        else:
            parts.append(f"{verb} {pct(outcome.estimate)} ({_range(outcome.low, outcome.high)})")
    return f"{reversal.words} Next session: " + "; ".join(parts) + "."


def _trap(trap: Trap) -> str:
    return trap.words if not trap.available else f"{trap.words} Source: {trap.source}."


def _lineage(stock: StockBriefing) -> str:
    lineage = stock.lineage
    if lineage is None:
        return DASH
    return (
        f"{lineage.estimator} · model {lineage.model_version} · features "
        f"{lineage.feature_version} · evaluation {lineage.evaluation_run_id or DASH} · made "
        f"{when(lineage.created_at)} · evidence: {evidence_words(lineage.evidence_status)}"
    )


def _total(stock: StockBriefing) -> str:
    return (
        f"close of {stock.lineage.as_of_session} to {stock.cumulative_end}, band "
        f"{band(stock.lineage.cumulative_band)}: {_three(stock.cumulative)}"
    )


def _detail_rows(stock: StockBriefing) -> list[tuple[str, str]]:
    """The stock's labelled lines after its probabilities (shared by both renderings)."""

    rows = []
    if stock.status != FRESH and stock.described_as_of is not None:
        lead = "Apart from the observed close, the" if stock.close.source == "observed" else "The"
        rows.append(
            (
                "Note",
                (
                    f"{lead} notes below describe the close of {stock.described_as_of}, the "
                    "newest forecast's, not the latest session."
                ),
            )
        )
    rows.append(("Close", _close(stock)))
    rows += [
        ("Streak", stock.streak.words),
        ("Reversal", _reversal(stock)),
        ("Momentum", _momentum(stock)),
        ("Volatility", _volatility(stock)),
        ("Bull trap", _trap(stock.bull_trap)),
        ("Bear trap", _trap(stock.bear_trap)),
        ("Forecast", _lineage(stock)),
    ]
    return rows


def _glance(stock: StockBriefing) -> tuple[str, ...] | str:
    """At a glance: (increase, stable, decrease, change) for a fresh stock, else words."""

    if stock.status != FRESH or stock.next_session is None:
        reason = REASON_SHORT.get(stock.reason or "", stock.reason or "")
        if stock.status == STALE and stock.lineage is not None:
            reason = f"stale: newest forecast as of {stock.lineage.as_of_session}"
        return f"{STATUS_WORDS[stock.status]}: {reason}"
    p = stock.next_session.probabilities
    change = stock.change
    moved = (
        " ".join(f"{MARKERS[c]} {points(getattr(change.next_session, c))}" for c in CATEGORIES)
        if change.comparable
        else DASH
    )
    return (pct(p.increase), pct(p.stable), pct(p.decrease), moved)


def _development_lines(briefing: Briefing) -> list[str]:
    section = briefing.reliability
    found = section.development
    if found is None:
        return [section.development_words]
    dates = "an unknown number of" if found.dates is None else f"{found.dates:,}"
    lines = [
        (
            f"{section.development_words} Evaluation {found.run_id}, outcomes through "
            f"{found.outcome_end or DASH}, {len(found.stocks)} stocks."
        )
    ]
    scores = (
        f"RPS {found.candidate['rps']:.4f} · Brier {found.candidate['brier']:.4f} · log loss "
        f"{found.candidate['log_loss']:.4f}"
    )
    if found.baseline is None:
        lines.append(
            f"Next session, {found.estimator} (the baseline itself): {scores} on {found.n:,} "
            f"forecasts over {dates} dates."
        )
    else:
        base = found.baseline
        lines.append(
            f"Next session, on the same {found.n:,} forecasts over {dates} dates (lower is "
            f"better): {found.estimator} {scores}; B1 RPS {base['rps']:.4f} · Brier "
            f"{base['brier']:.4f} · log loss {base['log_loss']:.4f}."
        )
    if found.reported is not None:
        reported = found.reported
        difference = f"{reported['rps_difference']:+.5f}".replace("-", MINUS)
        lines.append(
            f"Stored comparison with B1, reported only: RPS difference {difference}; one-sided "
            f"p {_p(reported['p_newey_west'])} (Newey-West), {_p(reported['p_fold_block_t'])} "
            "(fold-block t)."
        )
    lines.append(
        f"Selected by the stored evaluation's rule: {found.selected}. The rule: {found.rule}."
    )
    return lines


def _p(value: float | None) -> str:
    return DASH if value is None else f"{value:.3f}"


def _bucket_line(bucket, *, development: bool) -> str:
    span = _range(bucket.low, bucket.high)
    dates = "unknown" if bucket.dates is None else f"{bucket.dates:,}"
    count = f"{bucket.n:,} forecasts on {dates} dates"
    if bucket.status == "insufficient":
        return f"{span}: predicted {pct(bucket.mean_predicted)}; insufficient evidence ({count})"
    note = ", low sample" if bucket.status == "low_sample" else ""
    return (
        f"{span}: predicted {pct(bucket.mean_predicted)}, happened {pct(bucket.observed)} "
        f"(90% interval {_range(bucket.ci_low, bucket.ci_high)}; {count}{note})"
    )


def _group_label(group) -> str:
    return (
        f"{_target_words(group.target)}, {WORDS[group.category]}: {group.estimator} "
        f"(model {group.model_version}), band {band(group.band)}, {group.n:,} forecasts on "
        f"{group.dates:,} dates"
    )


def _rolling_text(entry) -> str:
    rolling = entry.rolling
    if rolling.means is None:
        text = (
            f"Rolling, latest 63 forecast dates ({entry.estimator}, next session): insufficient "
            f"evidence, {rolling.n:,} forecasts on {rolling.dates} dates."
        )
    else:
        text = (
            f"Rolling, latest 63 forecast dates ({entry.estimator}, next session, {rolling.first} "
            f"to {rolling.last}): RPS {rolling.means['rps']:.4f} · Brier "
            f"{rolling.means['brier']:.4f} · log loss {rolling.means['log_loss']:.4f} on "
            f"{rolling.n:,} forecasts over {rolling.dates} dates."
        )
    return text + " Small samples swing widely; read nothing into a short run."


INTERVAL_NOTE = (
    "Intervals reflect date-to-date noise in the period observed, not long calm or volatile "
    "stretches; a short record cannot show those."
)


# --- plain text ---


def _wrap(text: str, indent: str = "", hang: str | None = None) -> list[str]:
    """Wrapped lines; continuation lines start at `hang` (under the text of a "- " bullet)."""

    following = hang if hang is not None else (" " * len(indent) if indent.strip() else indent)
    lines = textwrap.wrap(text, width=WIDTH, initial_indent=indent, subsequent_indent=following)
    return lines or [indent]


def _labelled(label: str, text: str, width: int = 11) -> list[str]:
    head = f"{label:<{width - 1}} "
    return _wrap(head + text, hang=" " * width)


def render_text(briefing: Briefing) -> str:
    """The briefing as plain text (about 88 columns)."""

    out: list[str] = []
    if briefing.data_class == "synthetic":
        out += ["*** SYNTHETIC DEMO: generated data for demo stocks, not real companies ***", ""]
    out += [
        "PREMARKET BRIEFING (experimental, Stock Probability Lab)"
        if briefing.state == "premarket"
        else "BRIEFING PREVIEW (experimental, Stock Probability Lab)",
        f"For {_market(briefing)} · target 08:30 New York time",
        f"Prepared {_prepared(briefing)} · {briefing.state_words}",
        f"Data: {_data_words(briefing)}",
        "",
    ]
    if briefing.warnings:
        out.append("NOTICES")
        for warning in briefing.warnings:
            out += _wrap(warning, "- ")
        out.append("")
    if briefing.stocks:
        out += ["AT A GLANCE (next session; change since the prior forecast, in points)"]
        out.append(f"  {'Stock':<12}{'▲ Incr.':>8}{'● Stab.':>8}{'▼ Decr.':>8}   Change")
        for stock in briefing.stocks:
            cells = _glance(stock)
            if isinstance(cells, str):
                out += _wrap(f"  {stock.symbol:<12}{cells}", hang=" " * 14)
            else:
                out.append(
                    f"  {stock.symbol:<12}{cells[0]:>8}{cells[1]:>8}{cells[2]:>8}   {cells[3]}"
                )
        out.append("")
    out.append("STATUS")
    out += _labelled("Data", _data_words(briefing))
    out += _labelled("Freshness", _freshness(briefing))
    out += _labelled("Model", ", ".join(briefing.model_versions) or DASH)
    out += _labelled("Updates", briefing.latest_update.words)
    out += _labelled("Ledger", _ledger(briefing))
    out += _labelled("Calendar", briefing.calendar_id)
    if briefing.excluded:
        out += _labelled("Excluded", f"By the owner's choice: {', '.join(briefing.excluded)}")
    out += ["", "STOCKS"]
    for stock in briefing.stocks:
        out += ["", f"== {stock.symbol} · {STATUS_WORDS[stock.status]} =="]
        out += _wrap(stock.words)
        if stock.next_session is not None and stock.lineage is not None:
            session = stock.next_session
            out.append(
                f"Next session {session.session} (band {band(stock.lineage.band)}; that day's "
                "own move):"
            )
            for category in CATEGORIES:
                value = pct(getattr(session.probabilities, category))
                line = f"  {MARKERS[category]} {WORDS[category]:<9}{value:>5}"
                track = _track_text(stock, category)
                out += _wrap(line + (f"  [{track}]" if track else ""), hang=" " * 20)
            out += _wrap(f"Change: {_change_text(stock)}", "  ")
            out.append(
                f"  Five sessions (each session's own move, band {band(stock.lineage.band)}):"
            )
            for s in stock.five_sessions:
                cells = "  ".join(
                    f"{MARKERS[c]} {WORDS[c]} {pct(getattr(s.probabilities, c)):>4}"
                    for c in CATEGORIES
                )
                out.append(f"    {s.session}  {cells}")
            if stock.cumulative is not None:
                out += _wrap(f"Five-session total: {_total(stock)}", "  ")
        for label, text in _detail_rows(stock):
            out += _wrap(f"{label}: {text}", "  ", hang="    ")
    out += ["", "RELIABILITY", "Development evidence"]
    for line in _development_lines(briefing):
        out += _wrap(line, "  ")
    found = briefing.reliability.development
    if found is not None and found.buckets:
        out.append(
            f"  When {found.estimator} said Increase in these ranges, Increase happened "
            f"(intervals: {DEVELOPMENT_METHOD}):"
        )
        for bucket in found.buckets:
            out += _wrap(_bucket_line(bucket, development=True), "    ")
    prospective = briefing.reliability.prospective
    out.append("Prospective record (recorded forecasts, scored as sessions closed)")
    out += _wrap(prospective.words, "  ")
    for group in prospective.groups:
        if group.target != "1":
            continue
        out += _wrap(f"{_group_label(group)} (intervals: {PROSPECTIVE_METHOD}):", "  ")
        for bucket in group.buckets:
            out += _wrap(_bucket_line(bucket, development=False), "    ")
    for entry in prospective.rolling:
        out += _wrap(_rolling_text(entry), "  ")
    out += _wrap(f"Compared with B1: {prospective.comparison_words}", "  ")
    if prospective.comparison_caveat:
        out += _wrap(prospective.comparison_caveat, "  ")
    for note in (*briefing.reliability.notes, INTERVAL_NOTE):
        out += _wrap(note, "  ")
    out += ["", "LIMITATIONS"]
    for line in briefing.limitations:
        out += _wrap(line, "- ")
    out += ["", f"Briefing schema {briefing.schema} {briefing.schema_version}."]
    return "\n".join(out) + "\n"


# --- HTML ---


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _slug(name: str) -> str:
    return name.lower().replace("/", "-")


def _variables(scheme: Mapping) -> str:
    colours = scheme["colours"]
    lines = [f"--{key.replace('_', '-')}: {value};" for key, value in colours.items()]
    for name, tone in scheme["status"].items():
        lines += [f"--status-{_slug(name)}-{part}: {value};" for part, value in tone.items()]
    lines += [f"--dir-{key}: {value};" for key, value in scheme["direction"].items()]
    return "\n    ".join(lines)


def _css(theme: Theme) -> str:
    fonts = theme["fonts"]
    tones = "".join(
        f".tone-{_slug(name)} {{ color: var(--status-{_slug(name)}-fg); background: "
        f"var(--status-{_slug(name)}-bg); border-color: var(--status-{_slug(name)}-border); }}\n"
        f".say-{_slug(name)} {{ color: var(--status-{_slug(name)}-fg); }}\n"
        for name in theme["light"]["status"]
    )
    return f"""
:root {{
    color-scheme: light dark;
    {_variables(theme["light"])}
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    {_variables(theme["dark"])}
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--canvas); color: var(--text);
  font-family: {fonts["sans"]}; font-size: 16px; line-height: 1.55; }}
main {{ max-width: 860px; margin: 0 auto; padding: 32px 20px 48px; }}
h1, h2, h3 {{ font-family: {fonts["serif"]}; font-weight: 600; color: var(--text); }}
h1 {{ font-size: 2.3rem; line-height: 1.15; margin: 4px 0 8px; }}
h2 {{ font-size: 1.45rem; margin: 36px 0 12px; padding-top: 12px;
  border-top: 1px solid var(--border); }}
h3 {{ font-size: 1.2rem; margin: 0; }}
p {{ margin: 6px 0; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem;
  color: var(--accent); font-weight: 600; margin: 0; }}
.dateline, .muted {{ color: var(--text-muted); }}
.subtle {{ color: var(--text-subtle); font-size: 0.9rem; }}
.banner {{ border: 2px solid var(--status-warning-border); background: var(--status-warning-bg);
  color: var(--status-warning-fg); border-radius: 14px; padding: 12px 16px; font-weight: 600;
  margin: 16px 0; }}
.notices {{ list-style: none; padding: 0; margin: 16px 0; }}
.notices li {{ border-left: 4px solid; padding: 8px 12px; margin: 6px 0;
  border-radius: 0 10px 10px 0; }}
.facts {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 12px 0;
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 18px; }}
.facts dt, dl.rows dt {{ color: var(--text-muted); font-weight: 600; }}
.facts dd, dl.rows dd {{ margin: 0; }}
.stock {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 18px; margin: 14px 0; }}
.stock header {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  margin-bottom: 6px; }}
.pill {{ display: inline-block; border: 1px solid; border-radius: 999px; padding: 1px 10px;
  font-size: 0.82rem; font-weight: 600; }}
{tones}.probs {{ margin: 10px 0 6px; }}
.prob {{ display: grid; grid-template-columns: 8.5em 3.5em 1fr; align-items: center; gap: 10px;
  margin: 4px 0; }}
.prob .label {{ font-weight: 600; }}
.prob .value {{ font-variant-numeric: tabular-nums; text-align: right; }}
.bar {{ height: 10px; background: var(--surface-muted); border-radius: 999px; overflow: hidden;
  border: 1px solid var(--border); }}
.fill {{ display: block; height: 100%; }}
.fill.increase, .fill.stable, .fill.decrease {{ background: var(--dir-stable); }}
.fill.increase {{ background: var(--dir-increase); }}
.fill.decrease {{ background: var(--dir-decrease); }}
.marker.increase {{ color: var(--dir-increase); }}
.marker.stable {{ color: var(--dir-stable); }}
.marker.decrease {{ color: var(--dir-decrease); }}
.track {{ grid-column: 1 / -1; color: var(--text-subtle); font-size: 0.86rem; margin: -2px 0 4px; }}
.change {{ margin: 8px 0 4px; }}
.scroll {{ overflow-x: auto; margin: 8px 0 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
caption {{ caption-side: top; text-align: left; color: var(--text-subtle); font-size: 0.88rem;
  padding: 0 0 6px; position: sticky; left: 0; max-width: 85vw; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border);
  vertical-align: top; }}
thead th {{ background: var(--surface-muted); color: var(--text-muted); font-weight: 600; }}
th[scope="row"] {{ font-weight: 600; white-space: nowrap; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.words {{ color: var(--text-muted); }}
dl.rows {{ display: grid; grid-template-columns: 9.5em 1fr; gap: 4px 14px; margin: 10px 0 0; }}
footer {{ margin-top: 36px; color: var(--text-muted); font-size: 0.9rem; }}
@media (max-width: 600px) {{
  main {{ padding: 20px 14px 36px; }}
  h1 {{ font-size: 1.9rem; }}
  .facts, dl.rows {{ grid-template-columns: 1fr; }}
  .prob {{ grid-template-columns: 7em 3em 1fr; }}
  th, td {{ padding: 4px; }}
}}
""".strip()


def _table(caption: str, head: Sequence[tuple[str, bool]], rows: Iterable[str]) -> str:
    """A captioned, scrollable table; `head` is (label HTML, numeric) per column."""

    numeric_class = ' class="num"'
    cells = "".join(
        f'<th scope="col"{numeric_class if numeric else ""}>{label}</th>' for label, numeric in head
    )
    return (
        f'<div class="scroll"><table><caption>{_e(caption)}</caption><thead><tr>{cells}</tr>'
        f"</thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _marked(category: str) -> str:
    return f'<span class="marker {category}" aria-hidden="true">{MARKERS[category]}</span> {WORDS[category]}'


def _glance_html(briefing: Briefing) -> str:
    rows = []
    for stock in briefing.stocks:
        cells = _glance(stock)
        head = f'<th scope="row">{_e(stock.symbol)}</th>'
        if isinstance(cells, str):
            tone = STATUS_TONE[stock.status]
            rows.append(f'<tr>{head}<td colspan="4" class="words say-{tone}">{_e(cells)}</td></tr>')
        else:
            body = "".join(f'<td class="num">{_e(cell)}</td>' for cell in cells[:3])
            rows.append(f'<tr>{head}{body}<td class="num">{_e(cells[3])}</td></tr>')
    head = [("Stock", False), *((_marked(c), True) for c in CATEGORIES), ("Change (pts)", True)]
    return _table(
        "Next session's probabilities, and the change since the prior forecast at the same band",
        head,
        rows,
    )


def _probability_rows(stock: StockBriefing) -> str:
    session = stock.next_session
    rows = []
    for category in CATEGORIES:
        value = getattr(session.probabilities, category)
        track = _track_text(stock, category)
        rows.append(
            f'<div class="prob"><span class="label">{_marked(category)}</span>'
            f'<span class="value">{_e(pct(value))}</span>'
            f'<span class="bar" aria-hidden="true">'
            f'<span class="fill {category}" style="width: {value * 100:.1f}%"></span></span>'
            + (f'<span class="track">{_e(track)}</span>' if track else "")
            + "</div>"
        )
    return "".join(rows)


def _five_table(stock: StockBriefing) -> str:
    rows = [
        f'<tr><th scope="row">{_e(s.session)}</th><td class="num">{_e(s.horizon)}</td>'
        + "".join(
            f'<td class="num">{_e(pct(getattr(s.probabilities, c)))}</td>' for c in CATEGORIES
        )
        + "</tr>"
        for s in stock.five_sessions
    ]
    head = [("Session", False), ("Ahead", True), *((_marked(c), True) for c in CATEGORIES)]
    return _table(f"Each session's own move, band {band(stock.lineage.band)}", head, rows)


def _stock_html(stock: StockBriefing) -> str:
    tone = STATUS_TONE[stock.status]
    words_class = "muted" if stock.status == FRESH else f"say-{tone}"
    parts = [
        (
            f'<article class="stock" id="stock-{_e(stock.symbol)}"><header>'
            f"<h3>{_e(stock.symbol)}</h3>"
            f'<span class="pill tone-{tone}">{STATUS_WORDS[stock.status]}</span></header>'
        ),
        f'<p class="{words_class}">{_e(stock.words)}</p>',
    ]
    rows = _detail_rows(stock)
    if stock.next_session is not None and stock.lineage is not None:
        parts.append(
            f'<p class="subtle">Next session {_e(stock.next_session.session)}, band '
            f"{_e(band(stock.lineage.band))}; that day's own move</p>"
            f'<div class="probs">{_probability_rows(stock)}</div>'
            f'<p class="change"><strong>Change.</strong> {_e(_change_text(stock))}</p>'
            f"{_five_table(stock)}"
        )
        if stock.cumulative is not None:
            rows.insert(0, ("Five-session total", _total(stock)))
    parts.append(
        '<dl class="rows">'
        + "".join(f"<dt>{_e(label)}</dt><dd>{_e(text)}</dd>" for label, text in rows)
        + "</dl></article>"
    )
    return "".join(parts)


def _bucket_table(caption: str, buckets: Iterable, method: str) -> str:
    rows = []
    for bucket in buckets:
        if bucket.status == "insufficient":
            happened, interval = "Insufficient evidence", DASH
        else:
            happened, interval = pct(bucket.observed), _range(bucket.ci_low, bucket.ci_high)
        dates = DASH if bucket.dates is None else f"{bucket.dates:,}"
        rows.append(
            f'<tr><th scope="row">{_range(bucket.low, bucket.high)}</th>'
            f'<td class="num">{_e(pct(bucket.mean_predicted))}</td>'
            f'<td class="num">{_e(happened)}</td><td class="num">{_e(interval)}</td>'
            f'<td class="num">{bucket.n:,}</td><td class="num">{dates}</td></tr>'
        )
    head = [
        ("Predicted range", False),
        ("Mean predicted", True),
        ("Happened", True),
        ("90% interval", True),
        ("Forecasts", True),
        ("Dates", True),
    ]
    return _table(f"{caption} Interval: {method}.", head, rows)


def _reliability_html(briefing: Briefing) -> str:
    section = briefing.reliability
    parts = ["<h2>Reliability</h2>", "<h3>Development evidence</h3>"]
    parts += [f"<p>{_e(line)}</p>" for line in _development_lines(briefing)]
    found = section.development
    if found is not None and found.buckets:
        parts.append(
            _bucket_table(
                f"When {found.estimator} said Increase in these ranges, how often Increase "
                "happened (development test).",
                found.buckets,
                DEVELOPMENT_METHOD,
            )
        )
    prospective = section.prospective
    parts += ["<h3>Prospective record</h3>", f"<p>{_e(prospective.words)}</p>"]
    for group in prospective.groups:
        if group.target == "1":
            parts.append(
                _bucket_table(f"{_group_label(group)}.", group.buckets, PROSPECTIVE_METHOD)
            )
    for entry in prospective.rolling:
        parts.append(f"<p>{_e(_rolling_text(entry))}</p>")
    parts.append(f"<p><strong>Compared with B1.</strong> {_e(prospective.comparison_words)}</p>")
    if prospective.comparison_caveat:
        parts.append(f"<p>{_e(prospective.comparison_caveat)}</p>")
    parts += [f'<p class="subtle">{_e(note)}</p>' for note in (*section.notes, INTERVAL_NOTE)]
    return "".join(parts)


def _facts(briefing: Briefing) -> str:
    rows: list[tuple[str, str]] = [
        ("Data", _data_words(briefing)),
        ("Freshness", _freshness(briefing)),
        ("Model", ", ".join(briefing.model_versions) or DASH),
        ("Updates", briefing.latest_update.words),
        ("Ledger", _ledger(briefing)),
        ("Prepared", f"{_prepared(briefing)}. {briefing.state_words}."),
        ("Calendar", briefing.calendar_id),
    ]
    if briefing.excluded:
        rows.append(("Excluded", f"By the owner's choice: {', '.join(briefing.excluded)}"))
    return (
        '<dl class="facts">'
        + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rows)
        + "</dl>"
    )


def _notices(briefing: Briefing) -> str:
    items = []
    for warning in briefing.warnings:
        severe = not briefing.ledger.verified and warning.startswith("The ledger check failed")
        tone = "invalid-blocked" if severe else "notice"
        items.append(f'<li class="tone-{tone}">{_e(warning)}</li>')
    return f'<ul class="notices">{"".join(items)}</ul>' if items else ""


def render_html(briefing: Briefing, theme: Theme) -> str:
    """The briefing as one self-contained HTML document."""

    banner = (
        '<p class="banner">Synthetic demo: generated data for demo stocks, not real companies. '
        "Nothing here is evidence about any real stock.</p>"
        if briefing.data_class == "synthetic"
        else ""
    )
    heading = "Premarket briefing" if briefing.state == "premarket" else "Briefing preview"
    glance = (
        f"<h2>At a glance</h2>{_glance_html(briefing)}"
        if briefing.stocks
        else '<p class="muted">The watchlist is empty: there is nothing to brief.</p>'
    )
    stocks = "".join(_stock_html(stock) for stock in briefing.stocks)
    limitations = "".join(f"<li>{_e(line)}</li>" for line in briefing.limitations)
    local = generated_local(briefing)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_e(subject(briefing))}</title>
<style>
{_css(theme)}
</style>
</head>
<body>
<main>
<header>
<p class="eyebrow">Stock Probability Lab · Experimental</p>
<h1>{heading}</h1>
<p class="dateline">For <strong>{_e(_market(briefing))}</strong> · target 08:30 New York time ·
prepared {local:%H:%M}</p>
{banner}
</header>
{_notices(briefing)}
{glance}
{_facts(briefing)}
{"<h2>Stocks</h2>" + stocks if stocks else ""}
{_reliability_html(briefing)}
<footer>
<h2>Limitations</h2>
<ul>{limitations}</ul>
<p class="subtle">Briefing schema {_e(briefing.schema)} {_e(briefing.schema_version)}.</p>
</footer>
</main>
</body>
</html>
"""
