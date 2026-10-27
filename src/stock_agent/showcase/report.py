"""
The private showcase report: one HTML page and a Markdown twin, rendered from
a results record (`showcase.run.run_showcase`). Pure functions: no I/O, no
clock.

Every page carries the showcase's labels (`run.LABELS`): exploratory,
hindsight-selected, survivorship-biased, 2003-2024 only, not evidence of
skill and not advice, vendor-reported dividends, before taxes, fully
invested. Variants appear in registered order and are never ranked; the
wording states what happened and never which rule "won".

Charts are inline SVG on a log scale (equal distances are equal percentage
changes), one per universe: the rule is the line's colour, the rebalancing
cadence its dash, a legend names every line, and a hover readout gives each
line's value on a weekly grid. A line breaks where wealth is unavailable,
never bridging a gap. The page uses the Lab's editorial tokens
(`ui.theme`), light and dark, and its bundled fonts when they sit beside the
page; nothing is fetched from another host.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence

from stock_agent.ui.theme import LAB_COLOURS, LAB_DARK_COLOURS, LAB_TYPE

TITLE = "Rebalancing on real prices, 2003-2024"
SUBTITLE = "An exploratory, private look at registered rules on three lists of holdings"
FONT_FILES = (
    ("SourceSerif4", "SourceSerif4Variable-Roman.woff2", "normal"),
    ("SourceSans3", "SourceSans3VF-Upright.woff2", "normal"),
    ("SourceSans3", "SourceSans3VF-Italic.woff2", "italic"),
)
# Categorical slots 1-3 of the data-viz reference palette (validated on all
# pairs in both schemes): the rule is the colour.
RULE_COLOURS = {
    "light": {
        "equal_weight": "#2a78d6",
        "inverse_volatility": "#eb6834",
        "buy_and_hold": "#1baf7a",
    },
    "dark": {"equal_weight": "#3987e5", "inverse_volatility": "#d95926", "buy_and_hold": "#199e70"},
}
# The cadence is the dash (sessions between rebalances).
DASHES = {5: "", 21: "7 4", 63: "2 3"}
CHART_WIDTH, CHART_HEIGHT = 960, 380
MARGIN = {"left": 70, "right": 18, "top": 16, "bottom": 34}

WHAT_IT_SHOWS = (
    (
        "How $10,000 would have moved under each registered rule on real 2003-2024 prices, for "
        "three lists of holdings, after the provisional trading costs. Every variant run is shown."
    ),
    (
        "How much trading each rule needed (turnover) and what it cost, with the same decisions "
        "replayed at 0x, 1x and 2x the registered cost rates."
    ),
    "The size and dates of each variant's worst decline, and its calendar-year swings.",
    (
        "What each holding did on its own over the same span, which drives most of what the rules "
        "show."
    ),
)
WHAT_IT_DOES_NOT = (
    (
        "Skill. This is one 22-year path. No interval or significance test is approved, so a "
        "difference between two rules here may be noise."
    ),
    "Anything about the future, or about 2025 on: that period is sealed and never used.",
    (
        "A fair selection test. Every list was chosen in 2026, knowing how 2003-2024 turned out. "
        "The stock list keeps only companies that survived under one ticker, and the four "
        "semiconductors were picked after they rose enormously, so any rule holding them looks good."
    ),
    (
        "Sharpe ratios, a cash return, benchmarks or intervals: none is computed here yet. The "
        "owner approved them on 2026-10-26, and each needs something first: the T-bill rate "
        "series, benchmark prices, or a coverage check of the interval method."
    ),
    (
        "Taxes, or income exactly as the issuers paid it: dividends are as the vendor reported them, "
        "and spin-offs and other non-cash distributions are not in the data. The stock list's "
        "exchange-offer separations (which do not lower the price of shares kept) are assumed to "
        "have left no clean-up spin-off."
    ),
    (
        "Today's sectors: each stock carries the sector its fund gave it in 2002 (Walmart was then "
        "a consumer discretionary holding)."
    ),
    "Real trading: trades happen at a close at a flat provisional cost, in fractional shares.",
    "Advice of any kind.",
)
FORBIDDEN_WORDS = re.compile(
    r"\b(beat|beats|outperform\w*|underperform\w*|best|winners?|winning|wins|won|alpha|"
    r"better|worse|superior|inferior)\b",
    re.IGNORECASE,
)


# --- formatting ---


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    text = f"{abs(value) * 100:,.{digits}f}%"
    return f"−{text}" if value < 0 else f"+{text}"


def money(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    return f"${value:,.0f}"


def _drawdown_text(drawdown: Mapping) -> str:
    if drawdown.get("value") is None:
        return "unavailable"
    text = f"−{drawdown['value'] * 100:.1f}%"
    if drawdown["measure"] != "maximum_drawdown":
        text += " (largest observed; history incomplete)"
    return text


def _drawdown_dates(drawdown: Mapping) -> str:
    if drawdown.get("value") is None:
        return ""
    end = drawdown["recovery"] or "not recovered by the end"
    return f"peak {drawdown['peak']}, low {drawdown['trough']}, back {end}"


def _stop_text(strategy: Mapping) -> str:
    """Why a run stopped: each holding's unavailable return at the stop, else the engine's reason."""

    detail = strategy.get("stop_detail") or {}
    reasons = "; ".join(f"{t} {r}" for t, r in (detail.get("reasons") or {}).items())
    if reasons:
        return f"stopped on {detail['session']}: {reasons}"
    return f"stopped: {strategy['status_reason']}"


def _turnover_text(turnover: Mapping) -> str:
    return f"{turnover['total_one_way'] * 100:,.0f}%"


# --- the chart ---


def _log_ticks(low: float, high: float) -> list[float]:
    """
    1-2-5 ticks inside [low, high]; on a range too narrow for four of them,
    evenly spaced round ticks (steps of 1, 2, 2.5 or 5 times a power of ten).
    """

    ticks = [
        mantissa * 10.0**exponent
        for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
        for mantissa in (1, 2, 5)
        if low <= mantissa * 10.0**exponent <= high
    ]
    if len(ticks) >= 4:
        return ticks
    raw = (high - low) / 4
    power = 10.0 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    first = math.ceil(low / step) * step
    return [first + i * step for i in range(int((high - first) // step) + 1)]


def _runs(values: Sequence[float | None]) -> list[list[int]]:
    """Index runs of consecutive valued points (a gap breaks a run)."""

    runs, current = [], []
    for index, value in enumerate(values):
        if value is None:
            if current:
                runs.append(current)
            current = []
        else:
            current.append(index)
    if current:
        runs.append(current)
    return runs


def _decimate(run: list[int], values: Sequence[float], slots: int) -> list[int]:
    """Per slice: its first, lowest, highest and last points, so extremes stay drawn."""

    if len(run) <= slots * 4:
        return run
    keep: set[int] = set()
    size = len(run) / slots
    for slot in range(slots):
        part = run[int(slot * size) : int((slot + 1) * size)]
        if part:
            low = min(part, key=lambda i: values[i])
            high = max(part, key=lambda i: values[i])
            keep.update({part[0], part[-1], low, high})
    return sorted(keep)


def growth_chart(universe: Mapping) -> str:
    """The universe's growth-of-$10,000 chart at 1x costs (log scale), as inline SVG."""

    dates = universe["dates"]
    strategies = universe["strategies"]
    valued = [v for s in strategies for v in s["wealth"] if v is not None]
    if not dates or not valued:
        return '<p class="note">No valued session to draw.</p>'
    low, high = min(valued) * 0.95, max(valued) * 1.05
    left, top = MARGIN["left"], MARGIN["top"]
    width = CHART_WIDTH - left - MARGIN["right"]
    height = CHART_HEIGHT - top - MARGIN["bottom"]
    span = max(len(dates) - 1, 1)

    def x(index: int) -> float:
        return left + width * index / span

    def y(value: float) -> float:
        share = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
        return top + height * (1 - share)

    parts = [
        (
            f'<svg class="chart" viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img" '
            f'aria-label="Growth of $10,000 for each rule, {_e(universe["label"])}, '
            f'{_e(dates[0])} to {_e(dates[-1])}, log scale" data-chart="{_e(universe["key"])}">'
        )
    ]
    for tick in _log_ticks(low, high):
        ty = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" x2="{left + width}" y1="{ty:.1f}" y2="{ty:.1f}"/>'
            f'<text class="tick" x="{left - 8}" y="{ty + 4:.1f}" text-anchor="end">'
            f"{_e(money(tick))}</text>"
        )
    years: dict[str, int] = {}
    for index, day in enumerate(dates):
        years.setdefault(day[:4], index)
    step = 2 if len(years) > 12 else 1
    for number, (year, index) in enumerate(years.items()):
        if number % step:
            continue
        tx = x(index)
        parts.append(
            f'<line class="axis" x1="{tx:.1f}" x2="{tx:.1f}" y1="{top + height}" '
            f'y2="{top + height + 5}"/>'
            f'<text class="tick" x="{tx:.1f}" y="{top + height + 20}" text-anchor="middle">'
            f"{_e(year)}</text>"
        )
    parts.append(
        f'<line class="axis" x1="{left}" x2="{left + width}" y1="{top + height}" '
        f'y2="{top + height}"/>'
    )
    for strategy in strategies:
        values = strategy["wealth"]
        path = []
        for run in _runs(values):
            points = _decimate(run, values, int(width))
            path.append("M" + " L".join(f"{x(i):.1f},{y(values[i]):.1f}" for i in points))
        dash = DASHES.get(strategy["every"], "") if strategy["rule"] != "buy_and_hold" else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<path class="line rule-{_e(strategy["rule"])}" d="{" ".join(path)}"{dash_attr}>'
            f"<title>{_e(strategy['label'])}</title></path>"
        )
    parts.append(
        f'<line class="cursor" x1="0" x2="0" y1="{top}" y2="{top + height}" visibility="hidden"/>'
        f'<rect class="hit" x="{left}" y="{top}" width="{width}" height="{height}"/>'
    )
    parts.append("</svg>")
    grid = list(range(0, len(dates), 5))
    if grid[-1] != len(dates) - 1:
        grid.append(len(dates) - 1)
    data = {
        "left": left,
        "width": width,
        "span": span,
        "dates": [dates[i] for i in grid],
        "positions": grid,
        "series": [
            {
                "label": s["label"],
                "rule": s["rule"],
                "dash": "" if s["rule"] == "buy_and_hold" else DASHES.get(s["every"], ""),
                "values": [s["wealth"][i] for i in grid],
            }
            for s in strategies
        ],
    }
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    legend = "".join(
        f'<li><svg width="34" height="10" aria-hidden="true"><line class="line '
        f'rule-{_e(s["rule"])}" x1="1" x2="33" y1="5" y2="5"'
        + (
            f' stroke-dasharray="{DASHES.get(s["every"], "")}"'
            if s["rule"] != "buy_and_hold" and DASHES.get(s["every"])
            else ""
        )
        + f"/></svg>{_e(s['label'])}</li>"
        for s in strategies
    )
    return (
        f'<figure class="figure"><figcaption>Growth of $10,000 at the registered costs (1x), '
        f"from the common build close {_e(dates[0])}. Log scale: equal distances are equal "
        f'percentage changes. Hover for each line\'s value.</figcaption><ul class="legend">'
        f'{legend}</ul><div class="plot">{"".join(parts)}<div class="readout" '
        f'hidden></div></div><script type="application/json" class="chart-data">{payload}'
        f"</script></figure>"
    )


# --- tables ---


def _results_table(universe: Mapping) -> str:
    rows = []
    for s in universe["strategies"]:
        sens = s["cost_sensitivity"]
        turnover, costs = s["turnover"], s["costs"]
        status = "" if s["complete"] else f' <span class="flag">{_e(_stop_text(s))}</span>'
        rows.append(
            "<tr>"
            f'<th scope="row">{_e(s["label"])}{status}<div class="sub">{_e(s["description"])}'
            "</div></th>"
            f'<td class="num">{_e(money(s["final_wealth"]))}</td>'
            f'<td class="num">{_e(pct(s["total_return"]))}</td>'
            f'<td class="num">{_e(_drawdown_text(s["drawdown"]))}'
            f'<div class="sub">{_e(_drawdown_dates(s["drawdown"]))}</div></td>'
            f'<td class="num">{turnover["rebalances"]:,}</td>'
            f'<td class="num">{_e(_turnover_text(turnover))}</td>'
            f'<td class="num">{_e(money(costs["paid_dollars"]))}</td>'
            f'<td class="num">{_e(money(sens["0x"]["final_wealth"]))}</td>'
            f'<td class="num">{_e(money(sens["2x"]["final_wealth"]))}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><caption>Every registered variant, in registered '
        "order (not ranked), at 1x costs unless a column says otherwise.</caption><thead><tr>"
        '<th scope="col">Rule</th><th scope="col" class="num">$10,000 became</th>'
        '<th scope="col" class="num">Total return</th>'
        '<th scope="col" class="num">Maximum drawdown</th>'
        '<th scope="col" class="num">Rebalances</th>'
        '<th scope="col" class="num">Turnover, one way, summed (build included)</th>'
        '<th scope="col" class="num">Costs paid</th>'
        '<th scope="col" class="num">At 0x costs</th>'
        '<th scope="col" class="num">At 2x costs</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _calendar_table(universe: Mapping) -> str:
    strategies = universe["strategies"]
    years = sorted({row["year"] for s in strategies for row in s["calendar_years"]})
    by = {s["key"]: {row["year"]: row for row in s["calendar_years"]} for s in strategies}
    head = "".join(f'<th scope="col" class="num">{_e(s["label"])}</th>' for s in strategies)
    rows = []
    for year in years:
        cells = []
        partial = False
        for s in strategies:
            row = by[s["key"]].get(year)
            if row is None:
                cells.append('<td class="num">-</td>')
                continue
            partial |= row["partial"]
            value = row["total_return"]
            cls = "num neg" if value is not None and value < 0 else "num"
            cells.append(f'<td class="{cls}">{_e(pct(value))}</td>')
        label = f"{year} (part)" if partial else str(year)
        rows.append(f'<tr><th scope="row">{_e(label)}</th>{"".join(cells)}</tr>')
    return (
        '<details><summary>Calendar-year returns</summary><div class="table-wrap"><table>'
        "<caption>Year-end close to year-end close at 1x costs; a part year is never "
        f'annualized.</caption><thead><tr><th scope="col">Year</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div></details>"
    )


def _components_table(universe: Mapping) -> str:
    rows = []
    for c in universe["components"]:
        notes = []
        if c["missing_sessions"]:
            notes.append(f"{c['missing_sessions']} sessions without a close")
        for reason, count in c["unavailable_returns"].items():
            notes.append(f"{count} unavailable ({reason})")
        if c["revised_sessions"]:
            notes.append(f"{c['revised_sessions']} revised sessions")
        rows.append(
            f'<tr><th scope="row">{_e(c["ticker"])}</th><td>{_e(c["sector"])}</td>'
            f'<td class="num">{_e(pct(c["total_return"]))}</td>'
            f'<td class="num">{_e(_drawdown_text(c["drawdown"]))}</td>'
            f"<td>{_e('; '.join(notes) or 'none')}</td></tr>"
        )
    return (
        '<details><summary>Each holding on its own</summary><div class="table-wrap"><table>'
        "<caption>Each holding's growth of $10,000 held alone over the same span, before costs "
        "(context for the rules above, not a rule).</caption><thead><tr>"
        '<th scope="col">Holding</th><th scope="col">Sector</th>'
        '<th scope="col" class="num">Total return</th>'
        '<th scope="col" class="num">Maximum drawdown</th><th scope="col">Data notes</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>"
    )


# --- the page ---


def _css(font_dir: str | None) -> str:
    light, dark = LAB_COLOURS, LAB_DARK_COLOURS
    faces = ""
    if font_dir is not None:
        faces = "".join(
            f'@font-face{{font-family:"{family}";src:url("{font_dir}/{file}") format("woff2");'
            f"font-weight:200 900;font-style:{style};font-display:swap}}"
            for family, file, style in FONT_FILES
        )

    def tokens(colours: Mapping[str, str], scheme: str) -> str:
        rules = RULE_COLOURS[scheme]
        return (
            f"color-scheme:{scheme};--canvas:{colours['canvas']};--surface:{colours['surface']};"
            f"--muted:{colours['surface_muted']};--border:{colours['border']};"
            f"--border-strong:{colours['border_strong']};--text:{colours['text']};"
            f"--text-muted:{colours['text_muted']};--text-subtle:{colours['text_subtle']};"
            f"--accent:{colours['accent']};--accent-soft:{colours['accent_soft']};"
            f"--grid:{colours['grid']};--ew:{rules['equal_weight']};"
            f"--iv:{rules['inverse_volatility']};--bh:{rules['buy_and_hold']};"
        )

    return (
        faces
        + f":root{{{tokens(light, 'light')}}}"
        + '@media (prefers-color-scheme: dark){:root:not([data-theme="light"])'
        + f"{{{tokens(dark, 'dark')}}}}}"
        + f':root[data-theme="dark"]{{{tokens(dark, "dark")}}}'
        + f"""
*{{box-sizing:border-box}}
body{{margin:0;background:var(--canvas);color:var(--text);font-family:{LAB_TYPE["sans"]};
font-size:16px;line-height:1.55;font-variant-numeric:tabular-nums}}
main{{max-width:1120px;margin:0 auto;padding:32px 16px 64px}}
h1,h2,h3{{font-family:{LAB_TYPE["serif"]};font-weight:600;line-height:1.2;margin:0 0 .4em}}
h1{{font-size:2.3rem}} h2{{font-size:1.65rem;margin-top:2.2em}} h3{{font-size:1.15rem;margin-top:1.4em}}
p{{margin:.4em 0 .9em}} .lede{{color:var(--text-muted);font-size:1.1rem}}
.brand{{display:flex;align-items:center;gap:12px;color:var(--text-muted);font-size:.95rem;margin-bottom:18px}}
.brand svg{{width:40px;height:40px;flex:none}}
.labels{{background:var(--accent-soft);border:1px solid var(--border-strong);border-radius:14px;
padding:14px 18px;margin:18px 0 8px}}
.labels h2{{font-size:1.05rem;margin:0 0 .4em;color:var(--accent)}}
.labels ul{{margin:0;padding-left:1.2em}} .labels li{{margin:.15em 0}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;margin:16px 0}}
.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px 24px;margin:0}}
.facts dt{{color:var(--text-subtle);font-size:.85rem}} .facts dd{{margin:0;font-weight:600}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;padding:0;list-style:none;margin:.6em 0}}
.chips li{{border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:2px 10px;font-size:.88rem}}
.chips b{{font-weight:650}}
.figure{{margin:16px 0;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:14px 16px}}
figcaption{{color:var(--text-muted);font-size:.92rem;margin-bottom:6px}}
.legend{{display:flex;flex-wrap:wrap;gap:4px 18px;list-style:none;padding:0;margin:4px 0 8px;font-size:.9rem}}
.legend li{{display:flex;align-items:center;gap:6px}}
.plot{{position:relative}}
.chart{{width:100%;height:auto;display:block}}
.chart .grid{{stroke:var(--grid);stroke-width:1}} .chart .axis{{stroke:var(--border-strong);stroke-width:1}}
.chart .tick{{fill:var(--text-subtle);font-size:12px}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.rule-equal_weight{{stroke:var(--ew)}} .rule-inverse_volatility{{stroke:var(--iv)}}
.rule-buy_and_hold{{stroke:var(--bh)}}
.chart .cursor{{stroke:var(--text-subtle);stroke-width:1;stroke-dasharray:3 3}}
.chart .hit{{fill:transparent;cursor:crosshair}}
.readout{{position:absolute;top:8px;pointer-events:none;background:var(--surface);color:var(--text);
border:1px solid var(--border-strong);border-radius:10px;padding:8px 10px;font-size:.84rem;
box-shadow:0 2px 10px rgba(0,0,0,.12);min-width:230px}}
.readout b{{display:block;margin-bottom:2px}} .readout span{{display:flex;justify-content:space-between;gap:12px}}
.table-wrap{{overflow-x:auto;margin:10px 0}}
table{{border-collapse:collapse;width:100%;background:var(--surface);font-size:.92rem}}
caption{{text-align:left;color:var(--text-muted);font-size:.88rem;padding:6px 0}}
th,td{{border-bottom:1px solid var(--border);padding:7px 10px;text-align:left;vertical-align:top}}
thead th{{background:var(--muted);font-weight:600;font-size:.86rem}}
.num{{text-align:right;white-space:nowrap}} tbody th{{min-width:200px}} td.neg{{color:var(--text)}}
.sub{{color:var(--text-subtle);font-size:.8rem;font-weight:400;white-space:normal}}
.flag{{color:var(--accent);font-weight:600}}
details{{margin:12px 0}} summary{{cursor:pointer;font-weight:600}}
summary:focus-visible,.hit:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media (max-width:760px){{.cols{{grid-template-columns:1fr}} h1{{font-size:1.8rem}}}}
.note{{color:var(--text-muted);font-size:.9rem}}
footer{{color:var(--text-subtle);font-size:.85rem;margin-top:40px;border-top:1px solid var(--border);padding-top:14px}}
code{{font-family:{LAB_TYPE["mono"]};font-size:.85em}}
"""
    )


_SCRIPT = """
document.querySelectorAll('.figure').forEach(function (figure) {
  var data = JSON.parse(figure.querySelector('.chart-data').textContent);
  var svg = figure.querySelector('svg.chart'), hit = svg.querySelector('.hit');
  var cursor = svg.querySelector('.cursor'), readout = figure.querySelector('.readout');
  var fmt = function (v) { return v === null ? 'unavailable' : '$' + Math.round(v).toLocaleString('en-US'); };
  function show(evt) {
    var box = svg.getBoundingClientRect(), scale = box.width / svg.viewBox.baseVal.width;
    var sx = (evt.clientX - box.left) / scale, nearest = 0, nearestGap = Infinity;
    data.positions.forEach(function (p, i) {
      var gap = Math.abs(data.left + data.width * p / data.span - sx);
      if (gap < nearestGap) { nearestGap = gap; nearest = i; }
    });
    var px = data.left + data.width * data.positions[nearest] / data.span;
    cursor.setAttribute('x1', px); cursor.setAttribute('x2', px); cursor.setAttribute('visibility', 'visible');
    var rows = data.series.map(function (s) {
      var swatch = '<svg width="22" height="8" aria-hidden="true"><line class="line rule-' + s.rule +
        '" x1="1" x2="21" y1="4" y2="4" stroke-dasharray="' + s.dash + '"/></svg>';
      return '<span><i>' + swatch + ' ' + s.label + '</i><b>' + fmt(s.values[nearest]) + '</b></span>';
    }).join('');
    readout.innerHTML = '<b>' + data.dates[nearest] + '</b>' + rows;
    readout.hidden = false;
    var left = px * scale + 14;
    if (left + readout.offsetWidth > box.width) { left = px * scale - readout.offsetWidth - 14; }
    readout.style.left = Math.max(0, left) + 'px';
  }
  hit.addEventListener('mousemove', show);
  hit.addEventListener('mouseleave', function () { readout.hidden = true; cursor.setAttribute('visibility', 'hidden'); });
});
"""


OFFSET_NOTE = (
    "Which day a rule rebalances on is a choice. Each rebalancing rule was also run on every "
    "possible rebalancing day of its cycle (5 for weekly, 21 for monthly, 63 for quarterly), at "
    "1x costs, with the same start. The registered day (offset 0) is the headline above. The "
    "spread is the gap between the highest and the lowest total return across those days: how "
    "much of a result is the luck of the day. The days are one variant with a sensitivity "
    "family, not separate trials, and no day was chosen afterwards."
)


def _families(universe: Mapping) -> list[Mapping]:
    return [
        s
        for s in universe["strategies"]
        if s.get("offset_family") and s["offset_family"].get("applies")
    ]


def _spread_points(measure: Mapping) -> str:
    if measure["spread"] is None:
        return "unavailable"
    return f"{measure['spread'] * 100:,.1f} percentage points"


def _offsets_table(universe: Mapping) -> str:
    families = _families(universe)
    if not families:
        return ""
    rows = []
    for s in families:
        family = s["offset_family"]
        total = family["total_return"]
        stopped = family["stopped_offsets"]
        note = (
            f'<div class="sub">{_e(f"{stopped} days stopped early, so no spread")}</div>'
            if stopped
            else ""
        )
        rows.append(
            f'<tr><th scope="row">{_e(s["label"])}{note}</th>'
            f'<td class="num">{family["k"]}</td>'
            f'<td class="num">{_e(pct(total["registered"]))}</td>'
            f'<td class="num">{_e(_spread_points(total))}</td></tr>'
        )
    return (
        "<details open><summary>Rebalancing-day sensitivity</summary>"
        f'<p class="note">{_e(OFFSET_NOTE)}</p><div class="table-wrap"><table>'
        "<caption>Total return across every rebalancing day, at 1x costs. Buy and hold never "
        "rebalances, so the day changes nothing for it.</caption>"
        '<thead><tr><th scope="col">Rule</th><th scope="col" class="num">Rebalancing days (k)</th>'
        '<th scope="col" class="num">Registered day: total return</th>'
        '<th scope="col" class="num">Spread across the days</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>"
    )


GROUP_D_NOTE = (
    "Growth rate (CAGR) is per year on calendar days, with the build's cost inside it. "
    "Volatility is the yearly scale (square root of 252) of the daily net returns' standard "
    "deviation; a Sharpe ratio needs the T-bill rate series and is not shown yet. Each stress "
    "window runs from the S&P 500's closing peak to its closing low: its return and its worst "
    "fall inside the window. Before taxes."
)
TRADING_NOTE = (
    "Turnover is one way (selling 10% to buy another 10% is 10%). The build is shown on its "
    "own; the rebalances are summed per calendar year, a part year is never scaled, and the "
    "yearly figure is the mean of the full years. Participation flags a trade larger than {p} "
    "of the asset's median dollar volume over the {w} sessions before it, for a portfolio of "
    "{n}: a flag, not a capacity figure, and a fund's exchange volume understates what it can "
    "absorb."
)


def _with_group_d(universe: Mapping) -> list[Mapping]:
    return [s for s in universe["strategies"] if s.get("group_d")]


def _cagr_text(cagr: Mapping | None) -> str:
    if cagr is None or cagr["value"] is None:
        reason = None if cagr is None else cagr.get("reason")
        return "unavailable" + (f" ({reason})" if reason else "")
    text = pct(cagr["value"])
    if not cagr["annualized"]:
        text += " (part year, not annualized)"
    return text


def _share(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 100:,.0f}%"


def _flag_text(found: Mapping) -> str:
    text = f"{found['flagged']:,} of {found['assessed']:,}"
    if found["unavailable"]:
        text += f" ({found['unavailable']:,} without volume data)"
    return text


def _volatility_text(vol: Mapping) -> str:
    if vol["value"] is None:
        return f"unavailable ({vol['reason']})"
    text = f"{vol['value'] * 100:.1f}%"
    if not vol["complete"]:
        text += (
            f" (incomplete: {vol['sessions']:,} of {vol['of']:,} sessions; the moves left out "
            "across gaps can make it read too low)"
        )
    return text


def _window_text(window: Mapping) -> str:
    if window["status"] != "ok":
        return f"unavailable ({window['reason']})"
    fall = window["drawdown"]
    worst = (
        ""
        if fall is None
        else f"; worst fall {_drawdown_text({**fall, 'measure': 'maximum_drawdown' if fall['complete'] else 'observed'})}"
    )
    return f"{pct(window['return'])}{worst}"


def _group_d_tables(universe: Mapping) -> str:
    rows = _with_group_d(universe)
    if not rows:
        return ""
    windows = rows[0]["group_d"]["stress_windows"]
    head = "".join(
        f'<th scope="col" class="num">{_e(w["label"])}<div class="sub">{_e(w["start"])} to '
        f"{_e(w['end'])}</div></th>"
        for w in windows
    )
    growth_rows = "".join(
        f'<tr><th scope="row">{_e(s["label"])}</th>'
        f'<td class="num">{_e(_cagr_text(s["group_d"]["cagr"]))}</td>'
        f'<td class="num">{_e(_volatility_text(s["group_d"]["volatility"]))}</td>'
        + "".join(
            f'<td class="num">{_e(_window_text(w))}</td>' for w in s["group_d"]["stress_windows"]
        )
        + "</tr>"
        for s in rows
    )
    flag = rows[0]["group_d"]["participation"]
    trading_rows = []
    for s in rows:
        g = s["group_d"]
        turnover, part = g["turnover_by_year"], g["participation"]
        build, rebalances = part["build"], part["rebalances"]
        annual = turnover["annual_turnover"]
        trading_rows.append(
            f'<tr><th scope="row">{_e(s["label"])}</th>'
            f'<td class="num">{turnover["construction"]["turnover"] * 100:,.0f}%</td>'
            f'<td class="num">{_e("unavailable" if annual is None else f"{annual * 100:,.0f}%")}'
            f'<div class="sub">mean of {turnover["full_years"]} full years; traded '
            f"{_e(_share(turnover['annual_notional']))} a year</div></td>"
            f'<td class="num">{_e(_flag_text(build))}</td>'
            f'<td class="num">{_e(_flag_text(rebalances))}</td>'
            f'<td class="num">{_e(_q_text(rebalances["max_q"]))}</td></tr>'
        )
    trading_note = TRADING_NOTE.format(
        p=f"{flag['threshold'] * 100:g}%", w=flag["window"], n=money(flag["notional"])
    )
    return (
        "<details open><summary>Growth rate, volatility and stress windows</summary>"
        f'<p class="note">{_e(GROUP_D_NOTE)}</p><div class="table-wrap"><table>'
        "<caption>Every registered variant at 1x costs, in registered order (not ranked)."
        '</caption><thead><tr><th scope="col">Rule</th>'
        '<th scope="col" class="num">Growth rate (CAGR)</th>'
        f'<th scope="col" class="num">Volatility</th>{head}</tr></thead>'
        f"<tbody>{growth_rows}</tbody></table></div></details>"
        "<details open><summary>Trading per year and participation</summary>"
        f'<p class="note">{_e(trading_note)}</p><div class="table-wrap"><table>'
        "<caption>One-way turnover and participation flags at 1x costs.</caption><thead><tr>"
        '<th scope="col">Rule</th><th scope="col" class="num">Build turnover</th>'
        '<th scope="col" class="num">Rebalance turnover a year</th>'
        '<th scope="col" class="num">Build trades flagged</th>'
        '<th scope="col" class="num">Rebalance trades flagged</th>'
        '<th scope="col" class="num">Largest rebalance participation</th>'
        f"</tr></thead><tbody>{''.join(trading_rows)}</tbody></table></div>"
        f"{_turnover_years_table(rows)}</details>"
    )


def _q_text(q: float | None) -> str:
    return "unavailable" if q is None else f"{q * 100:.2f}%"


def _turnover_years_table(rows: list[Mapping]) -> str:
    by = {s["key"]: {r["year"]: r for r in s["group_d"]["turnover_by_year"]["years"]} for s in rows}
    years = sorted({year for found in by.values() for year in found})
    head = "".join(f'<th scope="col" class="num">{_e(s["label"])}</th>' for s in rows)
    body = []
    for year in years:
        cells, partial = [], False
        for s in rows:
            row = by[s["key"]].get(year)
            if row is None:
                cells.append('<td class="num">-</td>')
                continue
            partial |= row["partial"]
            cells.append(f'<td class="num">{row["turnover"] * 100:,.0f}%</td>')
        label = f"{year} (part)" if partial else str(year)
        body.append(f'<tr><th scope="row">{_e(label)}</th>{"".join(cells)}</tr>')
    return (
        '<details><summary>Rebalance turnover by calendar year</summary><div class="table-wrap">'
        f'<table><thead><tr><th scope="col">Year</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></details>"
    )


def _universe_section(universe: Mapping) -> str:
    chips = "".join(
        f"<li><b>{_e(m['ticker'])}</b> {_e(m['sector'])}</li>" for m in universe["members"]
    )
    return (
        f'<section id="{_e(universe["key"])}"><h2>{_e(universe["label"])}</h2>'
        f'<p>{_e(universe["selection"])}</p><ul class="chips">{chips}</ul>'
        f"{growth_chart(universe)}{_results_table(universe)}{_group_d_tables(universe)}"
        f"{_offsets_table(universe)}"
        f"{_calendar_table(universe)}"
        f"{_components_table(universe)}</section>"
    )


def _family_runs(results: Mapping) -> str:
    runs = results.get("offset_family_simulations")
    return f", and {runs:,} more on the other rebalancing days at 1x" if runs else ""


def render_html(results: Mapping, *, logo_svg: str = "", font_dir: str | None = None) -> str:
    """The report page for `results`; `font_dir` is the bundled fonts' folder beside it."""

    universes = results["universes"]
    labels = "".join(f"<li>{_e(label)}</li>" for label in results["labels"])
    shows = "".join(f"<li>{_e(item)}</li>" for item in WHAT_IT_SHOWS)
    not_shows = "".join(f"<li>{_e(item)}</li>" for item in WHAT_IT_DOES_NOT)
    builds = ", ".join(f"{u['label']}: {u['build']}" for u in universes)
    batches = sorted(
        {b["retrieved_at"] for u in universes for rows in u["lineage"].values() for b in rows}
    )
    retrieved = f"{batches[0]} to {batches[-1]}" if batches else "none"
    gaps = sum(c["missing_sessions"] for u in universes for c in u["components"])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Rebalancing Showcase</title>
<style>{_css(font_dir)}</style></head>
<body><main>
<div class="brand">{logo_svg}<span>Portfolio Lab · private exploratory report</span></div>
<h1>{_e(TITLE)}</h1>
<p class="lede">{_e(SUBTITLE)}</p>
<aside class="labels" aria-label="Read this first"><h2>Read this first</h2><ul>{labels}</ul></aside>
<div class="card"><dl class="facts">
<div><dt>Variants run</dt><dd>{results["variants_run"]} ({len(universes)} lists × {
        len(universes[0]["strategies"]) if universes else 0
    } rules), none chosen afterwards</dd></div>
<div><dt>Simulations</dt><dd>{results["simulations_run"]} (each variant at 0x, 1x and 2x costs){
        _family_runs(results)
    }</dd></div>
<div><dt>Window</dt><dd>{_e(results["window"]["start"])} to {
        _e(results["window"]["end"])
    }</dd></div>
<div><dt>Starting amount</dt><dd>{
        _e(money(results["initial_wealth"]))
    }, built from cash at one common close</dd></div>
</dl></div>
<div class="cols">
<div class="card"><h3>What this shows</h3><ul>{shows}</ul></div>
<div class="card"><h3>What it doesn't show</h3><ul>{not_shows}</ul></div>
</div>
{"".join(_universe_section(u) for u in universes)}
<section id="method"><h2>Data and method</h2>
<div class="card"><ul>
<li>{_e(results["return_label"])}.</li>
<li>Every rule runs through the approved portfolio engine: a decision uses only closes up to its
own session and trades at the next close; every list builds from cash at one common close
({_e(builds)}); costs are the provisional per-trade rates (sector funds {
        _e(results["cost_classes"]["etf"])
    }, stocks {_e(results["cost_classes"]["stock"])}),
charged on every trade, the build included.</li>
<li>Sessions without a close in the stored prices: {gaps}. A return across such a gap counts
as one return in the inverse-volatility window, and wealth is unavailable while a held asset has
no close.</li>
<li>Turnover is one way: selling 10% of the portfolio to buy another 10% is 10%. The results
table sums it over every trade, the build included; where it is shown a year, the build is apart
and the yearly figure is the mean of the full calendar years.</li>
<li>Maximum drawdown is the largest fall from a previous high, close to close. Where a session
has no value, only the largest observed decline is given, labelled as such.</li>
<li>Prices: Tiingo end-of-day data, retrieved {_e(retrieved)}, for private use only (no
redistribution). Configuration <code>{_e(results["config"]["name"])}</code>, sha256
<code>{_e(results["config"]["sha256"][:16])}</code>; code <code>{
        _e(results["code_sha"] or "uncommitted")
    }</code>; generated {_e(results["generated_at"])}.</li>
</ul></div></section>
<footer>Exploratory and private. Hindsight-selected and survivorship-biased lists, 2003-2024
only. Not evidence of strategy skill, and not investment advice.</footer>
</main><script>{_SCRIPT}</script></body></html>
"""
    return page


# --- the Markdown twin ---


def render_markdown(results: Mapping) -> str:
    """The report as Markdown (no charts; the HTML page has them)."""

    lines = [f"# {TITLE}", "", f"_{SUBTITLE}_", "", "## Read this first", ""]
    lines += [f"- {label}" for label in results["labels"]]
    lines += [
        "",
        (
            f"Variants run: {results['variants_run']}, none chosen afterwards. Simulations: "
            f"{results['simulations_run']} (each variant at 0x, 1x and 2x costs)"
            f"{_family_runs(results)}. Window: "
            f"{results['window']['start']} to {results['window']['end']}. Starting amount: "
            f"{money(results['initial_wealth'])}."
        ),
        "",
        "## What this shows",
        "",
        *(f"- {item}" for item in WHAT_IT_SHOWS),
        "",
        "## What it doesn't show",
        "",
        *(f"- {item}" for item in WHAT_IT_DOES_NOT),
    ]
    for universe in results["universes"]:
        lines += [
            "",
            f"## {universe['label']}",
            "",
            universe["selection"],
            "",
            "Holdings: "
            + ", ".join(f"{m['ticker']} ({m['sector']})" for m in universe["members"])
            + f". Common build close: {universe['build']}.",
            "",
            "Every registered variant, in registered order (not ranked), at 1x costs:",
            "",
            (
                "| Rule | $10,000 became | Total return | Maximum drawdown | Rebalances | "
                "Turnover (one way, summed, build included) | Costs paid | At 0x costs | At 2x costs |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for s in universe["strategies"]:
            sens = s["cost_sensitivity"]
            stopped = "" if s["complete"] else f" ({_stop_text(s)})"
            lines.append(
                f"| {s['label']}{stopped} | {money(s['final_wealth'])} | {pct(s['total_return'])} "
                f"| {_drawdown_text(s['drawdown'])} | {s['turnover']['rebalances']:,} "
                f"| {_turnover_text(s['turnover'])} | {money(s['costs']['paid_dollars'])} "
                f"| {money(sens['0x']['final_wealth'])} | {money(sens['2x']['final_wealth'])} |"
            )
        families = _families(universe)
        if families:
            lines += [
                "",
                (
                    f"Rebalancing-day sensitivity. {OFFSET_NOTE} Buy and hold never rebalances, so "
                    "the day changes nothing for it."
                ),
                "",
                (
                    "| Rule | Rebalancing days (k) | Registered day: total return | "
                    "Spread across the days |"
                ),
                "|---|---:|---:|---:|",
            ]
            for s in families:
                family = s["offset_family"]
                total = family["total_return"]
                stopped = family["stopped_offsets"]
                label = s["label"] + (f" ({stopped} days stopped early)" if stopped else "")
                lines.append(
                    f"| {label} | {family['k']} | {pct(total['registered'])} "
                    f"| {_spread_points(total)} |"
                )
        with_d = _with_group_d(universe)
        if with_d:
            windows = with_d[0]["group_d"]["stress_windows"]
            flag = with_d[0]["group_d"]["participation"]
            lines += [
                "",
                f"Growth rate, volatility and stress windows. {GROUP_D_NOTE}",
                "",
                "| Rule | Growth rate (CAGR) | Volatility | "
                + " | ".join(f"{w['label']} ({w['start']} to {w['end']})" for w in windows)
                + " |",
                "|---|---:|---:|" + "---:|" * len(windows),
            ]
            for s in with_d:
                g = s["group_d"]
                lines.append(
                    f"| {s['label']} | {_cagr_text(g['cagr'])} | {_volatility_text(g['volatility'])} | "
                    + " | ".join(_window_text(w) for w in g["stress_windows"])
                    + " |"
                )
            lines += [
                "",
                "Trading per year and participation. "
                + TRADING_NOTE.format(
                    p=f"{flag['threshold'] * 100:g}%", w=flag["window"], n=money(flag["notional"])
                ),
                "",
                (
                    "| Rule | Build turnover | Rebalance turnover a year | Build trades flagged | "
                    "Rebalance trades flagged | Largest rebalance participation |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
            for s in with_d:
                g = s["group_d"]
                turnover, part = g["turnover_by_year"], g["participation"]
                annual = turnover["annual_turnover"]
                lines.append(
                    f"| {s['label']} | {turnover['construction']['turnover'] * 100:,.0f}% "
                    f"| {'unavailable' if annual is None else f'{annual * 100:,.0f}%'} "
                    f"(mean of {turnover['full_years']} full years) "
                    f"| {_flag_text(part['build'])} | {_flag_text(part['rebalances'])} "
                    f"| {_q_text(part['rebalances']['max_q'])} |"
                )
        strategies = universe["strategies"]
        years = sorted({row["year"] for s in strategies for row in s["calendar_years"]})
        by = {s["key"]: {r["year"]: r for r in s["calendar_years"]} for s in strategies}
        lines += [
            "",
            "Calendar-year returns at 1x costs (a part year is never annualized):",
            "",
            "| Year | " + " | ".join(s["label"] for s in strategies) + " |",
            "|---|" + "---:|" * len(strategies),
        ]
        for year in years:
            rows = [by[s["key"]].get(year) for s in strategies]
            part = any(r and r["partial"] for r in rows)
            cells = [pct(r["total_return"]) if r else "-" for r in rows]
            lines.append(f"| {year}{' (part)' if part else ''} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Each holding on its own over the same span, before costs (context, not a rule):",
            "",
            "| Holding | Sector | Total return | Maximum drawdown |",
            "|---|---|---:|---:|",
        ]
        for c in universe["components"]:
            lines.append(
                f"| {c['ticker']} | {c['sector']} | {pct(c['total_return'])} "
                f"| {_drawdown_text(c['drawdown'])} |"
            )
    lines += [
        "",
        "## Data and method",
        "",
        f"- {results['return_label']}.",
        (
            "- The approved portfolio engine: next-close execution, one common build from cash, "
            "provisional per-trade costs on every trade, the build included."
        ),
        (
            "- Turnover is one way. The results table sums it over every trade, the build "
            "included; a yearly figure keeps the build apart and is the mean of the full years."
        ),
        (
            f"- Configuration `{results['config']['name']}` (sha256 "
            f"`{results['config']['sha256'][:16]}`), code `{results['code_sha'] or 'uncommitted'}`, "
            f"generated {results['generated_at']}."
        ),
        "- Tiingo end-of-day data for private use only (no redistribution).",
        "",
    ]
    return "\n".join(lines)


def check_wording(text: str) -> list[str]:
    """Words the report never uses (it states what happened, never who 'won')."""

    return sorted({match.group(0).lower() for match in FORBIDDEN_WORDS.finditer(text)})
