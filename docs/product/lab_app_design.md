# The Lab app: simple screens for the Stock Lab

Status (2026-10-23): **design approved by the owner.**
- The owner approved the clickable mockup "as is" (`mockups/lab_app_mockup.html`, made-up
  data) on 2026-10-23, after viewing it on this Mac.
- On the same day the owner chose build approach B (below) and chose to run the Mac app from
  the runner copy (`~/stock-agent-runner`), so macOS no longer asks for Desktop access.

## 1. Why

The Stock Lab went live on 2026-10-20 and records real end-of-day forecasts for AMD, MRVL, MU
and NVDA with a frozen B1 model. The owner opened the Portfolio Lab from the new Mac app and
found it "so complicated to use": the Streamlit pages are research pages, dense with terms,
tables and caveats. The owner asked for something simple, clean and glanceable, in the spirit
of a phone investing app (Robinhood was the reference), that stays honest, private and
non-advisory.

Success: the owner opens the app in the morning and, at a glance, knows whether last night's
update ran, sees each stock's last close and move, and reads the recorded outlook. Every
number is one tap away from a plain-English explanation. The research pages stay available
one level down, for learning.

## 2. Goals, non-goals and anti-goals

**Goals**
- Four screens: Home (watchlist), Stock detail, History, Health, plus More (appearance,
  research pages, about).
- Plain words on every main screen; explanations behind a tap.
- Dark by default, with Light on request; big numbers and generous spacing.
- Works at phone width and in a desktop browser window.

**Non-goals**
- Trading, orders, alerts or notifications (the Stock Lab has its own notifications).
- New forecasts, new models or any new calculation on prices.
- Replacing the research pages: they stay, unchanged, under More.
- Hosting or sharing: the app runs on this Mac only.

**Anti-goals** (a design that does any of these has failed, even if it works)
- Anything that reads as advice: no Buy or Sell buttons, no "recommended", no ranking or
  "best" sort. The watchlist keeps the store's own order.
- Green or red on an outlook. Green and red mean only that a price went up or down.
- A number that was not stored: probabilities are shown exactly as recorded (rounded only
  for display), never recomputed, renormalized, smoothed or carried forward.
- A stale or missing outlook dressed up as today's: missing data gets an honest empty state
  with its reason.
- Implying an end-of-day outlook knows about later news.

## 3. Screens (the approved mockup is the reference for layout and copy)

- **Home.** The status line, then one row per watchlist stock with the ticker, company name,
  a one-month line, the last close, that session's move (green or red), and an outlook chip
  for the next session ("Thu ↑ 47% · ↓ 42%"). A tap on the chip explains it. The footer
  states that the outlooks are experimental and that this is not advice and not a brokerage.
- **Stock detail.**
  - A large price chart with 1W, 1M, 3M and 1Y ranges. Drag across it to read any day's close
    and that day's move.
  - The next session's outlook (Up / Flat / Down bar), and the five-session total, with each
    day on its own one tap away.
  - "How reliable has this been?": the development record, pooled over the four stocks, and
    the live record's progress toward Model Health's 50-session minimum.
  - Context chips from the forecast record (streak, past 4 weeks, swings).
  - A link to the research page.
- **History.** Each evening's next-session outlook and what happened (outcome and move from
  the stored score), as a timeline. Missed evenings are shown as missed and never filled in.
- **Health.** One green, amber or red indicator with one sentence; the Stock Lab's checks
  behind "Show details", each with its Model Health check id; a link to the full Model
  Health page.
- **More.** Dark / Light, the research pages, and "About" (private, not a brokerage, not
  advice, not yet proven).

### Home in detail (PR 1)

**Status line** (the first matching rule):

| Condition | Level | Words (example) |
|---|---|---|
| The record cannot be opened (the store, the timing or the calendar refused) | red | "The forecast record can't be read." |
| The ledger check failed | red | "The forecast record didn't pass its check." |
| A stock's own record failed its check | red | "A forecast record didn't pass its check." |
| The latest run says real outlooks are not activated | neutral | "Outlooks haven't started yet." |
| No real daily-job run is recorded | neutral | "No update has run yet." |
| The latest run finished after the last success and was not `ok` | amber | "The last update didn't finish (last night, 8:40 PM)." / "...was only partly done..." / "...recorded nothing..." |
| A stock is stale and not pending (below) | amber | "Last night's update didn't run." / "Last night's update didn't record 1 of 4 stocks." |
| An outlook is withheld | amber | "1 outlook was withheld." |
| Otherwise | green | "Updated last night, 8:31 PM" |

Tapping the status line shows the details: the last successful update, the latest run in
words ("finished", "only partly done"), and the briefing's note about closed days.

- "Last night" is used only when that is true in New York time. Otherwise the app says
  "tonight", the weekday ("Friday") or the date.
- Times are New York time, with "ET" added when the Mac's own time zone differs.
- **Pending:** from 20:00 New York on a session day (the briefing's `advance` state), a stock
  whose newest forecast is one session behind is waiting for tonight's update. It is not
  missed. Its chip reads "Fri outlook after tonight's update". Pending ends once a run for
  tonight's session has finished: a stock it left out is then "not recorded".

**Outlook chip** (from the briefing's per-stock status):

| Briefing status / reason | Chip |
|---|---|
| `fresh` | "Thu ↑ 47% · ↓ 42%", the next session's increase and decrease as recorded |
| `stale`, pending | "Fri outlook after tonight's update" (neutral) |
| `stale` | "Thu outlook not recorded" (amber) |
| `unavailable`, `no_recorded_forecast` | "No outlook yet" (neutral) |
| `unavailable`, `refused` | "No outlook: the model gave none" (neutral) |
| `unavailable`, `record_check_failed` or `ledger_unverified` | "Outlook can't be verified" (red) |
| `unavailable`, `as_of_not_prior_session` or `target_mismatch` | "Outlook withheld" (amber) |

The chip's sheet shows the three probabilities, the band in words (from the record), the
forecast's as-of close and when it was recorded, and the briefing's own explanation.

**Price** (from the recent-prices reader, decision A6):
- The latest served close and its session, and that session's `change` exactly as the reader
  gives it. Nothing else is computed from the prices, so there is no dollar change and no
  change over a period.
- The line is the split-consistent `level` of the last 22 stored sessions, drawn with a dot in
  the move's colour. A gap stays a gap.
- When no prices are stored, or the store is refused, the row says so ("No prices stored
  yet", "Prices can't be read") and shows no number.
- A close older than the other stocks' names its own day ("on Tue, Oct 20").

## 4. Data and point-in-time rules

- **One source of truth for status and outlooks:** `stock_lab.briefing.report.build_briefing`,
  prepared at the request time (`briefing.timing.briefing_timing(now, calendar)`).
  - It reads the research store read-only (`ReadOnlyRecords`, SQLite `mode=ro`).
  - It counts only the records that existed by then.
  - It never recomputes a probability.
  - It gives fresh, stale or unavailable per stock, with reasons.
- **Prices:** `stock_lab.inference.recent.read_recent(symbol, count=22, retrieved_by=now)`.
  It shows observed closes only, opened read-only (decision A6).
- **No new calculation on 2025+ prices.** Context (streak, past 4 weeks, swings) comes from
  the forecast record's `signals`, as the frozen job recorded it. The chart adds nothing up.
  The mockup's "Why no total for the period?" sheet explains this.
- **Nothing is written** by the app: not the stores, not the ledger, and not the checkout. The
  briefing's existing note about SQLite's WAL side files applies unchanged.

## 5. Architecture (option B)

Chosen by the owner on 2026-10-23 over A (restyle Streamlit) and C (a pywebview window, a
possible later decision):
- B uses Python's standard-library HTTP server and plain HTML, CSS and JavaScript, with no
  new dependency, no build step and nothing fetched from another host.
- C can wrap B later without rework.

| Piece | Where | Role |
|---|---|---|
| View models | `src/stock_agent/app/web/home.py` (later `detail.py`, `history.py`, `health.py`) | Pure functions from the briefing and the prices to JSON-ready dicts of display-ready words and the raw recorded values |
| Copy and formatting | `src/stock_agent/app/web/words.py` | Every user-facing sentence, time words, percentages |
| Server | `src/stock_agent/app/web/server.py` | Standard-library `ThreadingHTTPServer`, read-only routes, security headers |
| Front end | `src/stock_agent/app/web/static/` | `index.html`, `app.css`, `app.js`, the mark; the bundled Source Sans 3 is served from `app/static/fonts/` |
| Launcher | `scripts/run_lab_app.py` | Wires the NYSE rules calendar, the recent-prices reader, the records and a clock into the server; `--demo` serves a synthetic replay from a temporary store |

**Boundaries.**
- No module under `src/` may import the rules calendar or the inference store's reader. The
  launcher script passes them in, exactly as `scripts/stock_lab_briefing.py` does.
- So the launcher is added, with a comment, to `SCRIPT_CONSUMERS` in
  `tests/test_prospective_boundaries.py` (rules calendar only) and in
  `tests/test_stock_lab_inference_boundaries.py` (`read_recent` only). This is the reviewed
  addition the owner approved with option B.
- The new package imports only what the app boundary already allows (the briefing's report
  and timing).

**Security** (local, single user, read-only):
- Binds 127.0.0.1 only; refuses to start on any other address.
- DNS-rebinding protection: a request whose Host header is not `127.0.0.1:<port>` or
  `localhost:<port>` gets 421 and no content.
- GET and HEAD only (405 otherwise), over HTTP/1.0 or 1.1, with a path as the request target
  (400 otherwise). Fixed routes; static files from a fixed allowlist (no path from the request
  reaches the file system).
- A request another website starts in the background gets 403 before any work. The app
  tells by the browser's Fetch metadata (`Sec-Fetch-Site`) or a foreign `Origin`. Only a
  top-level visit to the page may come from a link elsewhere.
- A connection that sends nothing is dropped after 15 seconds. A failed request logs only
  its error type.
- Headers on every response:
  - `Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`
  - `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`
  - `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Resource-Policy: same-origin`
  - `Cache-Control: no-store`
- No CORS headers, no cookies, no credentials, no secrets in the environment.
- The page builds its DOM with `textContent` and CSSOM (no `innerHTML` of data, no inline
  styles), so the strict policy holds.
- Errors return a short JSON message; no traceback, no path.

**Mac app and the Desktop prompt** (a later PR):
- The launcher (`scripts/portfolio_lab_app.py`) will start the Lab app, plus the research
  pages on a second port.
- The app will be built with `--checkout ~/stock-agent-runner`. The runner copy is outside
  Desktop, so macOS asks nothing, and it already has every package the app needs.
- The app writes nothing into the runner copy (tested). Its evening job already refuses a
  dirty checkout, so a stray write would show as a missed evening, never a wrong forecast.

## 6. Visual design

The tokens are in the mockup's `:root` and move into `app.css`. "Ember" dark is the default:

| Role | Dark | Light |
|---|---|---|
| Background | `#121110` | `#FBF8F3` |
| Surface, raised | `#1C1A18`, `#262320` | `#FFFFFF`, `#F3EEE6` |
| Text, muted, subtle | `#F3EEE6`, `#B9AFA2`, `#968C80` | `#1D1B18`, `#544E46`, `#6B6258` |
| Price line (ivory, as in the logo) | `#F6F1E9` | `#2A2622` |
| Price went up / down | `#3FD68B` / `#FF7262` | `#0B7A41` / `#C8352A` |
| Outlook (mist; never green or red) | `#AFC3D6`, Flat `#6C665E` | `#3E5A75`, Flat `#B9B0A4` |
| Status ok / warning / problem / note | `#8FD3A0` / `#F2BC72` / `#F4A7AE` / `#97CAD7` | `#1E6633` / `#8A4B0F` / `#99202C` / `#26505D` |
| Brand glow (logo and the chart's "now" point only) | `#DE8C66` | `#9A4325` |

- Every text colour is at least 4.5:1 on every surface, and up and down are at least 3:1 (in
  fact 5:1 or more). A test checks the pairs.
- Direction is never carried by colour alone: ▲ / ▼ and +/− carry it too.
- Font: the bundled Source Sans 3, tabular figures for numbers.
- Motion only on sheets (200 ms), and off under `prefers-reduced-motion`.

This updates `ux_blueprint.md` section 8 ("gains and losses stay neutral") at the owner's
request: green and red now mark past price moves, and only those.

## 7. Delivery

Small PRs against `main`, each green on `scripts/verify.py`, reviewed by the frontend,
test and security reviewers:
1. **Home.** The server, the Home view model, the static front end (Home and More), the
   launcher with `--demo`, the boundary additions, this document and the mockup.
2. **Stock detail.**
3. **History and Health.**
4. **Mac app.** The launcher starts the Lab app and the research pages; build from the runner
   copy; runbook steps for the owner.

## 8. Testing

- View models: pytest on fixture stores built with the existing helpers. This covers every
  status-line rule and chip state, the A6 rule (no value derived from prices), exact
  probabilities, the time words (last night, tonight, weekday, ET), and empty states.
- Server: request handling without sockets (Host check, methods, routes, headers, errors);
  one end-to-end test on a free loopback port, as `tests/test_local_server_hosts.py` does.
- Static files: no external URL, no inline script or style, no `innerHTML` of data, no advice
  words (the app's existing advice scan), and token contrast pairs.
- Launcher: `--demo` serves synthetic data end to end; the real path is wired with injected
  fakes; nothing is written to the checkout.
- No test touches the network, `data/`, `reports/` or the private stores.

## 9. Left to the builder, and the calls made in PR 1

- The transition port (the Streamlit Lab keeps 8502 until PR 4): 8510.
- Exact JSON field names, the module split inside `app/web/`, and the JavaScript structure,
  as long as the page stays a thin renderer of server-prepared words.
- Home re-fetches when the window comes back into view. It never polls.

Calls made while building PR 1, for the owner to check:
- The status table above gained the red, neutral and amber rows a builder review found
  missing. Without them, a green status line could sit above a red or amber chip.
- Times are New York time, and "ET" is added only when this Mac's own zone, at that moment,
  differs.
- The synthetic demo (`--demo`) shows the demo stocks' own names.
- The launcher claims its port before the demo's replay, so a busy port is refused at once.
- In the mockup's welcome sheet, "There is nothing to buy or sell here" became the Home
  footer's "Nothing here can be bought or sold", which passes the app's advice scan.
- From the frontend review:
  - A chance that would round to 0% or 100% reads "<1%" or ">99%".
  - The footer says "after each trading day's close", since the update doesn't run on
    market holidays.
  - Research-style reasons in a sheet are labelled "Technical detail".
  - The page shows when it last read the records ("Checked 7:40 AM") and reads them again
    when its window comes back into focus.
  - "Not yet proven" no longer names a start date in the page itself.

## 10. Open for later PRs

- **The one-month lines' scale (owner decision; frontend review, 2026-10-23).** Each line is
  scaled to its own lowest and highest close, as in the approved mockup. So a calm stock
  that moved 2% in a month looks as jagged as one that moved 10%.
  - Options: one shared scale for all rows, or a minimum height (for example ±2% around the
    last close).
  - Either is drawing only, with no value computed from prices. Today's behaviour stays
    until the owner chooses.

- **A per-launch secret (owner decision; security review, 2026-10-23).** Any program or
  macOS account on this Mac can read the app's loopback address, while the stores
  themselves are readable by the owner only. The data is low in sensitivity, and the app
  has one user.
  - Proposed for PR 4: the Mac app opens the page with a fresh token, which the page sends
    with each data request.
  - That replaces this design's "no cookies, no credentials" only if the owner agrees.

- **Health (PR 3).** The Home status line is live, but Model Health's summary is written
  once a day (07:30), and two Stock Lab checks (input drift, the ticker check) exist only
  there.
  - Proposed: Health's headline follows the same live rules as the status line, so the two
    never disagree.
  - The summary's checks are shown below it as details, with the summary's time. A missing
    summary, or one more than 26 hours old, is named as such.
  - The app keeps importing only `health.model` and `health.view`.
- **History (PR 3).** History needs every evening's forecast, its stored score and the
  missed evenings. Today those rules live in `briefing.report` (private helpers) and
  `stock_lab.job` (outside the app's allowed imports).
  - Proposed: a small public, read-only history reader in `stock_lab.briefing`, reusing
    those rules rather than copying them.
  - Any change to the app's import boundary goes to the owner before it is built.
  - History shows the live record. A recalculation that replaced a forecast is what
    History shows, as in the research page's history table.
