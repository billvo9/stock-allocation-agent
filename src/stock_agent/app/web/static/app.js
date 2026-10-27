// The Lab app's page: a thin renderer of /api/home, /api/stock/<ticker>,
// /api/history, /api/health and /api/strategies. Every number, and every sentence about the data
// (status, outlooks, the chart's days, results, checks, explanations), comes from
// the server already worded; only the page's own fixed labels (the tabs, More,
// About, the back button, the Health badge's words and the error messages) are
// written here. It builds elements with textContent (never
// innerHTML) and sets no inline style: shapes are SVG drawn to their box's
// measured size, so the page's strict content security policy holds.
"use strict";

(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const THEME_KEY = "lab-theme";
  const KEY_STORE = "lab-key";  // this launch's key, kept for this tab only (sessionStorage)
  const KEY = /^#key=([A-Za-z0-9_-]{32,128})$/;
  const TICKER = /^#stock\/([A-Z0-9][A-Z0-9_-]{0,19})$/;  // the server's own name rule
  const state = {
    tab: "home", home: null, problem: null, lastFocus: null, dirty: false, asked: 0,
    stock: null, detail: null, detailProblem: null, askedDetail: 0,
    range: null, eachDay: false, fromHome: false, homeScroll: 0, lastFocusKey: null, refreshed: 0,
    key: null,
    // History and Health: their data, their problem, and the latest request for each.
    history: null, health: null, strategies: null,
    // Strategies: the lines hidden on each list's chart (by label) and where its readout is.
    hiddenLines: {}, readAt: {},
    problems: { history: null, health: null, strategies: null },
    askedTab: { history: 0, health: 0, strategies: 0 },
    hfilter: "All", details: false,
  };
  let patterns = 0;  // unique ids for the Down segments' hatching

  // --- small builders ---------------------------------------------------------

  // A true attribute is present and a false one absent, except ARIA states,
  // which are always written out ("true" / "false").
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      const aria = key.startsWith("aria-");
      if (value === null || value === undefined || (value === false && !aria)) continue;
      if (key === "class") el.className = value;
      else el.setAttribute(key, value === true && !aria ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  function svg(tag, attrs) {
    const el = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs || {})) el.setAttribute(key, String(value));
    return el;
  }

  const arrow = (direction) => (direction === "up" ? "▲ " : direction === "down" ? "▼ " : "");

  // Shapes drawn to their box's width: redrawn whenever the box is resized.
  const drawers = new Map();
  const resizer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const draw = drawers.get(entry.target);
      if (draw) draw(Math.round(entry.contentRect.width));
    }
  });

  function fitted(box, draw) {
    drawers.set(box, draw);
    resizer.observe(box);
    return box;
  }

  // The watchlist's lines share one vertical scale (the owner's choice, 2026-10-23): each
  // line is placed by the logarithm of its levels against its own latest one, and every
  // row gets the widest line's span, so a calmer stock draws a flatter line and a rise and
  // an equal fall look the same size. Drawing only: these positions are never shown, kept
  // or sent anywhere.
  const levels = (line) => line.filter((v) => typeof v === "number" && v > 0);

  function sharedSpan(stocks) {
    let span = 0;
    for (const stock of stocks) {
      const values = levels(stock.price.line || []);
      if (values.length < 2) continue;
      span = Math.max(span, Math.log(Math.max(...values) / Math.min(...values)));
    }
    return span;
  }

  // The one-month line: the reader's levels, broken at a gap, with a dot in the
  // colour of the latest move, on the watchlist's shared scale (`span`).
  function spark(line, direction, span) {
    const w = 96, h2 = 38, pad = 5;
    const box = svg("svg", { class: "spark", viewBox: `0 0 ${w} ${h2}`, "aria-hidden": "true" });
    const values = levels(line);
    if (values.length < 2) return box;
    const latest = values[values.length - 1];
    const at = (v) => Math.log(v / latest);  // drawing only: a height, never a number shown
    const lo = at(Math.min(...values)), hi = at(Math.max(...values));
    const middle = (lo + hi) / 2, half = Math.max(span || 0, hi - lo) / 2;
    const x = (i) => pad + (i / (line.length - 1)) * (w - 2 * pad);
    const y = (v) => pad + (half === 0 ? 0.5 : 0.5 - (at(v) - middle) / (2 * half)) * (h2 - 2 * pad);
    let d = "", pen = false, last = null;
    line.forEach((v, i) => {
      if (typeof v !== "number" || v <= 0) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
      last = [x(i), y(v)];
    });
    box.append(svg("path", { class: "spark-line", d }));
    if (last) {
      const tone = direction === "up" || direction === "down" ? direction : "";
      box.append(svg("circle", { class: `spark-halo ${tone}`, cx: last[0], cy: last[1], r: 5 }));
      box.append(svg("circle", { class: `spark-dot ${tone}`, cx: last[0], cy: last[1], r: 2.6 }));
    }
    return box;
  }

  // An Up / Flat / Down bar: three segments as wide as the recorded chances.
  // Outlooks use the mist hue only; Down is hatched, so the shape carries it too.
  function outlookBar(bar, height, quiet) {
    const box = quiet
      ? h("div", { class: "obar", "aria-hidden": "true" })
      : h("div", { class: "obar", role: "img", "aria-label": bar.label });
    const id = `hatch-${++patterns}`;
    return fitted(box, (width) => {
      const gap = 3, room = Math.max(0, width - 2 * gap);
      const values = [bar.recorded.increase, bar.recorded.stable, bar.recorded.decrease];
      const total = values.reduce((a, b) => a + b, 0) || 1;
      const art = svg("svg", { viewBox: `0 0 ${Math.max(width, 1)} ${height}`, width, height, "aria-hidden": "true" });
      const pattern = svg("pattern", { id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
      pattern.append(svg("rect", { class: "hatch", width: 3, height: 6 }));
      const defs = svg("defs");
      defs.append(pattern);
      art.append(defs);
      let x = 0;
      ["up", "flat", "down"].forEach((kind, i) => {
        const w = (values[i] / total) * room;  // drawing only: the shares of the bar's width
        if (w > 0) {
          const attrs = { class: `seg ${kind}`, x: x.toFixed(2), y: 0.75, width: Math.max(w, 2).toFixed(2), height: height - 1.5, rx: 3.5 };
          if (kind === "down") attrs.fill = `url(#${id})`;
          art.append(svg("rect", attrs));
        }
        x += w + gap;
      });
      box.replaceChildren(art);
    });
  }

  function legend(bar) {
    return h("div", { class: "olegend num", "aria-hidden": "true" },
      bar.legend.map(([key, value]) => h("div", null, h("span", { class: "k" }, key), h("span", { class: "v" }, value))));
  }

  // Progress toward a minimum (the live record's trading days).
  function meter(found) {
    const box = h("div", { class: "meter", role: "img", "aria-label": found.label });
    return fitted(box, (width) => {
      const art = svg("svg", { viewBox: `0 0 ${Math.max(width, 1)} 6`, width, height: 6, "aria-hidden": "true" });
      art.append(svg("rect", { class: "meter-bar", width, height: 6, rx: 3 }));
      const done = found.max ? (found.value / found.max) * width : 0;
      if (done > 0) art.append(svg("rect", { class: "meter-done", width: Math.max(done, 6).toFixed(2), height: 6, rx: 3 }));
      box.replaceChildren(art);
    });
  }

  // --- the explanation sheet --------------------------------------------------

  function background(off) {
    for (const id of ["ribbon", "app"]) {
      const el = document.getElementById(id);
      if (off) el.setAttribute("inert", "");
      else el.removeAttribute("inert");
    }
  }

  function openSheet(sheet) {
    const box = document.getElementById("sheet");
    state.lastFocus = document.activeElement;
    state.lastFocusKey = keyOf(document.activeElement);
    const parts = [h("div", { class: "grab" }), h("h3", { id: "sheet-title" }, sheet.title)];
    if (sheet.rows && sheet.rows.length) {
      const [first, second] = sheet.columns || ["", ""];
      parts.push(
        h("table", { class: "num" },
          h("thead", null, h("tr", null, h("th", { scope: "col" }, first), h("th", { scope: "col" }, second))),
          h("tbody", null, sheet.rows.map(([label, value]) => h("tr", null, h("td", null, label), h("td", null, value)))))
      );
    }
    for (const text of sheet.paragraphs || []) parts.push(h("p", null, text));
    parts.push(h("button", { class: "close", type: "button", "data-close": true }, "Got it"));
    box.replaceChildren(...parts);
    box.hidden = false;
    document.getElementById("scrim").hidden = false;
    background(true);  // focus and screen readers stay in the sheet
    box.scrollTop = 0;
    box.focus({ preventScroll: true });
  }

  function closeSheet() {
    document.getElementById("sheet").hidden = true;
    document.getElementById("scrim").hidden = true;
    background(false);
    if (state.dirty) { state.dirty = false; render(); }
    const back = state.lastFocus && document.contains(state.lastFocus) ? state.lastFocus
      : keyed(state.lastFocusKey) || document.getElementById("screen");
    back.focus({ preventScroll: true });
  }

  const sheetOpen = () => !document.getElementById("sheet").hidden;

  function sheetButton(className, label, sheet, key, ...extra) {
    const button = h("button", { class: className, type: "button", "data-key": key }, label, ...extra);
    button.addEventListener("click", () => openSheet(sheet));
    return button;
  }

  // --- Home -------------------------------------------------------------------

  function statusLine(status) {
    const button = h("button", { class: `status ${status.level}`, type: "button", "data-key": "status" },
      h("span", { class: "dot", "aria-hidden": "true" }),
      h("span", null, status.words),
      h("span", { class: "sr-only" }, " Show details."),
      h("span", { class: "chev", "aria-hidden": "true" }, "›"));
    button.addEventListener("click", () => openSheet({ title: status.title, paragraphs: status.details }));
    return button;
  }

  function outlookChip(stock) {
    const o = stock.outlook;
    const label = `${o.label} Explain.`;
    let chip;
    if (o.chip) {
      chip = h("button", { class: "chip", type: "button", "aria-label": label, "data-key": `chip-${stock.ticker}` },
        h("span", { class: "lbl" }, o.day),
        h("span", { class: "fc num" }, `↑ ${o.chip.up}`),
        h("span", { class: "lbl" }, "·"),
        h("span", { class: "fc num" }, `↓ ${o.chip.down}`),
        h("span", { class: "info" }, "ⓘ"));
    } else {
      chip = h("button", { class: `chip empty ${o.level}`, type: "button", "aria-label": label, "data-key": `chip-${stock.ticker}` },
        o.words, h("span", { class: "info" }, "ⓘ"));
    }
    chip.addEventListener("click", () => openSheet(o.sheet));
    return chip;
  }

  function stockRow(stock, span) {
    const p = stock.price;
    const cells = p.available
      ? [
          h("span", { class: "tk" }, stock.ticker),
          spark(p.line, p.direction, span),
          h("span", { class: "px num" }, p.close_words),
          h("span", { class: "nm" }, stock.name || ""),
          h("span", { class: `chg num ${p.direction}` }, h("span", { "aria-hidden": "true" }, arrow(p.direction)), p.change_words),
          p.older ? h("span", { class: "when" }, `on ${p.session_words}`) : null,
        ]
      : [
          h("span", { class: "tk" }, stock.ticker),
          h("span", { class: "px none" }, p.words),
          h("span", { class: "nm" }, stock.name || ""),
        ];
    const link = h("a", { class: "row-main", href: `#stock/${stock.ticker}`, "data-key": `row-${stock.ticker}`, "aria-label": stock.open_label }, cells);
    link.addEventListener("click", (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;  // a new tab or window
      state.fromHome = true;
      state.homeScroll = window.scrollY;
    });
    return h("li", { class: "row" }, link, h("div", { class: "look" }, outlookChip(stock)));
  }

  function home(data) {
    const explain = h("button", { class: "textbtn", type: "button", "data-key": "explain" }, data.explain.title);
    explain.addEventListener("click", () => openSheet(data.explain));
    const notes = [data.prices_note, data.lines_note, data.checked].filter(Boolean).join(" ");
    const span = sharedSpan(data.stocks);
    return [
      h("h1", { class: "sr-only" }, "Home"),
      statusLine(data.status),
      notes ? h("p", { class: "asof" }, notes) : null,
      h("div", { class: "eyebrow-row" },
        h("h2", { class: "eyebrow" }, "Watchlist"),
        data.scale_note ? h("span", { class: "scale-note" }, data.scale_note) : null),
      h("ul", { class: "list", role: "list" }, data.stocks.map((stock) => stockRow(stock, span))),
      h("p", { class: "foot" }, data.footer, " ", explain),
    ];
  }

  // --- Stock detail: the chart ----------------------------------------------------

  // The reader's split-consistent levels as a line (broken at a gap), and each
  // day's close, change and notes, exactly as served, in the readout above it.
  function priceChart(chart, readout) {
    const ranges = chart.ranges;
    const chosen = ranges.find((r) => r.key === state.range) || ranges.find((r) => r.key === chart.default) || ranges[ranges.length - 1];
    const points = chart.points.slice(chosen.first);
    const latest = Math.max(0, chart.latest - chosen.first);
    const box = h("div", { class: "chart", tabindex: "0", role: "slider", "aria-label": chart.label, "data-key": "chart",
      "aria-valuemin": 0, "aria-valuemax": points.length - 1 });
    const H = 230, top = 28, bottom = 14, pad = 6;
    let geometry = null, current = latest;

    function show(i) {
      current = i;
      const point = points[i];
      readout(point);
      box.setAttribute("aria-valuenow", i);
      box.setAttribute("aria-valuetext", point.label);
      if (!geometry) return;
      const { x, y, cursor, dot, tag, width } = geometry;
      const off = i === latest;
      for (const el of [cursor, dot, tag]) el.classList.toggle("off", off);
      if (off) return;
      cursor.setAttribute("x1", x(i));
      cursor.setAttribute("x2", x(i));
      if (typeof point.level === "number") {
        dot.setAttribute("cx", x(i));
        dot.setAttribute("cy", y(point.level));
      } else dot.classList.add("off");
      tag.setAttribute("x", Math.min(Math.max(x(i), 48), width - 48));
      tag.textContent = point.day;
    }

    function draw(width) {
      if (width < 2) return;
      const n = points.length;
      const levels = points.map((p) => p.level).filter((v) => typeof v === "number");
      const lo = Math.min(...levels), hi = Math.max(...levels);
      const x = (i) => (n === 1 ? width / 2 : pad + (i / (n - 1)) * (width - 2 * pad));
      const y = (v) => (hi === lo ? top + (H - top - bottom) / 2 : top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom));
      let line = "", area = "", run = [];
      const close = () => {
        if (run.length > 1) area += `M${run[0][0]},${H}L${run.map(([a, b]) => `${a},${b}`).join("L")}L${run[run.length - 1][0]},${H}Z`;
        run = [];
      };
      points.forEach((p, i) => {
        if (typeof p.level !== "number") { close(); return; }
        const at = [x(i).toFixed(2), y(p.level).toFixed(2)];
        line += `${run.length ? "L" : "M"}${at[0]},${at[1]}`;
        run.push(at);
      });
      close();
      const art = svg("svg", { viewBox: `0 0 ${width} ${H}`, width, height: H, "aria-hidden": "true" });
      const gradient = svg("linearGradient", { id: "chart-fill", x1: 0, y1: 0, x2: 0, y2: 1 });
      gradient.append(svg("stop", { offset: 0, class: "fill-top" }), svg("stop", { offset: 1, class: "fill-bottom" }));
      const defs = svg("defs");
      defs.append(gradient);
      art.append(defs, svg("path", { class: "chart-area", d: area, fill: "url(#chart-fill)" }), svg("path", { class: "chart-line", d: line }));
      const cursor = svg("line", { class: "chart-cursor off", y1: top - 10, y2: H });
      const dot = svg("circle", { class: "chart-point off", r: 5 });
      const tag = svg("text", { class: "chart-tag off", y: 12, "text-anchor": "middle" });
      art.append(cursor);
      const now = points[latest];
      if (now && typeof now.level === "number") {
        art.append(svg("circle", { class: "chart-glow", cx: x(latest), cy: y(now.level), r: 13 }),
          svg("circle", { class: "chart-now", cx: x(latest), cy: y(now.level), r: 4.5 }));
      }
      art.append(dot, tag);
      box.replaceChildren(art);
      geometry = { x, y, cursor, dot, tag, width };
      show(current);
    }

    const at = (event) => {
      const rect = box.getBoundingClientRect();
      const share = (event.clientX - rect.left - pad) / Math.max(rect.width - 2 * pad, 1);
      return Math.max(0, Math.min(points.length - 1, Math.round(share * (points.length - 1))));
    };
    box.addEventListener("pointermove", (event) => show(at(event)));
    box.addEventListener("pointerdown", (event) => show(at(event)));
    box.addEventListener("pointerleave", () => show(latest));
    box.addEventListener("pointercancel", () => show(latest));  // a touch that became a scroll
    box.addEventListener("blur", () => show(latest));
    box.addEventListener("keydown", (event) => {
      const moves = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 };
      let next = null;
      if (event.key in moves) next = current + moves[event.key];
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = points.length - 1;
      if (next === null) return;
      event.preventDefault();
      show(Math.max(0, Math.min(points.length - 1, next)));
    });
    show(latest);
    return { box: fitted(box, draw), chosen };
  }

  function chartBlock(chart, readout) {
    const block = h("div", { class: "chart-block" });
    const build = () => {
      const { box, chosen } = priceChart(chart, readout);
      const buttons = chart.ranges.map((r) => {
        const button = h("button", { type: "button", "aria-pressed": r.key === chosen.key, "aria-label": r.name, "data-key": `range-${r.key}` }, r.label);
        button.addEventListener("click", () => {
          state.range = r.key;
          build();
          block.querySelector(`[aria-pressed="true"]`).focus({ preventScroll: true });
        });
        return button;
      });
      for (const old of block.querySelectorAll(".chart")) { drawers.delete(old); resizer.unobserve(old); }
      block.replaceChildren(
        box,
        h("div", { class: "ranges", role: "group", "aria-label": "Chart range" }, buttons),
        h("p", { class: "range-note" }, `${chosen.note} ${chart.drag} `, sheetButton("textbtn", chart.why.words, chart.why.sheet, "why")));
    };
    build();
    return block;
  }

  // --- Stock detail: the cards ------------------------------------------------------

  function card(title, id, ...children) {
    return h("section", { class: "card", "aria-labelledby": id }, h("h2", { id }, title), ...children);
  }

  function outlookCard(o) {
    if (o.state !== "fresh") {
      return h("section", { class: `card empty-card ${o.level}` },
        h("h2", null, o.title), h("p", null, o.words), sheetButton("textbtn", o.details, o.sheet, "outlook-details"));
    }
    return h("section", { class: "card", "aria-labelledby": "o-h" },
      h("h2", { id: "o-h" }, o.title, " ", sheetButton("pill", o.pill.words, o.pill.sheet, "experimental")),
      h("p", { class: "sub" }, o.sub),
      outlookBar(o.bar, 14), legend(o.bar),
      h("p", { class: "def" }, o.definition),
      o.notes.map((text) => h("p", { class: "note" }, text)));
  }

  function fiveCard(five) {
    const parts = [h("p", { class: "sub" }, five.sub || five.words)];
    if (five.bar) parts.push(outlookBar(five.bar, 14), legend(five.bar), h("p", { class: "def" }, five.definition));
    if (five.days.length) {
      const list = h("div", { class: "five", id: "five-days" },
        five.days.map((day) => h("div", { class: "r num" },
          h("span", { class: "day", "aria-hidden": "true" }, day.day),
          outlookBar(day.bar, 10, true),
          h("span", { class: "nums", "aria-hidden": "true" }, h("b", null, `↑${day.up}`), ` ↓${day.down}`),
          h("span", { class: "sr-only" }, day.label))),
        h("p", { class: "def" }, five.days_note));
      list.hidden = !state.eachDay;
      const toggle = h("button", { class: "more-btn", type: "button", "aria-expanded": state.eachDay, "aria-controls": "five-days", "data-key": "each-day" });
      const label = () => {
        toggle.replaceChildren(five.toggle[state.eachDay ? 1 : 0], h("span", { "aria-hidden": "true" }, state.eachDay ? " ▴" : " ▾"));
        toggle.setAttribute("aria-expanded", String(state.eachDay));
      };
      label();
      toggle.addEventListener("click", () => { state.eachDay = !state.eachDay; list.hidden = !state.eachDay; label(); });
      parts.push(toggle, list);
    }
    return card(five.title, "f-h", parts);
  }

  function reliableCard(r) {
    const d = r.development;
    const dev = d.state === "ok"
      ? [h("p", { class: "big" }, d.headline),
         h("p", { class: "meta num" }, d.meta, " ", sheetButton("pill", d.button, d.sheet, "read-this"))]
      : [d.flag ? h("span", { class: "flag" }, d.flag) : null, h("p", { class: "meta" }, d.words)];
    const parts = [h("div", { class: "rel-dev" }, dev)];
    if (r.live) {
      const live = r.live;
      parts.push(h("div", { class: "live" },
        live.flag ? h("span", { class: "flag" }, live.flag) : null,
        h("p", null, live.words),
        meter(live.meter),
        h("p", { class: "meta" }, live.scope, " ", sheetButton("pill", live.button, live.sheet, "judge"))));
    }
    return card(r.title, "r-h", h("div", { class: "rel" }, parts));
  }

  function latelyCard(l) {
    if (!l.chips.length) return card(l.title, "c-h", h("p", { class: "sub" }, l.words));
    return card(l.title, "c-h", h("p", { class: "sub" }, l.sub),
      h("div", { class: "ctx" }, l.chips.map((chip) => sheetButton("chip", chip.words, chip.sheet, `lately-${chip.key}`, h("span", { class: "info", "aria-hidden": "true" }, "ⓘ")))));
  }

  function learningCard(l) {
    const link = l.url && localUrl(l.url)
      ? h("a", { class: "link", href: l.url, target: "_blank", rel: "noopener noreferrer", "data-key": "learning" },
          h("span", null, h("span", { class: "t" }, l.words), h("span", { class: "s" }, l.sub),
            h("span", { class: "sr-only" }, " (opens in a new tab)")),
          h("span", { class: "chev", "aria-hidden": "true" }, "›"))
      : h("p", { class: "sub" }, l.elsewhere);
    return card(l.title, "m-h", link);
  }

  function backButton() {
    const button = h("button", { class: "back", type: "button", "data-key": "back" }, h("span", { "aria-hidden": "true" }, "‹ "), "Watchlist");
    button.addEventListener("click", goBack);
    return button;
  }

  function detail(data) {
    const px = h("p", { class: "d-px num" });
    const chg = h("p", { class: "d-chg num" });
    const readout = (point) => {
      px.textContent = point.close_words;
      chg.className = `d-chg num ${point.direction}`;
      chg.replaceChildren(h("span", { "aria-hidden": "true" }, arrow(point.direction)), point.change_words, " ",
        h("span", { class: "when" }, point.when));
    };
    let chart;
    if (data.chart.available) chart = chartBlock(data.chart, readout);
    else {
      px.textContent = data.price.close_words;
      chart = h("p", { class: "range-note" }, data.chart.words);
    }
    return [
      backButton(),
      h("div", { class: "d-head" }, h("h1", { class: "d-tk", tabindex: "-1", "data-key": "title" }, data.title), px, chg),
      chart,
      outlookCard(data.outlook),
      data.five ? fiveCard(data.five) : null,
      reliableCard(data.reliable),
      latelyCard(data.lately),
      learningCard(data.learning),
    ];
  }

  // --- History ----------------------------------------------------------------

  // A small button that shows one line's technical detail in a sheet, named by its stock and day.
  function detailButton(x, e) {
    const title = `${x.ticker}, ${e.title}`;
    const button = h("button", { class: "tech", type: "button", "aria-label": `Technical detail, ${title}`, "data-key": `tech-${e.as_of}-${x.ticker}` }, "ⓘ");
    button.addEventListener("click", () => openSheet({ title, paragraphs: [x.technical] }));
    return button;
  }

  // One stock's line in an evening: what it said (never coloured) and what happened (the
  // move's direction only). Screen readers get the server's whole sentence instead.
  function historyItem(x, e) {
    const tech = x.technical ? detailButton(x, e) : null;
    const said = x.said
      ? h("span", { class: "said num", "aria-hidden": "true" }, x.said.lead, " ",
          h("span", { class: "fc" }, `↑${x.said.up}`), " · ", h("span", { class: "fc" }, `↓${x.said.down}`))
      : h("span", { class: "said quiet" }, h("span", { "aria-hidden": "true" }, x.said_words), tech);
    let result = null;
    if (x.result) {
      result = h("span", { class: "res num", "aria-hidden": "true" },
        h("span", { class: `mv ${x.result.direction}` }, x.result.words), h("span", { class: "gave" }, x.result.gave));
    } else if (x.result_words) {
      result = h("span", { class: "res wait" }, h("span", { "aria-hidden": "true" }, x.result_words), x.said ? tech : null);
    }
    return h("li", { class: x.said ? "tl-item" : "tl-item quiet" },
      h("span", { class: "sr-only" }, x.label),
      h("span", { class: "tk", "aria-hidden": "true" }, x.ticker), said, result);
  }

  // An evening: its heading names the day the outlook was for; the line below it says when it
  // was recorded (or that it was missed).
  function evening(e, chosen) {
    const head = [h("h2", { class: "tl-h" }, e.title), h("p", { class: "tl-s" }, e.sub)];
    if (e.state === "missed") return h("li", { class: "tl-day missed" }, head, h("p", { class: "tl-miss" }, e.missed_words));
    const items = e.items.filter((x) => chosen === "All" || x.ticker === chosen);
    return h("li", { class: `tl-day ${e.state}` }, head,
      h("ul", { class: "tl-items", role: "list" }, items.map((x) => historyItem(x, e))));
  }

  function historyScreen(data) {
    const parts = [h("h1", { class: "screen" }, data.title), h("p", { class: "lede" }, data.lede)];
    if (data.problem) {
      parts.push(h("section", { class: "card empty-card bad" },
        h("h2", null, data.problem.title), h("p", null, data.problem.words), h("p", { class: "note" }, data.problem.technical)),
        h("p", { class: "asof" }, data.checked));
      return parts;
    }
    const asof = [data.counts, data.checked].filter(Boolean).join(" ");
    if (asof) parts.push(h("p", { class: "asof" }, asof));
    const chosen = data.filters.includes(state.hfilter) ? state.hfilter : "All";
    if (data.filters.length > 2) {  // "All" and two stocks or more
      parts.push(h("div", { class: "filters", role: "group", "aria-label": "Show stock" },
        data.filters.map((name) => {
          const button = h("button", { type: "button", "aria-pressed": name === chosen, "data-key": `filter-${name}` }, name);
          button.addEventListener("click", () => { state.hfilter = name; render(); });
          return button;
        })));
    }
    parts.push(h("p", { class: "hnote" }, h("span", { "aria-hidden": "true" }, "ℹ"),
      h("span", null, data.note.words, " ", sheetButton("textbtn", data.note.button, data.note.sheet, "judge"))));
    if (data.empty) parts.push(h("section", { class: "card empty-card" }, h("p", null, data.empty)));
    if (data.evenings.length) {
      parts.push(h("ol", { class: "tl", role: "list", "aria-label": "Evenings, newest first" }, data.evenings.map((e) => evening(e, chosen))));
    }
    if (data.more) parts.push(h("p", { class: "foot" }, data.more));
    return parts;
  }

  // --- Health -----------------------------------------------------------------

  const RING = {
    ok: "M5 12.5l4.5 4.5L19 7.5",
    warn: "M12 6.5v7M12 17.4v.1",
    bad: "M7 7l10 10M17 7 7 17",
    neutral: "M7 12h.1M12 12h.1M17 12h.1",
  };
  const MARK = { ok: "✓", note: "…", warn: "!", bad: "✕" };

  function check(c) {
    return h("div", { class: "check" },
      h("span", { class: `i ${c.mark}`, role: "img", "aria-label": c.state_word }, MARK[c.mark] || "!"),
      h("span", null, h("span", { class: "t" }, c.title),
        h("span", { class: "s" }, c.words, " ", h("code", null, c.check_id))));
  }

  function healthScreen(data) {
    const level = RING[data.level] ? data.level : "neutral";
    const icon = svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" });
    icon.append(svg("path", { d: RING[level] }));
    const toggle = h("button", { class: "disclose", type: "button", "aria-expanded": state.details, "aria-controls": "health-details", "data-key": "details" },
      data.toggle[state.details ? 1 : 0], h("span", { "aria-hidden": "true" }, state.details ? " ▴" : " ▾"));
    toggle.addEventListener("click", () => { state.details = !state.details; render(); });
    const link = data.link && localUrl(data.link.url)
      ? h("a", { class: "link", href: data.link.url, target: "_blank", rel: "noopener noreferrer", "data-key": "model-health" },
          h("span", null, h("span", { class: "t" }, data.link.words), h("span", { class: "s" }, data.link.sub),
            h("span", { class: "sr-only" }, " (opens in a new tab)")),
          h("span", { class: "chev", "aria-hidden": "true" }, "›"))
      : h("p", { class: "sub" }, data.elsewhere);
    const details = h("section", { class: "card", id: "health-details" },
      h("h2", null, data.updates.title),
      h("ul", { class: "updates" }, data.updates.lines.map((line) => h("li", null, line))),
      h("h2", { class: "next" }, data.checks.title),
      h("p", { class: "sub" }, data.checks.note),
      data.checks.technical ? h("p", { class: "note" }, data.checks.technical) : null,
      data.checks.items.length ? h("div", { class: "checks" }, data.checks.items.map(check)) : null,
      link);
    details.hidden = !state.details;
    return [
      h("div", { class: "hero" },
        h("div", { class: `ring ${level}` }, icon),
        h("h1", null, data.title),
        h("p", null, data.line),
        data.aside ? h("p", { class: "aside" }, data.aside) : null,
        data.flagged ? h("p", { class: "flagged" }, h("span", { class: "flag-mark", "aria-hidden": "true" }, "!"), data.flagged) : null,
        h("p", { class: "asof" }, data.checked),
        toggle),
      details,
    ];
  }

  // --- More -------------------------------------------------------------------

  function localUrl(text) {
    try {
      const url = new URL(text);
      return url.protocol === "http:" && ["127.0.0.1", "localhost"].includes(url.hostname) && !url.username;
    } catch (error) {
      return false;
    }
  }

  function more(data) {
    const theme = document.documentElement.dataset.theme || "dark";
    const choice = (value, label) => {
      const button = h("button", { type: "button", "data-theme-choice": value, "aria-pressed": theme === value, "data-key": `theme-${value}` }, label);
      button.addEventListener("click", () => setTheme(value));
      return button;
    };
    const research = data && data.research_url && localUrl(data.research_url)
      ? h("a", { class: "link", href: data.research_url, target: "_blank", rel: "noopener noreferrer", "data-key": "research" },
          "Open the research pages", h("span", { class: "sr-only" }, " (opens in a new tab)"),
          h("span", { class: "chev", "aria-hidden": "true" }, "›"))
      : h("p", { class: "sub" }, "They open from the Portfolio Lab app in your Dock.");
    return [
      h("h1", { class: "screen" }, "More"),
      h("section", { class: "card" }, h("h2", null, "Appearance"),
        h("div", { class: "seg2", role: "group", "aria-label": "Appearance" }, choice("dark", "Dark"), choice("light", "Light"))),
      h("section", { class: "card" }, h("h2", null, "Strategies"),
        h("p", { class: "sub" }, "How simple rebalancing rules would have done on real prices, 2003 to 2024. A look back, not advice."),
        openStrategies()),
      h("section", { class: "card" }, h("h2", null, "Research pages"),
        h("p", { class: "sub" }, "Today's detailed pages, kept for learning. They use more technical words."),
        research),
      h("section", { class: "card" }, h("h2", null, "About Portfolio Lab"),
        h("ul", { class: "about" },
          h("li", null, h("b", null, "Private. "), "It runs only on this Mac and sends nothing anywhere."),
          h("li", null, h("b", null, "Not a brokerage. "), "Nothing can be bought or sold here."),
          h("li", null, h("b", null, "Not advice. "), "The outlooks are an experiment in measuring chances honestly."),
          h("li", null, h("b", null, "Not yet proven. "), "Its live record is still too short to judge."))),
    ];
  }

  // --- More > Strategies ---------------------------------------------------------------

  function openStrategies() {
    const button = h("button", { class: "link", type: "button", "data-key": "strategies" },
      h("span", null, h("span", { class: "t" }, "Open Strategies"), h("span", { class: "s" }, "Read-only, from your private report")),
      h("span", { class: "chev", "aria-hidden": "true" }, "›"));
    button.addEventListener("click", () => {
      state.tab = "strategies";
      render();
      window.scrollTo(0, 0);
      document.getElementById("screen").focus({ preventScroll: true });
      loadTab("strategies");
    });
    return button;
  }

  function toMore() {
    const button = h("button", { class: "back", type: "button", "data-key": "back" }, h("span", { "aria-hidden": "true" }, "‹ "), "More");
    button.addEventListener("click", () => {
      state.tab = "more";
      render();
      window.scrollTo(0, 0);
      keyed("strategies") && keyed("strategies").focus({ preventScroll: true });
    });
    return button;
  }

  // A legend button hides or shows one line; the last visible line can't be hidden, so
  // the chart is never empty. Presentation only: the table and every number stay as served.
  function toggled(hidden, label, labels) {
    const next = hidden.includes(label) ? hidden.filter((l) => l !== label) : [...hidden, label];
    return labels.some((l) => !next.includes(l)) ? next : hidden;
  }

  // Every rule's growth of $10,000 on one log scale (drawing only: positions are never
  // shown), on a scale fixed by every line, so a toggle never moves the others. The rule
  // is the colour, the rebalancing cycle the dash. Hover, drag or the arrow keys read
  // every visible line's value, as the server worded it, for that week.
  function growthChart(chart, key) {
    const n = chart.days.length;
    const labels = chart.series.map((s) => s.label);
    const hidden = (state.hiddenLines[key] || []).filter((l) => labels.includes(l));
    const visible = chart.series.filter((s) => !hidden.includes(s.label));
    const box = h("div", { class: "chart growth", tabindex: "0", role: "slider", "aria-label": chart.label,
      "aria-valuemin": 0, "aria-valuemax": n - 1, "data-key": `growth-${key}` });
    const readout = h("div", { class: "g-readout", role: "group", "aria-label": "Values at the chosen week" });
    const H = 220, top = 10, bottom = 22, pad = 6;
    let geometry = null;
    let current = Number.isInteger(state.readAt[key]) && state.readAt[key] < n ? state.readAt[key] : n - 1;
    const valued = chart.series.flatMap((s) => s.values.filter((v) => typeof v === "number" && v > 0));
    const lo = Math.log(Math.min(...valued)), hi = Math.log(Math.max(...valued));

    function show(i) {
      current = i;
      state.readAt[key] = i;
      readout.replaceChildren(h("b", null, chart.days[i]),
        ...visible.map((s) => h("span", { class: "g-row" },
          h("i", { class: `g-key rule-${s.rule} every-${s.every}`, "aria-hidden": "true" }), s.label, h("b", { class: "num" }, s.words[i]))));
      box.setAttribute("aria-valuenow", i);
      box.setAttribute("aria-valuetext", `${chart.days[i]}: ${visible.map((s) => `${s.label} ${s.words[i]}`).join("; ")}`);
      if (!geometry) return;
      geometry.cursor.setAttribute("x1", geometry.x(i));
      geometry.cursor.setAttribute("x2", geometry.x(i));
    }

    function draw(width) {
      if (width < 2 || !valued.length) return;
      const x = (i) => pad + (i / Math.max(n - 1, 1)) * (width - 2 * pad);
      const y = (v) => top + (hi === lo ? 0.5 : 1 - (Math.log(v) - lo) / (hi - lo)) * (H - top - bottom);
      const art = svg("svg", { viewBox: `0 0 ${width} ${H}`, width, height: H, "aria-hidden": "true" });
      for (const tick of chart.ticks || []) {
        if (!(typeof tick.value === "number" && tick.value > 0)) continue;
        const label = svg("text", { class: "chart-tag", x: pad, y: y(tick.value) - 4 });
        label.textContent = tick.words;
        art.append(svg("line", { class: "g-grid", x1: pad, x2: width - pad, y1: y(tick.value), y2: y(tick.value) }), label);
      }
      let year = null;
      chart.days.forEach((day, i) => {
        const shown = day.slice(-4);
        if (shown === year) return;
        year = shown;
        if (Number(shown) % 4 !== 0) return;  // a label every four years keeps a phone's width clear
        const tick = svg("text", { class: "chart-tag", x: x(i), y: H - 4, "text-anchor": "middle" });
        tick.textContent = shown;
        art.append(svg("line", { class: "g-grid", x1: x(i), x2: x(i), y1: top, y2: H - bottom }), tick);
      });
      for (const s of visible) {
        let d = "", pen = false;
        s.values.forEach((v, i) => {
          if (typeof v !== "number" || v <= 0) { pen = false; return; }
          d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
          pen = true;
        });
        art.append(svg("path", { class: `g-line rule-${s.rule} every-${s.every}`, d }));
      }
      const cursor = svg("line", { class: "chart-cursor", y1: top, y2: H - bottom });
      art.append(cursor);
      box.replaceChildren(art);
      geometry = { x, cursor };
      show(current);
    }

    const at = (event) => {
      const rect = box.getBoundingClientRect();
      const share = (event.clientX - rect.left - pad) / Math.max(rect.width - 2 * pad, 1);
      return Math.max(0, Math.min(n - 1, Math.round(share * (n - 1))));
    };
    box.addEventListener("pointermove", (event) => show(at(event)));  // hover, and a drag
    box.addEventListener("pointerdown", (event) => show(at(event)));
    box.addEventListener("keydown", (event) => {
      const moves = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 };
      let next = null;
      if (event.key in moves) next = current + moves[event.key] * (event.shiftKey ? 52 : 1);
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = n - 1;
      if (next === null) return;
      event.preventDefault();
      show(Math.max(0, Math.min(n - 1, next)));
    });
    show(current);
    const legend = h("div", { class: "g-legend", role: "group", "aria-label": "Show or hide lines" },
      chart.series.map((s, i) => {
        const shown = !hidden.includes(s.label);
        const button = h("button", { type: "button", class: "g-toggle", "aria-pressed": shown, "data-key": `line-${key}-${i}` },
          h("i", { class: `g-key rule-${s.rule} every-${s.every}`, "aria-hidden": "true" }), s.label);
        button.addEventListener("click", () => {
          state.hiddenLines[key] = toggled(hidden, s.label, labels);
          render();
        });
        return button;
      }));
    let reset = null;
    if (hidden.length) {
      reset = h("button", { type: "button", class: "textbtn g-all", "data-key": `lines-all-${key}` }, "Show all lines");
      reset.addEventListener("click", () => {
        state.hiddenLines[key] = [];
        render();
        keyed(`line-${key}-0`) && keyed(`line-${key}-0`).focus({ preventScroll: true });
      });
    }
    return h("div", { class: "chart-block" }, legend, reset, fitted(box, draw), readout, h("p", { class: "note" }, chart.note));
  }

  function resultsTable(u) {
    return h("div", { class: "g-table", role: "region", "aria-label": `${u.title}: results`, tabindex: "0" },
      h("table", null,
        h("thead", null, h("tr", null, u.columns.map((c, i) => h("th", { scope: "col", class: i ? "num" : null }, c)))),
        h("tbody", null, u.rows.map((r) => h("tr", null,
          h("th", { scope: "row" }, h("i", { class: `g-key rule-${r.rule} every-${r.every}`, "aria-hidden": "true" }), r.label,
            r.stopped ? h("span", { class: "g-stop" }, r.stopped) : null),
          r.cells.map((c) => h("td", { class: "num" }, c)))))));
  }

  function strategiesScreen(data) {
    const parts = [toMore(), h("h1", { class: "screen" }, data.title), h("p", { class: "lede" }, data.lede)];
    if (data.problem) {
      parts.push(h("section", { class: "card empty-card" },
        h("h2", null, data.problem.title), h("p", null, data.problem.words),
        data.problem.technical ? h("p", { class: "note" }, data.problem.technical) : null));
      return parts;
    }
    parts.push(h("section", { class: "card g-labels", "aria-labelledby": "g-read" },
      h("h2", { id: "g-read" }, "Read this first"),
      h("ul", { class: "about" }, data.labels.map((label) => h("li", null, label)))));
    for (const u of data.universes) {
      const id = `g-${u.key}`;
      parts.push(h("section", { class: "card", "aria-labelledby": id },
        h("h2", { id }, u.title),
        h("p", { class: "sub" }, u.members),
        h("p", { class: "sub" }, u.span),
        growthChart(u.chart, u.key),
        resultsTable(u)));
    }
    parts.push(h("p", { class: "note" }, data.days_note), h("p", { class: "asof" }, data.source));
    return parts;
  }

  // --- render, load, route, theme -------------------------------------------------

  function problem(title, text, back, final) {
    const retry = final ? null : h("button", { type: "button", "data-key": "retry" }, "Try again");
    if (retry) retry.addEventListener("click", load);
    return [back ? backButton() : null,
      h("div", { class: "problem", role: "alert" }, h("b", null, title), text, retry ? [h("br"), retry] : null)];
  }

  function tabScreen(name, draw) {
    const said = state.problems[name];
    if (said) return problem(...said);
    if (!state[name]) return [h("p", { class: "loading" }, "Loading…")];
    return draw(state[name]);
  }

  function screen() {
    if (state.tab === "more") return more(state.home);
    if (state.tab === "history") return tabScreen("history", historyScreen);
    if (state.tab === "health") return tabScreen("health", healthScreen);
    if (state.tab === "strategies") return tabScreen("strategies", strategiesScreen);
    if (state.stock) {
      if (state.detailProblem) {
        const [title, text, final, noBack] = state.detailProblem;
        return problem(title, text, !noBack, final);
      }
      if (!state.detail) return [backButton(), h("p", { class: "loading" }, "Loading…")];
      return detail(state.detail);
    }
    if (state.problem) return problem(...state.problem);
    if (!state.home) return [h("p", { class: "loading" }, "Loading…")];
    return home(state.home);
  }

  // Controls carry a data-key, so a re-render puts focus back on the same control.
  const keyOf = (el) => (el && el.dataset && el.dataset.key) || null;
  const keyed = (key) => (key ? document.querySelector(`#screen [data-key="${CSS.escape(key)}"]`) : null);

  function render() {
    if (sheetOpen()) { state.dirty = true; return; }  // never pull a sheet's button away
    const main = document.getElementById("screen");
    const focused = main.contains(document.activeElement) ? keyOf(document.activeElement) : null;
    for (const box of drawers.keys()) resizer.unobserve(box);
    drawers.clear();
    let parts;
    try {
      parts = screen();
    } catch (error) {
      parts = problem("The Lab couldn't show this screen.", "Nothing has changed in your records.", Boolean(state.stock));
    }
    main.replaceChildren(...parts.filter(Boolean));
    const again = keyed(focused);
    if (again) again.focus({ preventScroll: true });
    const named = { history: "History", health: "Health", more: "More", strategies: "Strategies" }[state.tab];
    document.title = state.stock && state.detail ? `${state.detail.title} · Portfolio Lab`
      : named ? `${named} · Portfolio Lab` : "Portfolio Lab";
    const tabbed = state.tab === "strategies" ? "more" : state.tab;  // Strategies opens from More
    for (const tab of document.querySelectorAll(".tabs [data-tab]")) {
      if (tab.dataset.tab === tabbed) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    // The Health tab's badge, as the server sends it: amber or red with Home's status line, or
    // "!" when Model Health flags a check. A mark as well as a colour, and nothing otherwise.
    const shown = badgeNow();
    const badge = document.getElementById("health-badge");
    badge.hidden = !shown;
    badge.className = shown && (shown.level === "warn" || shown.level === "bad") ? `badge ${shown.level}` : "badge";
    badge.textContent = shown ? shown.mark : "";
    if (shown) badge.setAttribute("aria-label", shown.words);
    else badge.removeAttribute("aria-label");
    const demo = (state.home && state.home.demo) || (state.detail && state.detail.demo);
    const ribbon = document.getElementById("ribbon");
    ribbon.textContent = demo || "";
    ribbon.hidden = !demo;
    if (demo) document.documentElement.dataset.demo = "";
    else delete document.documentElement.dataset.demo;
  }

  function usable(data) {
    return data && data.schema === "lab_app_home" && data.version === 1 && Array.isArray(data.stocks)
      && data.status && data.explain;
  }

  const USABLE = {
    history: (d) => d && d.schema === "lab_app_history" && d.version === 1 && Array.isArray(d.evenings)
      && Array.isArray(d.filters) && d.note && typeof d.title === "string",
    health: (d) => d && d.schema === "lab_app_health" && d.version === 1 && typeof d.level === "string"
      && d.updates && Array.isArray(d.updates.lines) && d.checks && Array.isArray(d.checks.items)
      && Array.isArray(d.toggle),
    strategies: (d) => d && d.schema === "lab_app_strategies" && d.version === 1 && typeof d.title === "string"
      && Array.isArray(d.labels) && Array.isArray(d.universes)
      && d.universes.every((u) => u.chart && Array.isArray(u.chart.series) && Array.isArray(u.chart.days)
        && Array.isArray(u.rows) && Array.isArray(u.columns))
      && (d.problem === null || typeof d.problem.title === "string"),
  };

  // The newest answer's badge: Home's, or Health's when Health answered later (both come from
  // the same rule), so the tab never lags the Health screen it marks.
  function badgeNow() {
    const home = state.home, health = state.health;
    if (health && health.badge !== undefined
        && (!home || Date.parse(health.generated_at) > Date.parse(home.generated_at))) return health.badge;
    return home ? home.badge : null;
  }

  function usableDetail(data, ticker) {
    return data && data.schema === "lab_app_stock" && data.version === 1 && data.ticker === ticker
      && data.price && data.chart && data.outlook && data.reliable && data.lately && data.learning;
  }

  // This launch's key arrives in the address's fragment (#key=...), which browsers never
  // send to a server. It is kept for this tab only and taken out of the address at once.
  function takeKey() {
    const found = KEY.exec(location.hash);
    if (found) {
      state.key = found[1];
      try { sessionStorage.setItem(KEY_STORE, state.key); } catch (error) { /* storage may be off */ }
      history.replaceState(null, "", location.pathname + location.search);
      return;
    }
    let kept = null;
    try { kept = sessionStorage.getItem(KEY_STORE); } catch (error) { kept = null; }
    state.key = kept && KEY.test(`#key=${kept}`) ? kept : null;  // only a well-formed key is sent
  }

  async function ask(path) {
    let response;
    try {
      const headers = state.key ? { "X-Lab-Token": state.key } : {};
      response = await fetch(path, { cache: "no-store", credentials: "omit", headers });
    } catch (error) {
      return { gone: true };
    }
    let data = null;
    try { data = await response.json(); } catch (error) { data = null; }
    return { response, data };
  }

  const SILENT = ["The Lab isn't answering.", "If you closed Portfolio Lab, open it again. Nothing has changed in your records."];
  // A page opened without this launch's key (a bookmark, a copied address): the way back in.
  const LOCKED = ["Open Portfolio Lab from its app.", "For your privacy, this page shows your numbers only when Portfolio Lab opens it. Click its icon in the Dock.", false, true];

  async function loadHome() {
    const asked = ++state.asked;  // an older answer arriving late is ignored
    const { gone, response, data } = await ask("/api/home");
    if (asked !== state.asked) return;
    if (!gone && response.status === 401 && state.problem === LOCKED) return;  // said once is enough
    if (gone) state.problem = SILENT;
    else if (response.status === 401) state.problem = LOCKED;
    else if (!response.ok) {
      const said = data && typeof data.error === "string" ? data.error : "The Lab had a problem.";
      state.problem = [said, "Nothing has changed in your records."];
    } else if (!usable(data)) {
      state.problem = ["The Lab sent something this page doesn't understand.", "Reload the page after updating Portfolio Lab."];
    } else {
      state.home = data;
      state.problem = null;
    }
    if (!state.stock || state.tab !== "home") render();  // a stock's screen keeps its place
  }

  async function loadDetail() {
    const ticker = state.stock;
    if (!ticker) return;
    const asked = ++state.askedDetail;
    const arriving = !state.detail;
    const { gone, response, data } = await ask(`/api/stock/${encodeURIComponent(ticker)}`);
    if (asked !== state.askedDetail || ticker !== state.stock) return;
    if (response && response.ok && state.detail && !state.detailProblem && usableDetail(data, ticker) && same(state.detail, data)) return;
    if (!gone && response.status === 401 && state.detailProblem && state.detailProblem[0] === LOCKED[0]) return;
    if (gone) state.detailProblem = SILENT;
    else if (response.status === 401) state.detailProblem = [LOCKED[0], LOCKED[1], true, true];  // no Back: Home is locked too
    else if (response.status === 404) state.detailProblem = ["This stock isn't on the watchlist.", "Go back to see the stocks the Lab follows.", true];
    else if (!response.ok) {
      const said = data && typeof data.error === "string" ? data.error : "The Lab had a problem.";
      state.detailProblem = [said, "Nothing has changed in your records."];
    } else if (!usableDetail(data, ticker)) {
      state.detailProblem = ["The Lab sent something this page doesn't understand.", "Reload the page after updating Portfolio Lab."];
    } else {
      state.detail = data;
      state.detailProblem = null;
    }
    render();
    const main = document.getElementById("screen");
    if (arriving && state.detail && [main, document.body].includes(document.activeElement)) {
      keyed("title").focus({ preventScroll: true });  // say which stock opened
    }
  }

  // History or Health: the screen's own data, kept in place while it is asked again.
  async function loadTab(name) {
    const asked = ++state.askedTab[name];  // an older answer arriving late is ignored
    const { gone, response, data } = await ask(`/api/${name}`);
    if (asked !== state.askedTab[name]) return;
    if (response && response.ok && state[name] && !state.problems[name] && USABLE[name](data) && same(state[name], data)) return;
    if (!gone && response.status === 401 && state.problems[name] === LOCKED) return;
    if (gone) state.problems[name] = SILENT;
    else if (response.status === 401) state.problems[name] = LOCKED;
    else if (!response.ok) {
      const said = data && typeof data.error === "string" ? data.error : "The Lab had a problem.";
      state.problems[name] = [said, "Nothing has changed in your records."];
    } else if (!USABLE[name](data)) {
      state.problems[name] = ["The Lab sent something this page doesn't understand.", "Reload the page after updating Portfolio Lab."];
    } else {
      state[name] = data;
      state.problems[name] = null;
    }
    if (state.tab === name) render();
  }

  // The same screen as before, apart from when it was prepared.
  const same = (a, b) => JSON.stringify({ ...a, generated_at: null }) === JSON.stringify({ ...b, generated_at: null });

  const TABS = ["history", "health", "strategies"];  // the screens with data of their own

  function load() {
    loadHome();
    if (state.stock) loadDetail();
    if (TABS.includes(state.tab)) loadTab(state.tab);
  }

  function routed() {
    const found = TICKER.exec(location.hash);
    return found ? found[1] : null;
  }

  // The address names the screen: "#stock/NVDA" is NVDA's detail, anything else Home.
  function follow() {
    const ticker = routed();
    if (ticker === state.stock) return;
    const previous = state.stock;
    state.stock = ticker;
    state.detail = null;
    state.detailProblem = null;
    if (sheetOpen()) closeSheet();
    if (ticker && previous) state.fromHome = false;  // Back would land on the other stock
    if (ticker) {
      state.tab = "home";
      state.eachDay = false;
      render();
      window.scrollTo(0, 0);
      document.getElementById("screen").focus({ preventScroll: true });
      loadDetail();
      return;
    }
    state.fromHome = false;
    render();
    window.scrollTo(0, state.homeScroll);
    (keyed(`row-${previous}`) || document.getElementById("screen")).focus({ preventScroll: true });
  }

  function leaveDetail() {
    if (!state.stock) return;
    if (state.fromHome) history.back();  // no second Home entry in the history
    else history.replaceState(null, "", location.pathname + location.search);
    state.stock = null;
    state.detail = null;
    state.detailProblem = null;
    state.fromHome = false;
  }

  function goBack() {
    if (state.fromHome) history.back();  // to the watchlist the row was opened from
    else location.hash = "";
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch (error) { /* storage may be off */ }
    document.getElementById("theme-btn").setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    for (const button of document.querySelectorAll("[data-theme-choice]")) {
      button.setAttribute("aria-pressed", String(button.dataset.themeChoice === theme));
    }
  }

  function start() {
    let theme = "dark";
    try { theme = localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark"; } catch (error) { /* storage may be off */ }
    setTheme(theme);
    document.getElementById("theme-btn").addEventListener("click", () =>
      setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
    for (const tab of document.querySelectorAll(".tabs [data-tab]")) {
      tab.addEventListener("click", () => {
        leaveDetail();
        state.tab = tab.dataset.tab;
        render();
        window.scrollTo(0, 0);
        document.getElementById("screen").focus({ preventScroll: true });
        if (TABS.includes(state.tab)) loadTab(state.tab);
      });
    }
    document.getElementById("scrim").addEventListener("click", closeSheet);
    document.getElementById("sheet").addEventListener("click", (event) => {
      if (event.target.closest("[data-close]")) closeSheet();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && sheetOpen()) closeSheet();
    });
    // A key that arrives while the app is open (a reused keyed address) replaces the
    // address, so the screen follows it to Home, in step with the address and Back.
    window.addEventListener("hashchange", () => {
      if (KEY.test(location.hash)) { takeKey(); follow(); load(); } else follow();
    });
    // Fresh words when the window comes back, never on a timer. Both events fire together.
    const refresh = () => {
      if (document.hidden || Date.now() - state.refreshed < 1500) return;
      state.refreshed = Date.now();
      load();
    };
    document.addEventListener("visibilitychange", refresh);
    window.addEventListener("focus", refresh);
    takeKey();  // before routing: the fragment may hold the key, not a stock
    state.stock = routed();
    load();
  }

  document.addEventListener("DOMContentLoaded", start);
})();
