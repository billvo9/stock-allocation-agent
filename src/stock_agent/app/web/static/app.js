// The Lab app's page: a thin renderer of /api/home. Every word and number on
// the screen comes from the server, already formatted; this script only lays
// them out. It builds elements with textContent (never innerHTML) and sets no
// inline style, so the page's strict content security policy holds.
"use strict";

(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const THEME_KEY = "lab-theme";
  const state = { tab: "home", home: null, failed: false, lastFocus: null };

  // --- small builders ---------------------------------------------------------

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") el.className = value;
      else el.setAttribute(key, value === true ? "" : String(value));
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

  // The one-month line: the reader's levels, broken at a gap, with a dot in the
  // colour of the latest move. Drawing only: nothing is computed from prices.
  function spark(line, direction) {
    const w = 96, h2 = 38, pad = 5;
    const box = svg("svg", { class: "spark", viewBox: `0 0 ${w} ${h2}`, "aria-hidden": "true" });
    const values = line.filter((v) => typeof v === "number");
    if (values.length < 2) return box;
    const lo = Math.min(...values), hi = Math.max(...values);
    const x = (i) => pad + (i / (line.length - 1)) * (w - 2 * pad);
    const y = (v) => pad + (hi === lo ? (h2 - 2 * pad) / 2 : (1 - (v - lo) / (hi - lo)) * (h2 - 2 * pad));
    let d = "", pen = false, last = null;
    line.forEach((v, i) => {
      if (typeof v !== "number") { pen = false; return; }
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

  // --- the explanation sheet --------------------------------------------------

  function openSheet(sheet) {
    const box = document.getElementById("sheet");
    state.lastFocus = document.activeElement;
    const parts = [h("div", { class: "grab" }), h("h3", { id: "sheet-title" }, sheet.title)];
    if (sheet.rows && sheet.rows.length) {
      parts.push(
        h("table", { class: "num" },
          h("thead", null, h("tr", null, h("th", { scope: "col" }, "What could happen"), h("th", { scope: "col" }, "Chance"))),
          h("tbody", null, sheet.rows.map(([label, value]) => h("tr", null, h("td", null, label), h("td", null, value)))))
      );
    }
    for (const text of sheet.paragraphs || []) parts.push(h("p", null, text));
    const close = h("button", { class: "close", type: "button", "data-close": true }, "Got it");
    parts.push(close);
    box.replaceChildren(...parts);
    box.hidden = false;
    document.getElementById("scrim").hidden = false;
    close.focus();
  }

  function closeSheet() {
    document.getElementById("sheet").hidden = true;
    document.getElementById("scrim").hidden = true;
    if (state.lastFocus && document.contains(state.lastFocus)) state.lastFocus.focus();
  }

  // --- Home -------------------------------------------------------------------

  function statusLine(status) {
    const button = h("button", { class: `status ${status.level}`, type: "button", "aria-label": `${status.words} Show update details.` },
      h("span", { class: "dot", "aria-hidden": "true" }),
      h("span", null, status.words),
      h("span", { class: "chev", "aria-hidden": "true" }, "›"));
    button.addEventListener("click", () => openSheet({ title: "Updates", paragraphs: status.details }));
    return button;
  }

  function outlookChip(stock) {
    const o = stock.outlook;
    let chip;
    if (o.chip) {
      chip = h("button", { class: "chip", type: "button", "aria-label": `${stock.ticker}, ${o.day} outlook: up ${o.chip.up}, down ${o.chip.down}. Explain.` },
        h("span", { class: "lbl" }, o.day),
        h("span", { class: "fc num" }, `↑ ${o.chip.up}`),
        h("span", { class: "lbl", "aria-hidden": "true" }, "·"),
        h("span", { class: "fc num" }, `↓ ${o.chip.down}`),
        h("span", { class: "info", "aria-hidden": "true" }, "ⓘ"));
    } else {
      chip = h("button", { class: `chip empty ${o.level}`, type: "button", "aria-label": `${stock.ticker}: ${o.words}. Explain.` },
        o.words, h("span", { class: "info", "aria-hidden": "true" }, "ⓘ"));
    }
    chip.addEventListener("click", () => openSheet(o.sheet));
    return chip;
  }

  function stockRow(stock) {
    const p = stock.price;
    const arrow = p.direction === "up" ? "▲ " : p.direction === "down" ? "▼ " : "";
    const label = p.available
      ? `${stock.ticker}, ${p.close_words}, ${p.change_words} on ${p.session_words}.`
      : `${stock.ticker}. ${p.words}`;
    return h("li", { class: "row", "aria-label": label },
      h("span", { class: "tk" }, stock.ticker),
      spark(p.line, p.direction),
      p.available ? h("span", { class: "px num" }, p.close_words) : h("span", { class: "px none" }, p.words),
      h("span", { class: "nm" }, stock.name || ""),
      p.available ? h("span", { class: `chg num ${p.direction}` }, arrow + p.change_words) : h("span", null),
      p.older ? h("span", { class: "when" }, `on ${p.session_words}`) : null,
      h("span", { class: "look" }, outlookChip(stock)));
  }

  function home(data) {
    const explain = h("button", { class: "textbtn", type: "button" }, data.explain.title);
    explain.addEventListener("click", () => openSheet(data.explain));
    return [
      statusLine(data.status),
      data.prices_note ? h("p", { class: "asof" }, `${data.prices_note} Lines show the past month; the dot is that day's move.`) : null,
      h("h2", { class: "eyebrow" }, "Watchlist"),
      h("ul", { class: "list" }, data.stocks.map(stockRow)),
      h("p", { class: "foot" }, data.footer, " ", explain),
    ];
  }

  // --- More -------------------------------------------------------------------

  function more(data) {
    const theme = document.documentElement.dataset.theme || "dark";
    const choice = (value, label) => {
      const button = h("button", { type: "button", "aria-pressed": theme === value }, label);
      button.addEventListener("click", () => setTheme(value));
      return button;
    };
    const research = data && data.research_url
      ? h("a", { class: "link", href: data.research_url, target: "_blank", rel: "noopener noreferrer" },
          "Open the research pages", h("span", { class: "chev", "aria-hidden": "true" }, "›"))
      : h("p", { class: "sub" }, "They open from the Portfolio Lab app in your Dock.");
    return [
      h("h1", { class: "screen" }, "More"),
      h("section", { class: "card" }, h("h2", null, "Appearance"),
        h("div", { class: "seg2", role: "group", "aria-label": "Appearance" }, choice("dark", "Dark"), choice("light", "Light"))),
      h("section", { class: "card" }, h("h2", null, "Research pages"),
        h("p", { class: "sub" }, "Today's detailed pages, kept for learning. They use more technical words."),
        research),
      h("section", { class: "card" }, h("h2", null, "About Portfolio Lab"),
        h("ul", { class: "about" },
          h("li", null, h("b", null, "Private. "), "It runs only on this Mac and sends nothing anywhere."),
          h("li", null, h("b", null, "Not a brokerage. "), "Nothing can be bought or sold here."),
          h("li", null, h("b", null, "Not advice. "), "The outlooks are an experiment in measuring chances honestly."),
          h("li", null, h("b", null, "Not yet proven. "), "Its live record started on Oct 20, 2026."))),
    ];
  }

  // --- render, load, theme ------------------------------------------------------

  function render() {
    const main = document.getElementById("screen");
    let parts;
    if (state.tab === "more") parts = more(state.home);
    else if (state.failed) {
      const retry = h("button", { type: "button" }, "Try again");
      retry.addEventListener("click", load);
      parts = [h("div", { class: "problem", role: "alert" },
        h("b", null, "The Lab isn't answering."),
        "If you closed Portfolio Lab, open it again. Nothing has changed in your records.", h("br"), retry)];
    } else if (!state.home) parts = [h("p", { class: "loading" }, "Loading…")];
    else parts = home(state.home);
    main.replaceChildren(...parts.filter(Boolean));
    for (const tab of document.querySelectorAll(".tabs [data-tab]")) {
      if (tab.dataset.tab === state.tab) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    const ribbon = document.getElementById("ribbon");
    ribbon.textContent = state.home && state.home.demo ? state.home.demo : "";
    ribbon.hidden = !(state.home && state.home.demo);
  }

  async function load() {
    try {
      const response = await fetch("/api/home", { cache: "no-store", credentials: "omit" });
      if (!response.ok) throw new Error(String(response.status));
      state.home = await response.json();
      state.failed = false;
    } catch (error) {
      state.failed = true;
    }
    render();
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch (error) { /* storage may be off */ }
    document.getElementById("theme-btn").setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    render();
  }

  function start() {
    let theme = "dark";
    try { theme = localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark"; } catch (error) { /* storage may be off */ }
    document.documentElement.dataset.theme = theme;
    document.getElementById("theme-btn").setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    document.getElementById("theme-btn").addEventListener("click", () =>
      setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
    for (const tab of document.querySelectorAll(".tabs [data-tab]")) {
      tab.addEventListener("click", () => { state.tab = tab.dataset.tab; render(); window.scrollTo(0, 0); });
    }
    document.getElementById("scrim").addEventListener("click", closeSheet);
    document.getElementById("sheet").addEventListener("click", (event) => {
      if (event.target.closest("[data-close]")) closeSheet();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !document.getElementById("sheet").hidden) closeSheet();
    });
    // Fresh numbers when the window comes back, never on a timer.
    document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
    load();
  }

  document.addEventListener("DOMContentLoaded", start);
})();
