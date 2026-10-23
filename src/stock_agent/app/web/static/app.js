// The Lab app's page: a thin renderer of /api/home. Every number, and every
// sentence about the data (status, outlooks, explanations), comes from the
// server already worded; only the page's own fixed labels (the tabs, More,
// About and the error messages) are written here. It builds elements with
// textContent (never innerHTML) and sets no inline style, so the page's strict
// content security policy holds.
"use strict";

(() => {
  const SVG = "http://www.w3.org/2000/svg";
  const THEME_KEY = "lab-theme";
  const state = { tab: "home", home: null, problem: null, lastFocus: null, dirty: false, asked: 0 };

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
    const back = state.lastFocus && document.contains(state.lastFocus) ? state.lastFocus : document.getElementById("screen");
    back.focus({ preventScroll: true });
  }

  const sheetOpen = () => !document.getElementById("sheet").hidden;

  // --- Home -------------------------------------------------------------------

  function statusLine(status) {
    const button = h("button", { class: `status ${status.level}`, type: "button" },
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
      chip = h("button", { class: "chip", type: "button", "aria-label": label },
        h("span", { class: "lbl" }, o.day),
        h("span", { class: "fc num" }, `↑ ${o.chip.up}`),
        h("span", { class: "lbl" }, "·"),
        h("span", { class: "fc num" }, `↓ ${o.chip.down}`),
        h("span", { class: "info" }, "ⓘ"));
    } else {
      chip = h("button", { class: `chip empty ${o.level}`, type: "button", "aria-label": label },
        o.words, h("span", { class: "info" }, "ⓘ"));
    }
    chip.addEventListener("click", () => openSheet(o.sheet));
    return chip;
  }

  function stockRow(stock) {
    const p = stock.price;
    const arrow = p.direction === "up" ? "▲ " : p.direction === "down" ? "▼ " : "";
    if (!p.available) {
      return h("li", { class: "row" },
        h("span", { class: "tk" }, stock.ticker),
        h("span", { class: "px none" }, p.words),
        h("span", { class: "nm" }, stock.name || ""),
        h("span", { class: "look" }, outlookChip(stock)));
    }
    return h("li", { class: "row" },
      h("span", { class: "tk" }, stock.ticker),
      spark(p.line, p.direction),
      h("span", { class: "px num" }, p.close_words),
      h("span", { class: "nm" }, stock.name || ""),
      h("span", { class: `chg num ${p.direction}` }, h("span", { "aria-hidden": "true" }, arrow), p.change_words),
      p.older ? h("span", { class: "when" }, `on ${p.session_words}`) : null,
      h("span", { class: "look" }, outlookChip(stock)));
  }

  function home(data) {
    const explain = h("button", { class: "textbtn", type: "button" }, data.explain.title);
    explain.addEventListener("click", () => openSheet(data.explain));
    const notes = [data.prices_note, data.lines_note, data.checked].filter(Boolean).join(" ");
    return [
      h("h1", { class: "sr-only" }, "Home"),
      statusLine(data.status),
      notes ? h("p", { class: "asof" }, notes) : null,
      h("h2", { class: "eyebrow" }, "Watchlist"),
      h("ul", { class: "list" }, data.stocks.map(stockRow)),
      h("p", { class: "foot" }, data.footer, " ", explain),
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
      const button = h("button", { type: "button", "data-theme-choice": value, "aria-pressed": theme === value }, label);
      button.addEventListener("click", () => setTheme(value));
      return button;
    };
    const research = data && data.research_url && localUrl(data.research_url)
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
          h("li", null, h("b", null, "Not yet proven. "), "Its live record is still too short to judge."))),
    ];
  }

  // --- render, load, theme ------------------------------------------------------

  function problem(title, text) {
    const retry = h("button", { type: "button" }, "Try again");
    retry.addEventListener("click", load);
    return [h("div", { class: "problem", role: "alert" }, h("b", null, title), text, h("br"), retry)];
  }

  function render() {
    if (sheetOpen()) { state.dirty = true; return; }  // never pull a sheet's chip away
    const main = document.getElementById("screen");
    let parts;
    try {
      if (state.tab === "more") parts = more(state.home);
      else if (state.problem) parts = problem(...state.problem);
      else if (!state.home) parts = [h("p", { class: "loading" }, "Loading…")];
      else parts = home(state.home);
    } catch (error) {
      parts = problem("The Lab couldn't show this screen.", "Nothing has changed in your records.");
    }
    main.replaceChildren(...parts.filter(Boolean));
    for (const tab of document.querySelectorAll(".tabs [data-tab]")) {
      if (tab.dataset.tab === state.tab) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    const demo = state.home && state.home.demo;
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

  async function load() {
    const ask = ++state.asked;  // an older answer arriving late is ignored
    let response;
    try {
      response = await fetch("/api/home", { cache: "no-store", credentials: "omit" });
    } catch (error) {
      if (ask !== state.asked) return;
      state.problem = ["The Lab isn't answering.", "If you closed Portfolio Lab, open it again. Nothing has changed in your records."];
      render();
      return;
    }
    let data = null;
    try { data = await response.json(); } catch (error) { data = null; }
    if (ask !== state.asked) return;
    if (!response.ok) {
      const said = data && typeof data.error === "string" ? data.error : "The Lab had a problem.";
      state.problem = [said, "Nothing has changed in your records."];
    } else if (!usable(data)) {
      state.problem = ["The Lab sent something this page doesn't understand.", "Reload the page after updating Portfolio Lab."];
    } else {
      state.home = data;
      state.problem = null;
    }
    render();
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
        state.tab = tab.dataset.tab;
        render();
        window.scrollTo(0, 0);
        document.getElementById("screen").focus({ preventScroll: true });
      });
    }
    document.getElementById("scrim").addEventListener("click", closeSheet);
    document.getElementById("sheet").addEventListener("click", (event) => {
      if (event.target.closest("[data-close]")) closeSheet();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && sheetOpen()) closeSheet();
    });
    // Fresh words when the window comes back, never on a timer.
    document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
    window.addEventListener("focus", load);
    load();
  }

  document.addEventListener("DOMContentLoaded", start);
})();
