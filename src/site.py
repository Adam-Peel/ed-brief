"""Render the published site: docs/index.html (the rolling list), the dated
archive index, and each run's dated snapshot page under docs/briefs/.

Everything here is static output of already-ranked data -- the page never
scores, sorts, or fetches anything at view time. Client-side JS only
*filters* the already-sorted item list and toggles per-browser read state.

Unlike the proof of concept's render.py, this module builds pages with plain
`str.replace()` against unique tokens rather than `.format()` against a
double-braced template. There's a lot more inline JS in this version, and
every literal `{`/`}` in it would otherwise need doubling -- `.replace()`
sidesteps that whole class of mistake.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from .corpus import CorpusItem

TIER_LABELS = {
    "lead": "Read these",
    "worth": "Worth a look",
    "rest": "Everything else",
    "noise": "Low relevance",
}
TIER_ORDER = ["lead", "worth", "rest", "noise"]

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #fbfaf8;
  --surface: #ffffff;
  --border: #e4e0d8;
  --text: #23201c;
  --muted: #6c665d;
  --accent: #7a4b2a;
  --accent-soft: #f2e8e0;
  --lead: #8a3d1f;
  --worth: #4a6741;
  --rest: #6c665d;
  --shadow: 0 1px 2px rgba(35, 32, 28, .05), 0 4px 14px rgba(35, 32, 28, .04);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161513;
    --surface: #1e1d1a;
    --border: #33302b;
    --text: #ece8e1;
    --muted: #9b948a;
    --accent: #d09a6f;
    --accent-soft: #2b2520;
    --lead: #e0a077;
    --worth: #9dbb8f;
    --rest: #9b948a;
    --shadow: none;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 16px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 780px; margin: 0 auto; padding: 32px 20px 80px; }
header { margin-bottom: 28px; }
h1 {
  font-size: clamp(1.5rem, 4vw, 2rem);
  line-height: 1.2;
  margin: 0 0 6px;
  letter-spacing: -.02em;
}
.sub { color: var(--muted); font-size: .9rem; margin: 0; }
.sub a { color: var(--muted); }

.filters {
  position: sticky; top: 0; z-index: 5;
  background: var(--bg);
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.row + .row { margin-top: 8px; }
.chip {
  font: inherit; font-size: .78rem;
  padding: 4px 11px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  border-radius: 999px;
  cursor: pointer;
  transition: .12s;
  white-space: nowrap;
}
.chip:hover { border-color: var(--accent); color: var(--text); }
.chip[aria-pressed="true"] {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent);
  font-weight: 600;
}
.linklike {
  font: inherit; font-size: .78rem;
  background: none; border: none;
  color: var(--muted);
  text-decoration: underline;
  cursor: pointer;
  padding: 4px 2px;
}
.linklike:hover { color: var(--accent); }
#search {
  flex: 1; min-width: 160px;
  font: inherit; font-size: .85rem;
  padding: 5px 11px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--text);
}
#search:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
.count { color: var(--muted); font-size: .78rem; margin-left: auto; }

h2 {
  font-size: .78rem; text-transform: uppercase; letter-spacing: .1em;
  color: var(--muted); font-weight: 600;
  margin: 32px 0 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
h2[data-tier="lead"] { color: var(--lead); }
h2[data-tier="worth"] { color: var(--worth); }
article {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--rest);
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 10px;
  box-shadow: var(--shadow);
}
article[data-tier="lead"] { border-left-color: var(--lead); }
article[data-tier="worth"] { border-left-color: var(--worth); }
article.is-read { opacity: .55; }
article h3 { margin: 0 0 5px; font-size: 1.02rem; line-height: 1.35; font-weight: 600; display: flex; gap: 8px; align-items: baseline; }
article h3 a { color: var(--text); text-decoration: none; }
article h3 a:hover { color: var(--accent); text-decoration: underline; }
.score {
  flex: none;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .76rem; font-weight: 700;
  color: var(--rest); border: 1px solid var(--border);
  border-radius: 5px; padding: 1px 6px;
  cursor: help;
}
.score[data-tier="lead"] { color: var(--lead); border-color: var(--lead); }
.score[data-tier="worth"] { color: var(--worth); border-color: var(--worth); }
.meta { font-size: .76rem; color: var(--muted); margin-bottom: 7px; }
.mode { text-transform: uppercase; letter-spacing: .04em; font-size: .68rem; }
.why {
  font-size: .86rem; color: var(--text);
  background: var(--accent-soft);
  padding: 6px 10px; border-radius: 5px; margin: 0 0 8px;
}
.summary { font-size: .88rem; color: var(--muted); margin: 0 0 9px; }
.row-bottom { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.tags { display: flex; flex-wrap: wrap; gap: 4px; }
.tag {
  font-size: .68rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--muted); background: var(--bg);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 1px 6px;
}
.read-btn {
  font: inherit; font-size: .72rem;
  padding: 3px 9px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  border-radius: 999px;
  cursor: pointer;
  white-space: nowrap;
}
.read-btn:hover { border-color: var(--accent); color: var(--accent); }
.empty { color: var(--muted); font-style: italic; padding: 32px 0; text-align: center; }
footer {
  margin-top: 48px; padding-top: 16px;
  border-top: 1px solid var(--border);
  font-size: .78rem; color: var(--muted);
}
footer a { color: var(--muted); }
footer ul { padding-left: 18px; margin: 6px 0; }
details summary { cursor: pointer; }
@media (max-width: 560px) {
  .wrap { padding: 20px 14px 60px; }
  .count { width: 100%; margin-left: 0; }
}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>__HEADING__</h1>
  <p class="sub">__SUBTITLE__</p>
</header>

<div class="filters">
  <div class="row" id="tier-row">
    <button class="chip" data-tier="lead" aria-pressed="true">Read these</button>
    <button class="chip" data-tier="worth" aria-pressed="true">Worth a look</button>
    <button class="chip" data-tier="rest" aria-pressed="false">Everything else</button>
    <button class="chip" data-tier="noise" aria-pressed="false">Low relevance</button>
    <button class="chip" id="read-toggle" aria-pressed="false">Show read</button>
    <input id="search" type="search" placeholder="Search titles and summaries…" aria-label="Search">
  </div>
  <div class="row" id="tag-row">__TAG_CHIPS__</div>
  <div class="row" id="source-row">__SOURCE_CHIPS__<button class="linklike" id="mark-all">Mark all read</button><span class="count" id="count"></span></div>
</div>

<main id="list"></main>
<p class="empty" id="empty" hidden>Nothing matches these filters.</p>

<footer>
  <p>Ranking: __SCORING_LINE__. Generated __GENERATED__. Items stay listed for
  __RETENTION_DAYS__ days from when they're first seen, or until you mark
  them read.</p>
  <p>Read state is stored in this browser only. It will not follow you to
  another device or browser.</p>
  __PROBLEMS_HTML__
  __FOOTER_EXTRA__
  <p>Tags follow the Tag Index vocabulary, so a tag here is the same string as
  the matching tag in Zotero -- see the <a href="./tags.html">tag
  reference</a> for what each one means.</p>
</footer>
</div>

<script>
const ITEMS = __ITEMS_JSON__;
const TIER_LABELS = __TIER_LABELS_JSON__;
const TIER_ORDER = ["lead", "worth", "rest", "noise"];

const ReadStore = (() => {
  const KEY = "ed-brief:read";
  function safeLoad() {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch (e) {
      return new Set();
    }
  }
  let read = safeLoad();
  function persist() {
    try {
      localStorage.setItem(KEY, JSON.stringify([...read]));
    } catch (e) {
      // Private window or blocked site data: read-state just won't survive
      // a reload rather than breaking the page.
    }
  }
  return {
    isRead(id) { return read.has(id); },
    markRead(id) { read.add(id); persist(); },
    markUnread(id) { read.delete(id); persist(); },
    clear() { read = new Set(); persist(); },
  };
})();

function readFilterState() {
  const p = new URLSearchParams(location.search);
  // No ?tier= at all means "fresh visit": default to hiding "Everything
  // else" rather than showing all three tiers, so a muted or low-scoring
  // item doesn't have to be manually filtered out every time. An explicit
  // ?tier= (including an empty one, from clearing every tier chip) is a
  // deliberate choice and is always respected exactly.
  const tierParam = p.get("tier");
  const tiers = tierParam === null
    ? new Set(["lead", "worth"])
    : new Set(tierParam.split(",").filter(Boolean));
  return {
    tiers,
    tags: new Set((p.get("tag") || "").split(",").filter(Boolean)),
    sources: new Set((p.get("source") || "").split(",").filter(Boolean)),
    q: p.get("q") || "",
    showRead: p.get("read") === "1",
  };
}

function writeFilterState(state) {
  const p = new URLSearchParams();
  // Always written (even empty) so "explicitly cleared to show everything"
  // round-trips through a reload/bookmark instead of snapping back to the
  // lead+worth default the moment ?tier= disappears from the URL.
  p.set("tier", [...state.tiers].join(","));
  if (state.tags.size) p.set("tag", [...state.tags].join(","));
  if (state.sources.size) p.set("source", [...state.sources].join(","));
  if (state.q) p.set("q", state.q);
  if (state.showRead) p.set("read", "1");
  const qs = p.toString();
  history.replaceState(null, "", qs ? ("?" + qs) : location.pathname);
}

const state = readFilterState();
const list = document.getElementById("list");
const empty = document.getElementById("empty");
const countEl = document.getElementById("count");
const searchEl = document.getElementById("search");
const readToggle = document.getElementById("read-toggle");
const markAllBtn = document.getElementById("mark-all");

function initChips(rowId, key) {
  document.querySelectorAll("#" + rowId + " .chip[data-" + key + "]").forEach(btn => {
    const value = btn.dataset[key];
    const set = state[key + "s"];
    btn.setAttribute("aria-pressed", String(set.has(value)));
    btn.addEventListener("click", () => {
      if (set.has(value)) set.delete(value); else set.add(value);
      btn.setAttribute("aria-pressed", String(set.has(value)));
      writeFilterState(state);
      render();
    });
  });
}
initChips("tier-row", "tier");
initChips("tag-row", "tag");
initChips("source-row", "source");

searchEl.value = state.q;
searchEl.addEventListener("input", e => {
  state.q = e.target.value.toLowerCase().trim();
  writeFilterState(state);
  render();
});

readToggle.setAttribute("aria-pressed", String(state.showRead));
readToggle.addEventListener("click", () => {
  state.showRead = !state.showRead;
  readToggle.setAttribute("aria-pressed", String(state.showRead));
  writeFilterState(state);
  render();
});

markAllBtn.addEventListener("click", () => {
  for (const item of ITEMS.filter(matches)) ReadStore.markRead(item.id);
  render();
});

list.addEventListener("click", e => {
  const btn = e.target.closest("[data-toggle-read]");
  if (!btn) return;
  const id = btn.dataset.toggleRead;
  if (ReadStore.isRead(id)) ReadStore.markUnread(id); else ReadStore.markRead(id);
  render();
});

function matches(item) {
  if (state.tiers.size && !state.tiers.has(item.tier)) return false;
  if (state.sources.size && !state.sources.has(item.source_name)) return false;
  if (state.tags.size && !item.tags.some(t => state.tags.has(t))) return false;
  if (!state.showRead && ReadStore.isRead(item.id)) return false;
  if (state.q) {
    const hay = (item.title + " " + item.summary + " " + item.why).toLowerCase();
    if (!hay.includes(state.q)) return false;
  }
  return true;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : s;
  return d.innerHTML;
}

function card(item) {
  const tags = item.tags.map(t => `<span class="tag">${esc(t)}</span>`).join("");
  const read = ReadStore.isRead(item.id);
  const modeLabel = item.mode === "llm" ? "LLM-ranked" : "keyword-ranked";
  return `<article data-tier="${esc(item.tier)}" class="${read ? "is-read" : ""}">
    <h3>
      <span class="score" data-tier="${esc(item.tier)}" title="Rank score: relevance minus a small age decay -- this is what the list is sorted by.">${esc(item.score.toFixed(1))}</span>
      <a href="${esc(item.url)}" target="_blank" rel="noopener">${esc(item.title)}</a>
    </h3>
    <div class="meta">${esc(item.source_name)} &middot; ${esc(item.when)} &middot; <span class="mode">${modeLabel}</span></div>
    ${item.why ? `<p class="why">${esc(item.why)}</p>` : ""}
    ${item.summary ? `<p class="summary">${esc(item.summary)}</p>` : ""}
    <div class="row-bottom">
      <div class="tags">${tags}</div>
      <button class="read-btn" data-toggle-read="${esc(item.id)}">${read ? "Mark unread" : "Mark read"}</button>
    </div>
  </article>`;
}

function render() {
  // ITEMS arrives already sorted by rank_score at build time; this only
  // ever filters, never sorts -- ranking stays a build-time concern.
  const visible = ITEMS.filter(matches);
  const hiddenRead = state.showRead ? 0 : ITEMS.filter(i => ReadStore.isRead(i.id)).length;
  countEl.textContent = visible.length === ITEMS.length
    ? `${ITEMS.length} items`
    : `${visible.length} of ${ITEMS.length} items`;
  readToggle.textContent = state.showRead ? "Hide read" : `Show read (${hiddenRead})`;

  let out = "";
  for (const tier of TIER_ORDER) {
    const group = visible.filter(i => i.tier === tier);
    if (!group.length) continue;
    out += `<h2 data-tier="${tier}">${TIER_LABELS[tier]} <span style="font-weight:400;opacity:.6">(${group.length})</span></h2>`;
    out += group.map(card).join("");
  }
  list.innerHTML = out;
  empty.hidden = visible.length > 0;
}

render();
</script>
</body>
</html>
"""

ARCHIVE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brief archive</title>
<style>
:root { --bg:#fbfaf8; --surface:#fff; --border:#e4e0d8; --text:#23201c; --muted:#6c665d; --accent:#7a4b2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161513; --surface:#1e1d1a; --border:#33302b; --text:#ece8e1; --muted:#9b948a; --accent:#d09a6f; } }
body { margin:0; background:var(--bg); color:var(--text); font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:640px; margin:0 auto; padding:40px 20px; }
h1 { font-size:1.6rem; margin:0 0 4px; letter-spacing:-.02em; }
p.sub { color:var(--muted); font-size:.9rem; margin:0 0 28px; }
a { color:var(--accent); }
ul { list-style:none; padding:0; margin:0; }
li { border-bottom:1px solid var(--border); }
li a { display:flex; justify-content:space-between; gap:12px; padding:12px 2px; text-decoration:none; color:var(--text); }
li a:hover { color:var(--accent); }
li span { color:var(--muted); font-size:.82rem; white-space:nowrap; }
</style>
</head>
<body><div class="wrap">
<h1>Brief archive</h1>
<p class="sub"><a href="./">← Live list</a></p>
<ul>__ROWS__</ul>
</div></body>
</html>
"""

TAGS_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tag reference</title>
<style>
:root { --bg:#fbfaf8; --surface:#fff; --border:#e4e0d8; --text:#23201c; --muted:#6c665d; --accent:#7a4b2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161513; --surface:#1e1d1a; --border:#33302b; --text:#ece8e1; --muted:#9b948a; --accent:#d09a6f; } }
body { margin:0; background:var(--bg); color:var(--text); font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:720px; margin:0 auto; padding:40px 20px 80px; }
h1 { font-size:1.6rem; margin:0 0 4px; letter-spacing:-.02em; }
p.sub { color:var(--muted); font-size:.9rem; margin:0 0 22px; }
a { color:var(--accent); }
.domains { display:flex; flex-wrap:wrap; gap:7px; margin:0 0 26px; }
.domain {
  font-size:.76rem; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--border); border-radius:6px;
  padding:4px 10px; color:var(--muted);
}
.domain b { color:var(--text); }
table { width:100%; border-collapse:collapse; font-size:.87rem; }
th {
  text-align:left; font-size:.7rem; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); font-weight:600; padding:0 10px 8px; border-bottom:1px solid var(--border);
}
td { padding:10px; border-bottom:1px solid var(--border); vertical-align:top; }
td.tag { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.78rem; color:var(--accent); white-space:nowrap; }
td.tag .new { display:block; font-size:.66rem; color:var(--muted); font-family:inherit; margin-top:2px; }
td.count { color:var(--muted); text-align:right; white-space:nowrap; }
@media (max-width: 560px) {
  .wrap { padding: 24px 14px 60px; }
  th:nth-child(3), td.count { display:none; }
}
</style>
</head>
<body><div class="wrap">
<h1>Tag reference</h1>
<p class="sub"><a href="./">← Live list</a> &middot; these follow the Tag
Index vocabulary, so a tag here is the same string as the matching tag in
Zotero.</p>
<div class="domains">
  <span class="domain"><b>sub-</b> the history itself</span>
  <span class="domain"><b>disc-</b> historiography &amp; second-order concepts</span>
  <span class="domain"><b>ped-</b> pedagogy</span>
  <span class="domain"><b>prof-</b> professional practice</span>
  <span class="domain"><b>pol-</b> policy</span>
  <span class="domain"><b>cog-</b> cognitive science</span>
</div>
<table>
<thead><tr><th>Tag</th><th>What it means</th><th>Live items</th></tr></thead>
<tbody>__ROWS__</tbody>
</table>
</div></body>
</html>
"""


def _chips(pairs: list[tuple[str, str]], attr: str) -> str:
    return "".join(
        f'<button class="chip" data-{attr}="{html.escape(value, quote=True)}" '
        f'aria-pressed="false">{html.escape(label)}</button>'
        for value, label in pairs
    )


def _payload(items: list[CorpusItem]) -> list[dict]:
    return [
        {
            "id": item.id,
            "title": item.title,
            "url": item.url,
            "summary": item.summary,
            "source_name": item.source_name,
            "when": item.published.strftime("%-d %b, %H:%M"),
            "tier": item.tier,
            "tags": item.tags,
            "why": item.why,
            "mode": item.mode,
            # Round for display only -- the corpus/API keep full precision.
            "score": round(item.rank_score, 1),
        }
        for item in items
    ]


def _render_page(
    items: list[CorpusItem],
    title: str,
    heading: str,
    subtitle: str,
    meta: dict,
    generated_label: str,
    footer_extra: str = "",
) -> str:
    tags = sorted({t for i in items for t in i.tags})
    sources = sorted({i.source_name for i in items})

    down = [s for s in meta.get("sources", []) if not s.get("ok", True)]
    problems_html = ""
    if down:
        rows = "".join(
            f"<li>{html.escape(s['name'])}: {html.escape(s.get('detail') or 'failed')}</li>"
            for s in down
        )
        problems_html = (
            "<details><summary>Feeds that did not respond this run "
            f"({len(down)})</summary><ul>{rows}</ul></details>"
        )

    page = PAGE_TEMPLATE
    for token, value in {
        "__TITLE__": html.escape(title),
        "__HEADING__": heading,
        "__SUBTITLE__": html.escape(subtitle),
        "__TAG_CHIPS__": _chips([(t, t) for t in tags], "tag"),
        "__SOURCE_CHIPS__": _chips([(s, s) for s in sources], "source"),
        "__ITEMS_JSON__": json.dumps(_payload(items), ensure_ascii=False),
        "__TIER_LABELS_JSON__": json.dumps(TIER_LABELS),
        "__SCORING_LINE__": html.escape(meta.get("scoring", {}).get("status", "unknown")),
        "__GENERATED__": generated_label,
        "__RETENTION_DAYS__": str(meta.get("retention_days", 14)),
        "__PROBLEMS_HTML__": problems_html,
        "__FOOTER_EXTRA__": footer_extra,
    }.items():
        page = page.replace(token, value)
    return page


def write_site(
    live: list[CorpusItem],
    today_items: list[CorpusItem],
    meta: dict,
    root: Path,
    now: datetime,
    topics_cfg: list[dict],
) -> None:
    docs_dir = root / "docs"
    docs_briefs = docs_dir / "briefs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_briefs.mkdir(parents=True, exist_ok=True)

    generated_label = now.strftime("%-d %b %Y at %H:%M UTC")
    sources_count = len({i.source_name for i in live})

    index_page = _render_page(
        live,
        title="ed-brief",
        heading="Education brief",
        subtitle=(
            f"{len(live)} items from {sources_count} sources, ranked for history "
            f"teaching, routes into the profession, and curriculum policy. "
            f"Updated {generated_label}."
        ),
        meta=meta,
        generated_label=generated_label,
    )
    (docs_dir / "index.html").write_text(index_page, encoding="utf-8")
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")

    # Today's dated snapshot -- every item first seen today, matching the
    # markdown brief and the archive/YYYY-MM-DD.json API endpoint. Computed
    # once in build.py and passed in so a second run later the same day
    # accumulates into this page rather than a run-scoped list overwriting it.
    iso = now.strftime("%Y-%m-%d")
    date_label = now.strftime("%A %-d %B %Y")
    dated_page = _render_page(
        today_items,
        title=f"Education brief — {date_label}",
        heading=f"Education brief<br>{date_label}",
        subtitle=(
            f"{len(today_items)} items first seen today, ranked the same way as "
            "the live list."
        ),
        meta=meta,
        generated_label=generated_label,
        footer_extra=(
            "<p>This is a snapshot of items first seen today. See the "
            '<a href="../">live list</a> for current state, including this '
            "item if it's still within the retention window.</p>"
        ),
    )
    (docs_briefs / f"{iso}.html").write_text(dated_page, encoding="utf-8")

    _write_archive_index(root, docs_dir)
    _write_tags_page(live, topics_cfg, docs_dir)


def _write_archive_index(root: Path, docs_dir: Path) -> None:
    """Rebuilds the archive index from the API archive files already written
    this run (src/api.py runs before this in build.py), rather than parsing
    the rendered HTML back the way the proof of concept did -- the JSON is
    structured data we already produced, so there's nothing to reconstruct."""
    archive_dir = root / "docs" / "api" / "v1" / "archive"
    entries: list[tuple[str, int]] = []
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("*.json"), reverse=True):
            try:
                doc = json.loads(path.read_text("utf-8"))
                entries.append((doc["date"], doc.get("count", 0)))
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    rows = []
    for iso, count in entries:
        label = datetime.strptime(iso, "%Y-%m-%d").strftime("%A %-d %B %Y")
        rows.append(f'<li><a href="briefs/{iso}.html">{label}<span>{count} new items</span></a></li>')
    page = ARCHIVE_TEMPLATE.replace("__ROWS__", "".join(rows) or "<li><span>No briefs yet.</span></li>")
    (docs_dir / "archive.html").write_text(page, encoding="utf-8")


def _write_tags_page(live: list[CorpusItem], topics_cfg: list[dict], docs_dir: Path) -> None:
    """A plain-language lookup for the tag vocabulary, generated from
    scoring.yml's own `description` field so it can't drift out of sync with
    the config that actually produces these tags. Order follows scoring.yml
    (already curated by importance), not alphabetical."""
    counts: dict[str, int] = {}
    for item in live:
        for tag in item.tags:
            counts[tag] = counts.get(tag, 0) + 1

    rows = []
    for group in topics_cfg:
        tag = group.get("tag", "")
        description = html.escape(group.get("description", ""))
        new_marker = (
            '<span class="new">pending Tag Index addition</span>' if group.get("new") else ""
        )
        count = counts.get(tag, 0)
        rows.append(
            f"<tr><td class=\"tag\">{html.escape(tag)}{new_marker}</td>"
            f"<td>{description}</td><td class=\"count\">{count}</td></tr>"
        )
    page = TAGS_TEMPLATE.replace("__ROWS__", "".join(rows))
    (docs_dir / "tags.html").write_text(page, encoding="utf-8")
