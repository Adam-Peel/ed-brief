"""Weekly digest: a flowing prose round-up of the last 7 days' items,
written for LISTENING rather than reading -- the intended eventual use is
a script for a weekly audio briefing (see README), so it's prose, not a
card list. Fires once a week (Sunday morning, London time), not on every
run -- morning rather than evening so little enough has published
overnight that the episode is ready in time for a lazy Sunday afternoon
listen, not held back to Sunday evening.

Started life daily, scoped to `today_items` -- real data killed that: a
quiet day (a Saturday, as it happened) had six new items, the best of
which scored 4.64, below this project's own "worth a look" bar. Not
enough material, and not good enough material either. Weekly fixes both
at once, and incidentally fixes a second problem for free: `today_items`
never covered anything not first_seen on the exact calendar day a digest
ran, so anything ingested on a thin day was silently uncoverable forever.
A weekly window covers everything that arrived in the last 7 days
regardless of which specific day, with no separate rolling-window-or-
tracking mechanism needed, since the coverage window and the episode
cadence are now the same period.

Content is deliberately gated, not "everything new" -- owner request:
full coverage for anything that cleared scoring.yml's `worth` tier cut
(read from config, not hardcoded, so it can't drift out of sync with the
site's own bar), plus a brief mention for any HISTORY-typed item that
didn't clear that bar, since history content matters to this reader even
when the general relevance rubric doesn't rate it highly. Everything else
-- a low-scoring CAREER or SECTOR item, say -- is left out entirely,
not grouped into a passing mention the way the daily version tried to
cover everything.

Deliberately unlisted, not unauthenticated -- there's no login here, just
no path TO it: not linked from docs/index.html, archive.html, or the API
endpoint index, and marked noindex so a search engine won't surface it
either. Reachable only at docs/digest.html for whoever already has the
URL, per an explicit "hidden, not on the main page or linked from it"
request.

Dormant unless ANTHROPIC_API_KEY is set, same contract as classify.py/
llm.py -- a missing key (or any failure, or simply not being Sunday
morning) just means no digest.html gets written this run, never a broken
build.
"""

from __future__ import annotations

import html
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from . import llm
from .corpus import CorpusItem

LONDON = ZoneInfo("Europe/London")
WINDOW_DAYS = 7

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>__TITLE__</title>
<style>
:root {
  --bg: #fbfaf8; --surface: #ffffff; --border: #e4e0d8;
  --text: #23201c; --muted: #6c665d; --accent: #7a4b2a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161513; --surface: #1e1d1a; --border: #33302b;
    --text: #ece8e1; --muted: #9b948a; --accent: #d09a6f;
  }
}
body { margin: 0; background: var(--bg); color: var(--text);
  font: 17px/1.7 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 640px; margin: 0 auto; padding: 40px 20px 80px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; letter-spacing: -.02em; }
p.sub { color: var(--muted); font-size: .88rem; margin: 0 0 28px; }
.script p { margin: 0 0 1.1em; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--border);
  font-size: .78rem; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
<h1>__HEADING__</h1>
<p class="sub">__SUBTITLE__</p>
<div class="script">__BODY__</div>
<footer><p>Not linked from the main site -- generated once a week from the last 7 days' items.</p></footer>
</div>
</body>
</html>
"""


def _is_weekly_run(now: datetime) -> bool:
    """True on the Sunday MORNING run specifically (London time) -- owner
    request: little enough gets published overnight Saturday-Sunday that
    the morning run's episode is effectively ready for the same day, to
    listen to that afternoon, rather than making it wait for Sunday
    evening. `hour < 12` rather than an exact "07" match: the workflow's
    own shell guard already restricts every run that reaches this code to
    London hour 07 or 19 (nothing else gets this far), so this only ever
    needs to tell those two apart, robust to the run landing later than
    07:00 sharp under scheduling delay (a real, observed thing -- see
    scoring.yml's llm_classify_batch_size note on GitHub's own documented
    congestion) without hardcoding the literal hour twice in two places.
    Checked in local time rather than UTC to match the same "what day/
    run does this feel like to the reader" logic the workflow's guard
    already uses."""
    london_now = now.astimezone(LONDON)
    return london_now.weekday() == 6 and london_now.hour < 12  # Sunday, Monday=0..Sunday=6


def _select(items: list[CorpusItem], cfg: dict) -> tuple[list[CorpusItem], list[CorpusItem]]:
    """Splits the week's items into (main, history_mentions) per the
    owner's selection rule. `main` is full-coverage material: anything
    that cleared the site's own `worth` tier cut, any type. Everything
    else is dropped UNLESS it's a HISTORY item, in which case it still
    gets a brief mention -- see the module docstring for why this is a
    real gate, not "everything, grouped" the way the daily version was."""
    worth_cut = float(cfg.get("tiers", {}).get("worth", 4.95))
    main = sorted(
        (item for item in items if item.relevance >= worth_cut),
        key=lambda i: -i.relevance,
    )
    history_mentions = sorted(
        (item for item in items if item.relevance < worth_cut and item.item_type == "HISTORY"),
        key=lambda i: -i.relevance,
    )
    return main, history_mentions


def _build_prompt(main: list[CorpusItem], history_mentions: list[CorpusItem], reader_stage: str) -> str:
    main_listing = "\n".join(
        f"- [{item.relevance:.1f}/10, {item.tier}] {item.title} ({item.source_name}): "
        f"{item.why or item.summary[:150]}"
        for item in main
    )
    history_block = ""
    history_instruction = ""
    if history_mentions:
        history_titles = "\n".join(
            f"- {item.title} ({item.source_name}): {item.why or item.summary[:150]}"
            for item in history_mentions
        )
        history_block = f"\n\nHistory items that didn't clear the bar above, but are still worth a brief mention:\n\n{history_titles}"
        history_instruction = (
            " After the main stories, briefly mention the history items listed "
            "separately above too -- one short clause each, grouped into a "
            "sentence or two, not full individual treatment."
        )

    return f"""Write a spoken-word script for a weekly audio round-up for \
a career-changer in England entering secondary HISTORY teaching, \
{reader_stage}. This will be read aloud, so write flowing prose in \
complete sentences -- no bullet points, no headings, no markdown, nothing \
that only makes sense on a page.

This week's main stories, most relevant first -- each one already \
cleared this project's own "worth a look" bar:

{main_listing}{history_block}

Give the strongest few stories a proper sentence or two each, in \
descending order of relevance; every other main story above still gets \
at least one clear sentence of its own, none skipped, since all of them \
already cleared a real quality bar to be here.{history_instruction} Open \
with a one-sentence overview of the week, close with a brief, natural \
sign-off. Let the length follow how much is actually here -- a quiet \
week might run four or five minutes, a busy one nearer ten; don't pad a \
thin week out or cut a strong one short to hit a fixed target. Write the \
finished script itself -- no preamble, no "Here's your round-up", just \
the words to be spoken."""


def _generate(main: list[CorpusItem], history_mentions: list[CorpusItem], cfg: dict) -> tuple[str, str]:
    """Returns (script_text, status). Empty script means nothing gets
    written this run -- see write_digest."""
    if not main and not history_mentions:
        return "", "no items cleared the weekly digest's selection this week"
    if not llm.available():
        return "", "deterministic only (no API key set)"

    import anthropic

    model = llm._resolve_model(cfg)
    reader_stage = cfg.get("llm_reader_stage", "at the point of entering initial teacher training")
    max_tokens = int(cfg.get("llm_max_tokens", 12000))

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 -- must not break the run
        return "", f"digest generation failed (client init: {type(exc).__name__})"

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": _build_prompt(main, history_mentions, reader_stage)}],
        )
        text_block = llm._extract_text_block(response)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        return "", f"digest generation failed: {type(exc).__name__}: {exc}"

    return text_block.text.strip(), f"generated with {model}"


def write_digest(published: list[CorpusItem], cfg: dict, now: datetime, docs_dir: Path) -> str:
    """Writes docs/digest.html from the last WINDOW_DAYS' items if a
    script was generated; leaves any existing digest.html untouched
    otherwise (a failed, key-less, or non-Sunday run shouldn't erase last
    week's, since this isn't dated/archived -- it's a single always-
    current page). `published` is the full live/publishable list, not
    pre-windowed -- windowing and the worth-cut/HISTORY split both happen
    in here. Returns the status string for the caller to log, same shape
    as classify/llm's own status strings."""
    if not _is_weekly_run(now):
        return "not the weekly run (Sunday morning, London time)"

    window_start = now - timedelta(days=WINDOW_DAYS)
    recent = [item for item in published if item.first_seen >= window_start]
    main, history_mentions = _select(recent, cfg)

    script, status = _generate(main, history_mentions, cfg)
    if not script:
        return status

    date_label = now.strftime("%A %-d %B %Y")
    paragraphs = "".join(f"<p>{html.escape(p.strip())}</p>" for p in script.split("\n\n") if p.strip())
    covered = len(main) + len(history_mentions)
    page = PAGE_TEMPLATE
    for token, value in {
        "__TITLE__": f"Weekly digest — {date_label}",
        "__HEADING__": "Weekly digest",
        "__SUBTITLE__": html.escape(
            f"Week ending {date_label} · {covered} items ({len(main)} main, "
            f"{len(history_mentions)} history mentions) · not linked from the site"
        ),
        "__BODY__": paragraphs,
    }.items():
        page = page.replace(token, value)

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "digest.html").write_text(page, encoding="utf-8")
    return status
