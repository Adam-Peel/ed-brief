"""Once-per-build daily digest: a short flowing prose summary of today's
new items (build.py's `today_items`), written for LISTENING rather than
reading -- the intended eventual use is a script for a daily audio
briefing (see README), so it's prose, not a card list or a bullet
summary. That's also why every item gets at least brief coverage rather
than just the top few: a script that only mentioned "Read these" items
would miss the point of a daily *briefing*.

Deliberately unlisted, not unauthenticated -- there's no login here, just
no path TO it: not linked from docs/index.html, archive.html, or the API
endpoint index, and marked noindex so a search engine won't surface it
either. Reachable only at docs/digest.html for whoever already has the
URL, per an explicit "hidden, not on the main page or linked from it"
request.

Dormant unless ANTHROPIC_API_KEY is set, same contract as classify.py/
llm.py -- a missing key (or any failure) just means no digest.html gets
written this run, never a broken build.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from . import llm
from .corpus import CorpusItem

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
<footer><p>Not linked from the main site -- generated fresh each run from that day's new items.</p></footer>
</div>
</body>
</html>
"""


def _build_prompt(items: list[CorpusItem], reader_stage: str) -> str:
    ranked = sorted(items, key=lambda i: -i.relevance)
    listing = "\n".join(
        f"- [{item.relevance:.1f}/10, {item.tier}] {item.title} ({item.source_name}): "
        f"{item.why or item.summary[:150]}"
        for item in ranked
    )
    return f"""Write a spoken-word script for a short daily audio briefing \
for a career-changer in England entering secondary HISTORY teaching, \
{reader_stage}. This will be read aloud, so write flowing prose in \
complete sentences -- no bullet points, no headings, no markdown, nothing \
that only makes sense on a page.

Today's ranked items, most relevant first:

{listing}

Cover every item above at least briefly, in descending order of \
relevance -- the top few get a proper sentence or two each; minor or \
low-relevance ones can be grouped into a single closing sentence ("a few \
smaller items today: X, Y and Z") rather than skipped, since the point is \
a complete picture of the day, not just the highlights. Aim for about \
700-800 words, roughly five minutes spoken aloud. Open with a \
one-sentence overview of the day, close with a brief, natural sign-off. \
Write the finished script itself -- no preamble, no "Here's your \
briefing", just the words to be spoken."""


def _generate(items: list[CorpusItem], cfg: dict) -> tuple[str, str]:
    """Returns (script_text, status). Empty script means nothing gets
    written this run -- see write_digest."""
    if not items:
        return "", "no items today"
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
            messages=[{"role": "user", "content": _build_prompt(items, reader_stage)}],
        )
        text_block = llm._extract_text_block(response)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        return "", f"digest generation failed: {type(exc).__name__}: {exc}"

    return text_block.text.strip(), f"generated with {model}"


def write_digest(items: list[CorpusItem], cfg: dict, now: datetime, docs_dir: Path) -> str:
    """Writes docs/digest.html from today's items if a script was
    generated; leaves any existing digest.html untouched otherwise (a
    failed or key-less run shouldn't erase yesterday's, since this isn't
    dated/archived -- it's a single always-current page). Returns the
    status string for the caller to log, same shape as classify/llm's own
    status strings."""
    script, status = _generate(items, cfg)
    if not script:
        return status

    date_label = now.strftime("%A %-d %B %Y")
    paragraphs = "".join(f"<p>{html.escape(p.strip())}</p>" for p in script.split("\n\n") if p.strip())
    page = PAGE_TEMPLATE
    for token, value in {
        "__TITLE__": f"Daily digest — {date_label}",
        "__HEADING__": "Daily digest",
        "__SUBTITLE__": html.escape(f"{date_label} · {len(items)} items · not linked from the site"),
        "__BODY__": paragraphs,
    }.items():
        page = page.replace(token, value)

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "digest.html").write_text(page, encoding="utf-8")
    return status
