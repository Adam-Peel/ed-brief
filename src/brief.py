"""Dated markdown brief of each run's new items, committed to briefs/ as a
searchable archive that outlives any hosting decision.

A close port of the proof of concept's render_markdown, adapted to
CorpusItem. Unlike docs/index.html, a brief file only ever covers ONE run's
new stories -- never the rolling corpus -- so briefs/ stays a readable
chronological log even though the site itself is not.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .corpus import CorpusItem

TIER_LABELS = {"lead": "Read these", "worth": "Worth a look", "rest": "Everything else"}
TIER_ORDER = ["lead", "worth", "rest"]


def render_markdown(
    items: list[CorpusItem],
    run_date: datetime,
    feed_problems: list[str],
    llm_status: str,
) -> str:
    lines = [f"# Education brief — {run_date.strftime('%A %-d %B %Y')}", ""]

    if not items:
        lines += ["No new items since the last run.", ""]
    else:
        lines += [
            f"{len(items)} new items from {len({i.source_id for i in items})} sources. "
            "Ranked by relevance to history teaching, routes into the profession, "
            "and curriculum policy.",
            "",
        ]
        for tier in TIER_ORDER:
            tier_items = [i for i in items if i.tier == tier]
            if not tier_items:
                continue
            lines += [f"## {TIER_LABELS[tier]}", ""]
            for item in tier_items:
                lines.append(f"### [{item.title}]({item.url})")
                lines.append("")
                lines.append(f"*{item.source_name} · {item.published.strftime('%-d %b, %H:%M')}*")
                lines.append("")
                if item.why:
                    lines.append(f"**Why this matters:** {item.why}")
                    lines.append("")
                if item.summary:
                    lines.append(item.summary)
                    lines.append("")
                if item.tags:
                    lines.append(" ".join(f"`{t}`" for t in item.tags))
                    lines.append("")

    lines += ["---", "", f"*Ranking: {llm_status}.*", ""]
    if feed_problems:
        lines.append("*Feeds that did not respond this run:*")
        lines.append("")
        for problem in feed_problems:
            lines.append(f"- {problem}")
        lines.append("")
    return "\n".join(lines)


def write_brief(
    items: list[CorpusItem],
    feed_problems: list[str],
    llm_status: str,
    root: Path,
    run_date: datetime,
) -> Path:
    briefs_dir = root / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    path = briefs_dir / f"{run_date:%Y-%m-%d}.md"
    path.write_text(render_markdown(items, run_date, feed_problems, llm_status), encoding="utf-8")
    return path
