"""RSS 2.0 feed of the published corpus, for any standard feed reader --
Feeder, NetNewsWire, Feedly, whatever the reader already uses for every
other source. Static output of the already-ranked corpus, generated at
build time exactly like the JSON API and the site; a reader polling this
is a normal static-file fetch, no different from polling any other feed --
nothing here is computed at request time.

The custom docs/api/v1/*.json shape is deliberately NOT what this is --
that's a bespoke contract for a future purpose-built client (e.g. a mobile
app) that can be taught its exact fields. RSS 2.0 is for interoperating
with software this project doesn't control and never will.
"""

from __future__ import annotations

import html
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from .corpus import CorpusItem

TIER_LABELS = {"lead": "Read these", "worth": "Worth a look", "rest": "Everything else", "noise": "Low relevance"}

# Only lead/worth by default -- the same "worth your attention" bar the site
# itself defaults to showing. The full corpus, including "Everything else"
# and "Low relevance", is still there for anyone who wants it via the JSON
# API; a reader's inbox is a worse place than a browsable list for items
# that are borderline by design.
FEED_MIN_TIER = {"lead", "worth"}


def _cdata(text: str) -> str:
    """Escape the one sequence that would otherwise prematurely close a CDATA
    section. Everything else inside CDATA is literal, unescaped markup."""
    return text.replace("]]>", "]]]]><![CDATA[>")


def _item_xml(item: CorpusItem) -> str:
    mode_label = "LLM-ranked" if item.mode == "llm" else "keyword-ranked"
    body_parts = []
    if item.why:
        body_parts.append(f"<p><strong>Why this matters:</strong> {html.escape(item.why)}</p>")
    if item.summary:
        body_parts.append(f"<p>{html.escape(item.summary)}</p>")
    body_parts.append(
        f"<p><em>{html.escape(TIER_LABELS.get(item.tier, item.tier))} &middot; "
        f"score {item.rank_score:.1f} &middot; {mode_label} &middot; "
        f"{html.escape(item.source_name)}</em></p>"
    )
    # CDATA-wrapped, not escaped-and-inlined: <description> must contain TEXT
    # per the RSS spec, not child elements. Inlining raw <p> tags directly
    # (the first version of this function did) is well-formed XML but means
    # element.find("description").text comes back empty, since ElementTree
    # -- and most feed readers -- treat the <p> as a child element the
    # moment it isn't escaped or wrapped, not as part of the description's
    # own text. CDATA is what lets the HTML stay literal while still being
    # unambiguously "this element's text content" to any parser.
    description_html = _cdata("".join(body_parts))

    categories = "".join(f"<category>{escape(tag)}</category>" for tag in item.tags)

    return (
        "<item>"
        f"<title>{escape(item.title)}</title>"
        f"<link>{escape(item.url)}</link>"
        f"<guid isPermaLink=\"true\">{escape(item.url)}</guid>"
        f"<pubDate>{format_datetime(item.published)}</pubDate>"
        f"<description><![CDATA[{description_html}]]></description>"
        f"{categories}"
        f"<source url={quoteattr(item.url)}>{escape(item.source_name)}</source>"
        "</item>"
    )


def render_feed(items: list[CorpusItem], cfg: dict, now: datetime) -> str:
    site_url = cfg.get("site_url", "").rstrip("/") + "/"
    feed_items = [i for i in items if i.tier in FEED_MIN_TIER]
    # Chronological (first seen, newest first), not rank_score -- feed
    # readers do their own new-item detection and unread tracking, which
    # assumes a normal reverse-chronological stream. Ranking still happened
    # at build time (every item's relevance was frozen then); this is just
    # the order a reader expects, not a re-ranking.
    feed_items.sort(key=lambda i: i.first_seen, reverse=True)

    items_xml = "".join(_item_xml(i) for i in feed_items)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        "<channel>"
        "<title>ed-brief</title>"
        f"<link>{escape(site_url)}</link>"
        "<description>Education news for a career-changer entering secondary "
        "history teaching in England, pre-ranked at ingest -- nothing on this "
        "feed is re-sorted or re-scored by your reader.</description>"
        "<language>en-gb</language>"
        f"<lastBuildDate>{format_datetime(now)}</lastBuildDate>"
        "<ttl>360</ttl>"
        f'<atom:link href="{escape(site_url)}feed.xml" rel="self" type="application/rss+xml" />'
        f"{items_xml}"
        "</channel>\n"
        "</rss>\n"
    )


def write_feed(items: list[CorpusItem], cfg: dict, root: Path, now: datetime) -> None:
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "feed.xml").write_text(render_feed(items, cfg, now), encoding="utf-8")
