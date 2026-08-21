"""Fetch and normalise items from the configured feeds."""

from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser
import requests

USER_AGENT = (
    "ed-brief/1.0 (+https://github.com/) personal education news digest; "
    "python-requests"
)
TIMEOUT = 25
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_AVATAR_HOSTS = ("gravatar.com",)  # WordPress/Jetpack author avatars, not article images


@dataclass
class Item:
    """One story, normalised across feed formats. Ephemeral -- lives only for
    the duration of one ingest run. Anything that needs to persist becomes a
    corpus.CorpusItem once scoring is complete; see src/corpus.py."""

    uid: str
    title: str
    summary: str
    url: str
    source_id: str
    source_name: str
    published: datetime
    source_weight: float = 0.0

    image_url: str = ""

    # Filled in during ingestion: score.py then, optionally, llm.py.
    tags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    deterministic_raw: float = 0.0
    llm_score: float | None = None
    why: str = ""

    @property
    def age_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.published
        return max(delta.total_seconds() / 3600.0, 0.0)

    def haystack(self) -> tuple[str, str]:
        """Lowercased (title, body) used for term matching."""
        return self.title.lower(), self.summary.lower()


_SMART_QUOTES = str.maketrans({
    "‘": "'", "’": "'",  # curly single quotes/apostrophe
    "“": '"', "”": '"',  # curly double quotes
})


def clean_text(raw: str | None, limit: int = 600) -> str:
    """Strip markup and entities out of a feed summary.

    Also normalises curly quotes to straight ones -- found the hard way:
    scoring.yml terms like "teachers' standards" or "children's home" use a
    plain ASCII apostrophe, but plenty of real feeds (professionally
    typeset copy, mostly) use a proper U+2019 curly apostrophe instead, and
    word-boundary term matching treats those as different characters
    entirely. Normalising once here, at ingest, means every term in
    scoring.yml only ever needs the one straight-quote spelling.
    """
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = text.translate(_SMART_QUOTES)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > limit:
        cut = text[:limit].rsplit(" ", 1)[0]
        text = cut + "…"
    return text


def _parse_date(entry) -> datetime:
    """Best-effort publication date, defaulting to now if the feed omits one."""
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)
            except (ValueError, OverflowError, TypeError):
                continue
    return datetime.now(timezone.utc)


def _is_real_image(url: str) -> bool:
    return bool(url) and not any(host in url for host in _AVATAR_HOSTS)


def _extract_image(entry) -> str:
    """Best-effort article image, since different feeds represent this
    completely differently -- checked in the order each is likely to be a
    genuine, appropriately-sized article image rather than noise:

    1. media:thumbnail -- explicitly a thumbnail when present.
    2. media:content -- may list several sizes of the same photo (Guardian)
       or, on some WordPress/Jetpack feeds, the author's Gravatar avatar
       ahead of the real photo (Nottingham Hidden History Team) -- avatar
       hosts are filtered out, and the largest remaining width wins.
    3. An enclosure link with an image MIME type.
    4. A regex pull of the first <img src> in the raw (pre-clean_text) HTML
       body, for feeds with no structured image data at all.

    Returns "" rather than guessing when nothing usable is found -- a
    missing thumbnail is normal and every consumer treats it as optional.
    """
    for thumb in entry.get("media_thumbnail", []):
        url = thumb.get("url", "")
        if _is_real_image(url):
            return url

    candidates = [c for c in entry.get("media_content", []) if _is_real_image(c.get("url", ""))]
    if candidates:
        def width(c):
            try:
                return int(c.get("width", 0))
            except (TypeError, ValueError):
                return 0
        return max(candidates, key=width)["url"]

    for link in entry.get("links", []):
        if link.get("rel") == "enclosure" and link.get("type", "").startswith("image/"):
            if _is_real_image(link.get("href", "")):
                return link["href"]

    raw_html = entry.get("summary") or entry.get("description") or ""
    if not raw_html and entry.get("content"):
        raw_html = entry["content"][0].get("value", "")
    match = _IMG_SRC_RE.search(raw_html)
    if match and _is_real_image(match.group(1)):
        return match.group(1)

    return ""


def _make_uid(url: str, title: str) -> str:
    """Stable identity for deduplication.

    Keyed on the URL with query strings stripped, so the same story picked up
    with different tracking parameters is recognised as one item. Falls back
    to the title when a feed omits the link entirely.
    """
    basis = url.split("?")[0].rstrip("/") if url else title
    return hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:16]


def parse_feed(
    content: bytes, feed_cfg: dict, window_hours: int
) -> tuple[list[Item], str | None]:
    """Turn raw feed bytes into Items. Kept separate from the HTTP call so the
    normalisation can be tested against fixtures without network access."""
    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        return [], f"unparseable: {parsed.get('bozo_exception', 'unknown')}"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    items: list[Item] = []

    for entry in parsed.entries:
        published = _parse_date(entry)
        if published < cutoff:
            continue

        title = clean_text(entry.get("title"), limit=300)
        if not title:
            continue

        summary = clean_text(
            entry.get("summary") or entry.get("description") or ""
        )
        link = (entry.get("link") or "").strip()

        items.append(
            Item(
                uid=_make_uid(link, title),
                title=title,
                summary=summary,
                url=link,
                source_id=feed_cfg["id"],
                source_name=feed_cfg["name"],
                published=published,
                source_weight=float(feed_cfg.get("weight", 0.0)),
                image_url=_extract_image(entry),
            )
        )

    return items, None


def fetch_feed(feed_cfg: dict, window_hours: int) -> tuple[list[Item], str | None]:
    """Fetch one feed over HTTP. Returns (items, error_message)."""
    url = feed_cfg.get("url") or ""
    if not url:
        return [], "no URL configured"

    try:
        response = requests.get(
            url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return [], f"{type(exc).__name__}: {exc}"

    return parse_feed(response.content, feed_cfg, window_hours)


def fetch_all(feeds: list[dict], window_hours: int) -> tuple[list[Item], list[str]]:
    """Fetch every enabled feed, collecting errors rather than raising.

    A dead feed degrades the brief; it should never break the run. Errors are
    returned so the workflow can print them and you notice a rotted URL.
    """
    all_items: list[Item] = []
    problems: list[str] = []
    seen_uids: set[str] = set()

    for feed_cfg in feeds:
        if not feed_cfg.get("enabled", True):
            continue

        items, error = fetch_feed(feed_cfg, window_hours)
        if error:
            problems.append(f"{feed_cfg['name']}: {error}")
            continue
        if not items:
            continue

        # Within-run dedup: the Guardian's education and schools feeds overlap
        # heavily, so first source wins and the duplicate is dropped.
        for item in items:
            if item.uid in seen_uids:
                continue
            seen_uids.add(item.uid)
            all_items.append(item)

    return all_items, problems
