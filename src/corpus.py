"""The rolling corpus: data/corpus.json, the single source of truth for
everything the system knows about.

Replaces the proof of concept's state/seen.json. Where seen.json remembered
only a list of ids, the corpus remembers the full scored item, forever (or
until it expires), which is what makes cross-run dedup, retention, and a
persistent public API possible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1


def format_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_dt(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@dataclass
class CorpusItem:
    """One persisted story. `relevance` is frozen at ingest and never
    recomputed. `rank_score` and `tier` are recomputed for the whole corpus on
    every build (see recompute_rank below) -- they're still serialised so the
    committed corpus.json and the published API always show the last-computed
    values, not just what's true immediately after a rank happens to run."""

    id: str
    title: str
    url: str
    summary: str
    source_id: str
    source_name: str
    published: datetime
    first_seen: datetime
    expires: datetime
    relevance: float
    tags: list[str] = field(default_factory=list)
    why: str = ""
    image_url: str = ""
    deterministic_raw: float = 0.0
    deterministic_norm: float = 0.0
    llm_score: float | None = None
    mode: str = "deterministic"
    rank_score: float = 0.0
    tier: str = "rest"
    # Two-stage classify+score pipeline (src/classify.py, src/llm.py),
    # frozen at ingest alongside relevance -- never recomputed. `item_type`
    # (not `type`, to avoid shadowing the builtin) drives which section of
    # the site an item renders in; `locality` drives the build-time
    # percentile floor on rank_score (see recompute_rank below); `confidence`
    # is kept for debugging a classification that looks wrong, not currently
    # read by anything downstream. Defaults match what a pre-classify-stage
    # corpus item (i.e. one written before this pipeline existed) should be
    # treated as: OTHER/0/high, never IRRELEVANT -- an old item with no type
    # info at all must not silently disappear from every section.
    item_type: str = "OTHER"
    locality: int = 0
    confidence: str = "high"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "source": {"id": self.source_id, "name": self.source_name},
            "published": format_dt(self.published),
            "first_seen": format_dt(self.first_seen),
            "expires": format_dt(self.expires),
            "relevance": round(self.relevance, 2),
            "rank_score": round(self.rank_score, 2),
            "tier": self.tier,
            "type": self.item_type,
            "locality": self.locality,
            "confidence": self.confidence,
            "tags": self.tags,
            "why": self.why,
            "image_url": self.image_url,
            "scoring": {
                "mode": self.mode,
                "deterministic_raw": round(self.deterministic_raw, 2),
                "deterministic_norm": round(self.deterministic_norm, 2),
                "llm": round(self.llm_score, 2) if self.llm_score is not None else None,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusItem":
        scoring = d.get("scoring", {})
        source = d.get("source", {})
        llm_score = scoring.get("llm")
        return cls(
            id=d["id"],
            title=d["title"],
            url=d["url"],
            summary=d.get("summary", ""),
            source_id=source.get("id", ""),
            source_name=source.get("name", ""),
            published=parse_dt(d["published"]),
            first_seen=parse_dt(d["first_seen"]),
            expires=parse_dt(d["expires"]),
            relevance=float(d.get("relevance", 0.0)),
            tags=list(d.get("tags", [])),
            why=d.get("why", ""),
            image_url=d.get("image_url", ""),
            deterministic_raw=float(scoring.get("deterministic_raw", 0.0)),
            deterministic_norm=float(scoring.get("deterministic_norm", 0.0)),
            llm_score=float(llm_score) if llm_score is not None else None,
            mode=scoring.get("mode", "deterministic"),
            rank_score=float(d.get("rank_score", 0.0)),
            tier=d.get("tier", "rest"),
            item_type=d.get("type", "OTHER"),
            locality=int(d.get("locality", 0)),
            confidence=d.get("confidence", "high"),
        )


def load(path: Path) -> list[CorpusItem]:
    """Load the persisted item list. A missing or unreadable file means an
    empty corpus -- the normal state for the very first run."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [CorpusItem.from_dict(d) for d in raw.get("items", [])]


def save(
    path: Path, items: list[CorpusItem], generated: datetime, retention_days: int
) -> None:
    doc = {
        "schema": SCHEMA,
        "generated": format_dt(generated),
        "retention_days": retention_days,
        "items": [i.to_dict() for i in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def drop_expired(items: list[CorpusItem], now: datetime) -> list[CorpusItem]:
    return [i for i in items if i.expires > now]


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method), no
    external dependency. Robust to tiny inputs: empty -> 0.0, a single value
    -> that value, rather than raising the way statistics.quantiles() does
    below n=2."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (pct / 100.0) * (len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def recompute_rank(items: list[CorpusItem], cfg: dict, now: datetime) -> None:
    """Recompute rank_score and tier for the whole live corpus, in place, and
    sort by rank_score descending.

    Runs on every build, including runs with zero new items -- that's what
    keeps a quiet week's corpus.json changing (via rank_score and `generated`)
    rather than going stale, which in turn is what keeps GitHub from disabling
    the schedule after 60 days without a commit.
    """
    rank_cfg = cfg.get("rank_score", {})
    penalty_per_day = float(rank_cfg.get("age_penalty_per_day", 0.15))
    tiers = cfg.get("tiers", {})
    lead_cut = float(tiers.get("lead", 8.0))
    worth_cut = float(tiers.get("worth", 5.0))
    rest_cut = float(tiers.get("rest", 1.0))

    for item in items:
        age_days = max((now - item.first_seen).total_seconds() / 86400.0, 0.0)
        item.rank_score = max(0.0, item.relevance - age_days * penalty_per_day)
        item.tier = (
            "lead"
            if item.relevance >= lead_cut
            else "worth"
            if item.relevance >= worth_cut
            else "rest"
            if item.relevance >= rest_cut
            else "noise"
        )

    # Nottinghamshire premium: a build-time floor on rank_score, not summed
    # into relevance anywhere (locality is a tag, never a scoring input --
    # see scoring.yml's `locality` comment). Deliberately a PERCENTILE of
    # this build's own rank_score distribution, not a fixed number: a fixed
    # floor (this project tried 6.0 first) means whatever that number is
    # relative to a quiet fortnight can let a local story outrank real
    # curriculum/career news on a strong one. Computed from the un-floored
    # distribution BEFORE any floor is applied, so a local item's own score
    # never feeds back into the threshold being applied to it.
    locality_cfg = cfg.get("locality", {})
    floor_min_score = int(locality_cfg.get("floor_min_score", 3))
    floor_percentile = float(locality_cfg.get("floor_percentile", 85))
    if items:
        floor_value = _percentile([i.rank_score for i in items], floor_percentile)
        for item in items:
            if item.locality >= floor_min_score:
                item.rank_score = max(item.rank_score, floor_value)

    items.sort(key=lambda i: (-i.rank_score, -i.published.timestamp()))


def publishable(items: list[CorpusItem], cfg: dict) -> list[CorpusItem]:
    """Items below `publish_floor` are excluded from every published surface
    -- the site, the API, the archive -- even though they stay in the
    persisted corpus (see the comment on publish_floor in scoring.yml for
    why: dedup still needs to remember them, or they'd be re-fetched and
    re-scored by the LLM every run for no benefit). Call this on the full
    corpus right before writing anything public; `save()` itself should
    always get the unfiltered list.

    Checked against rank_score, not the frozen relevance -- the floor is
    meant to match what a reader actually sees on the site (which displays
    rank_score), so a story that ages past the cut disappears from the feed
    even though its relevance never changes. recompute_rank() must run on
    `items` before this, which build.py already guarantees."""
    floor = float(cfg.get("publish_floor", 0.0))
    return [i for i in items if i.rank_score >= floor]
