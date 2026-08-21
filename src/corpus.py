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
    deterministic_raw: float = 0.0
    deterministic_norm: float = 0.0
    llm_score: float | None = None
    mode: str = "deterministic"
    rank_score: float = 0.0
    tier: str = "rest"

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
            "tags": self.tags,
            "why": self.why,
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
            deterministic_raw=float(scoring.get("deterministic_raw", 0.0)),
            deterministic_norm=float(scoring.get("deterministic_norm", 0.0)),
            llm_score=float(llm_score) if llm_score is not None else None,
            mode=scoring.get("mode", "deterministic"),
            rank_score=float(d.get("rank_score", 0.0)),
            tier=d.get("tier", "rest"),
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

    items.sort(key=lambda i: (-i.rank_score, -i.published.timestamp()))
