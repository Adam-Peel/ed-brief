"""Entry point: fetch, dedupe, score, rank, publish. Everything here runs
during ingestion -- the published site and API are fully pre-ranked static
data by the time this exits; nothing is scored, sorted, or fetched again at
page-view time.

Usage:
    python -m src.build
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import api, brief, llm, rss, site
from .corpus import (
    SCHEMA,
    CorpusItem,
    drop_expired,
    format_dt,
    load,
    publishable,
    recompute_rank,
    save,
)
from .fetch import Item, fetch_all
from .score import Scorer, blend_relevance

ROOT = Path(__file__).resolve().parent.parent

# Runs are ~12h apart (07:00 and 19:00 London), so a 24h window (2x the gap)
# means nothing is missed if a single run is skipped. Cross-run dedup is the
# corpus itself now, not a window -- unlike the proof of concept, an item
# doesn't need to still be inside this window to be recognised as
# already-seen. A backfill or a fresh corpus can widen this via
# BRIEF_WINDOW_HOURS (or the workflow's manual "Run workflow" window_hours
# input) without changing the default; a wider window is always safe to
# rerun since dedup is by id against the corpus, not the window.
DEFAULT_WINDOW_HOURS = 24


def load_config() -> tuple[list[dict], dict]:
    feeds = yaml.safe_load((ROOT / "config" / "feeds.yml").read_text("utf-8"))
    scoring = yaml.safe_load((ROOT / "config" / "scoring.yml").read_text("utf-8"))
    return feeds.get("feeds", []), scoring


def score_new_items(
    new_raw: list[Item],
    scorer: Scorer,
    verdicts: dict[str, tuple[float, str]],
    cfg: dict,
    now: datetime,
    retention_days: int,
) -> list[CorpusItem]:
    """Deterministically score, blend with any LLM verdict, and freeze each
    new item into a CorpusItem. Existing corpus items never pass through
    here -- their relevance was frozen on a previous run and stays that way."""
    llm_weight = float(cfg.get("llm_weight", 0.6))
    expires = now + timedelta(days=retention_days)

    frozen: list[CorpusItem] = []
    for item in new_raw:
        scorer.score_item(item)
        norm = scorer.normalize(item.deterministic_raw)
        verdict = verdicts.get(item.uid)
        llm_score, why = verdict if verdict else (None, "")
        relevance = blend_relevance(norm, llm_score, llm_weight)
        frozen.append(
            CorpusItem(
                id=item.uid,
                title=item.title,
                url=item.url,
                summary=item.summary,
                source_id=item.source_id,
                source_name=item.source_name,
                published=item.published,
                first_seen=now,
                expires=expires,
                relevance=relevance,
                tags=item.tags,
                why=why,
                image_url=item.image_url,
                deterministic_raw=item.deterministic_raw,
                deterministic_norm=norm,
                llm_score=llm_score,
                mode="llm" if verdict else "deterministic",
            )
        )
    return frozen


def build_meta(
    published: list[CorpusItem],
    new_items: list[CorpusItem],
    feeds: list[dict],
    feed_problems: list[str],
    llm_status: str,
    now: datetime,
    retention_days: int,
) -> dict:
    """Run metadata: per-source health, counts, scoring mode, retention. Feeds
    the API's meta.json and the site footer from one place so they can't
    disagree with each other. `published` and `new_items` should already be
    publish_floor-filtered -- meta.json's counts describe what's actually
    shown, not the full persisted corpus."""
    problem_detail = {}
    for problem in feed_problems:
        name, _, detail = problem.partition(": ")
        problem_detail[name] = detail or "failed"

    sources = [
        {
            "id": f["id"],
            "name": f["name"],
            "ok": f["name"] not in problem_detail,
            "detail": problem_detail.get(f["name"]),
        }
        for f in feeds
        if f.get("enabled", True)
    ]

    tier_counts = {"lead": 0, "worth": 0, "rest": 0, "noise": 0}
    for item in published:
        tier_counts[item.tier] = tier_counts.get(item.tier, 0) + 1

    llm_count = sum(1 for item in new_items if item.mode == "llm")

    return {
        "schema": SCHEMA,
        "generated": format_dt(now),
        "retention_days": retention_days,
        "counts": {
            "live": len(published),
            "new": len(new_items),
            "by_tier": tier_counts,
        },
        "scoring": {
            "status": llm_status,
            "new_llm_ranked": llm_count,
            "new_deterministic": len(new_items) - llm_count,
        },
        "sources": sources,
    }


def main(argv: list[str] | None = None, *, output_root: Path | None = None) -> int:
    """`output_root` lets tests redirect data/docs/briefs output to a temp
    directory while still reading the real project's config/ vocabulary --
    the config is repo-fixed, the output location isn't."""
    now = datetime.now(timezone.utc)
    feeds, cfg = load_config()
    retention_days = int(cfg.get("retention_days", 14))

    out_root = output_root or ROOT
    corpus_path = out_root / "data" / "corpus.json"

    # Stage 1: load the existing corpus.
    existing = load(corpus_path)
    existing_ids = {item.id for item in existing}

    # Stage 2: fetch every enabled feed.
    window_hours = int(os.environ.get("BRIEF_WINDOW_HOURS") or DEFAULT_WINDOW_HOURS)
    enabled = [f for f in feeds if f.get("enabled", True)]
    print(f"Fetching {len(enabled)} feeds ({window_hours}h window)…", file=sys.stderr)
    fetched, feed_problems = fetch_all(feeds, window_hours)
    print(f"  {len(fetched)} items retrieved", file=sys.stderr)
    for problem in feed_problems:
        print(f"  ! {problem}", file=sys.stderr)

    # Stage 3: dedupe against the corpus (within-run dedup already happened
    # inside fetch_all).
    new_raw = [item for item in fetched if item.uid not in existing_ids]
    print(f"  {len(new_raw)} new since last run", file=sys.stderr)

    # Stages 4-5: deterministic + LLM scoring, new items only.
    scorer = Scorer(cfg)
    verdicts, llm_status = llm.rerank_new_items(new_raw, cfg)
    print(f"  ranking: {llm_status}", file=sys.stderr)
    new_items = score_new_items(new_raw, scorer, verdicts, cfg, now, retention_days)

    # Stages 6-8: merge, expire, recompute rank_score/tier across everything.
    live = drop_expired(existing + new_items, now)
    recompute_rank(live, cfg, now)

    # Persist the FULL corpus (including anything below publish_floor)
    # before emitting anything derived from it -- dedup needs to remember
    # every item ever seen, published or not.
    save(corpus_path, live, now, retention_days)

    # Everything from here on is what actually gets published. Below
    # publish_floor is excluded entirely, not just hidden behind a toggle --
    # see the comment on publish_floor in scoring.yml.
    published = publishable(live, cfg)
    published_new = publishable(new_items, cfg)

    meta = build_meta(published, published_new, feeds, feed_problems, llm_status, now, retention_days)

    # Everything keyed by calendar date (the dated archive, the markdown
    # brief) needs every item first seen *today*, not just this run's new
    # items -- with more than one run a day, a run-scoped list would let a
    # later run's write silently erase an earlier run's items from that
    # day's record. latest.json is the one place that's deliberately
    # run-scoped (it documents "added by the most recent run").
    today_iso = now.strftime("%Y-%m-%d")
    today_items = [item for item in published if item.first_seen.strftime("%Y-%m-%d") == today_iso]

    # Stage 9: emit the API, the site, and today's dated markdown brief.
    api.write_all(published, published_new, today_items, meta, out_root, now)
    site.write_site(published, today_items, meta, out_root, now, cfg.get("topics", []))
    brief.write_brief(today_items, feed_problems, llm_status, out_root, now)
    rss.write_feed(published, cfg, out_root, now)

    print(
        f"\nWrote {len(published)} published items ({len(live) - len(published)} below "
        f"publish_floor, kept in the corpus but not shown) ({len(new_items)} new).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
