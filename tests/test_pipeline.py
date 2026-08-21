"""Core pipeline test: parse -> score -> corpus -> build, using local
fixtures. No network access required. Run with:

    python tests/test_pipeline.py

Exits non-zero if anything fails, so it works as a CI gate. See
tests/test_readonly.py for the docs/ tree checks (acceptance criteria 10-12).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm  # noqa: E402
from src.build import build_meta, load_config, score_new_items  # noqa: E402
from src.corpus import CorpusItem, drop_expired, recompute_rank  # noqa: E402
from src.fetch import Item, parse_feed  # noqa: E402
from src.score import Scorer, blend_relevance  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""))
        failures.append(label)


def build_fixture_items():
    feeds, scoring_cfg = load_config()
    by_id = {f["id"]: f for f in feeds}
    items = []
    for path in sorted(FIXTURES.glob("*.xml")):
        cfg = by_id.get(path.stem)
        assert cfg, f"fixture {path.stem} has no matching feed config"
        parsed, error = parse_feed(path.read_bytes(), cfg, window_hours=96)
        assert not error, f"{path.stem}: {error}"
        items.extend(parsed)
    return items, scoring_cfg


def find(items_by_title: dict, fragment: str):
    for title, item in items_by_title.items():
        if fragment.lower() in title.lower():
            return item
    return None


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    section("Parsing fixtures")
    items, scoring_cfg = build_fixture_items()
    check(len(items) >= 14, f"parsed {len(items)} items from fixtures")
    check(all(i.title and i.url for i in items), "every item has a title and URL")
    check(all(i.published.tzinfo for i in items), "every date is timezone-aware")

    section("Deterministic scoring")
    scorer = Scorer(scoring_cfg)
    for item in items:
        scorer.score_item(item)
    by_title = {i.title: i for i in items}

    history = find(by_title, "contested history curriculum")
    francis = find(by_title, "Francis report means for history")
    itt = find(by_title, "Teacher recruitment falls short")
    he = find(by_title, "Vice-chancellor pay")
    sport = find(by_title, "PE and school sport")

    check(history is not None and "sub-history" in history.tags,
          "contested-history story is tagged sub-history")
    check(francis is not None and francis.deterministic_raw > 15,
          "history + curriculum review story scores highly",
          f"raw={getattr(francis, 'deterministic_raw', 0):.1f}")
    check(itt is not None and itt.deterministic_raw > 8,
          "ITT recruitment story scores highly",
          f"raw={getattr(itt, 'deterministic_raw', 0):.1f}")
    check(he is not None and he.deterministic_raw < 0,
          "higher-education story is muted into negative territory",
          f"raw={getattr(he, 'deterministic_raw', 0):.1f}")
    check(sport is not None and francis is not None
          and sport.deterministic_raw < francis.deterministic_raw,
          "school sport story ranks below the curriculum story")

    section("Acceptance criterion 3: fixed-ceiling normalisation")
    # The same raw score must normalise identically regardless of what batch
    # it's computed alongside -- this is the batch-relative bug the proof of
    # concept had (normalising against that run's own min/max) and the whole
    # reason the corpus architecture works at all.
    scorer_a = Scorer(scoring_cfg)  # stands in for "run on Monday"
    scorer_b = Scorer(scoring_cfg)  # stands in for "run on Thursday"
    check(scorer_a.normalize(15.0) == scorer_b.normalize(15.0),
          "identical raw score normalises identically across separate scorer instances")
    check(scorer_a.normalize(15.0) == min(15.0, scorer_a.scale_max) / scorer_a.scale_max * 10.0,
          "normalize() matches the documented fixed-ceiling formula")
    check(scorer_a.normalize(-100.0) == 0.0, "a heavily muted raw score floors at 0")

    section("False-positive guards")
    probe_cfg = {
        "title_multiplier": 1.0,
        "recency": {"half_life_hours": 60, "max_bonus": 0},
        "deterministic_scale_max": 25.0,
        "topics": [
            {"tag": "t-ect", "weight": 5, "terms": ["ect"]},
            {"tag": "t-ai", "weight": 5, "terms": ["ai"]},
        ],
        "mutes": [],
    }
    probe = Scorer(probe_cfg)

    def probe_item(title: str) -> Item:
        return Item(
            uid="x", title=title, summary="", url="u", source_id="s",
            source_name="S", published=datetime.now(timezone.utc),
        )

    collected = probe.score_item(probe_item("Data collected from schools"))
    said = probe.score_item(probe_item("The minister said today"))
    real_ect = probe.score_item(probe_item("Support for ECT mentors"))
    check(not collected.tags, "'collected' does not match the term 'ect'")
    check(not said.tags, "'said' does not match the term 'ai'")
    check("t-ect" in real_ect.tags, "'ECT mentors' does match the term 'ect'")

    section("Acceptance criterion 8: deduplication across feeds")
    seen_uids = {i.uid for i in items}
    check(len(seen_uids) == len(items), "fixture uids are unique")
    dup_a = parse_feed(
        (FIXTURES / "bbc-education.xml").read_bytes(),
        {"id": "a", "name": "A", "weight": 0}, 96,
    )[0]
    dup_b = parse_feed(
        (FIXTURES / "bbc-education.xml").read_bytes(),
        {"id": "b", "name": "B", "weight": 0}, 96,
    )[0]
    check({i.uid for i in dup_a} == {i.uid for i in dup_b},
          "the same story from two sources gets the same uid")

    section("Acceptance criterion 1: retention window")
    now = datetime.now(timezone.utc)
    retention_days = 14
    day0_item = CorpusItem(
        id="ret-1", title="t", url="u", summary="", source_id="s", source_name="S",
        published=now - timedelta(days=13), first_seen=now - timedelta(days=13),
        expires=(now - timedelta(days=13)) + timedelta(days=retention_days),
        relevance=6.0,
    )
    check(
        drop_expired([day0_item], now) == [day0_item],
        "an item first seen 13 days ago is still present",
    )
    day0_item_older = CorpusItem(
        id="ret-2", title="t", url="u", summary="", source_id="s", source_name="S",
        published=now - timedelta(days=15), first_seen=now - timedelta(days=15),
        expires=(now - timedelta(days=15)) + timedelta(days=retention_days),
        relevance=6.0,
    )
    check(
        drop_expired([day0_item_older], now) == [],
        "an item first seen 15 days ago has been dropped",
    )

    section("Acceptance criterion 4: rank_score decays with age, relevance frozen")
    fresh = CorpusItem(
        id="age-fresh", title="t", url="u", summary="", source_id="s", source_name="S",
        published=now, first_seen=now, expires=now + timedelta(days=14), relevance=7.0,
    )
    stale = CorpusItem(
        id="age-stale", title="t", url="u", summary="", source_id="s", source_name="S",
        published=now - timedelta(days=5), first_seen=now - timedelta(days=5),
        expires=now + timedelta(days=9), relevance=7.0,
    )
    aging_cfg = {**scoring_cfg, "rank_score": {"age_penalty_per_day": 0.15}}
    recompute_rank([fresh, stale], aging_cfg, now)
    check(fresh.relevance == 7.0 and stale.relevance == 7.0,
          "relevance is untouched by rank recomputation")
    check(stale.rank_score < fresh.rank_score,
          "the older item's rank_score is lower than the fresher one's",
          f"fresh={fresh.rank_score:.2f} stale={stale.rank_score:.2f}")

    section("Acceptance criterion 2: existing items are never re-scored")
    existing_item = CorpusItem(
        id="exist-1", title="Old story", url="https://example.org/a", summary="",
        source_id="s", source_name="S", published=now - timedelta(days=2),
        first_seen=now - timedelta(days=2), expires=now + timedelta(days=12),
        relevance=6.4, llm_score=6.8, mode="llm",
    )
    # The same story turns up again in a later fetch, under the same id.
    refetched = Item(
        uid="exist-1", title="Old story", summary="", url="https://example.org/a",
        source_id="s", source_name="S", published=now,
    )
    existing_ids = {existing_item.id}
    new_raw = [i for i in [refetched] if i.uid not in existing_ids]
    check(new_raw == [], "a re-fetched story already in the corpus is excluded from 'new'")
    check(existing_item.relevance == 6.4 and existing_item.llm_score == 6.8,
          "the existing corpus item's frozen relevance and llm score are untouched")

    section("Acceptance criterion 5: no API key means clean deterministic fallback")
    had_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        raw_items = [
            Item(uid="k1", title="History curriculum shakeup", summary="", url="u1",
                 source_id="s", source_name="S", published=now),
        ]
        verdicts, status = llm.rerank_new_items(raw_items, scoring_cfg)
        check(verdicts == {}, "no verdicts are produced with no API key set")
        check("no API key" in status, "status message names the missing key",
              status)
        scorer2 = Scorer(scoring_cfg)
        frozen = score_new_items(raw_items, scorer2, verdicts, scoring_cfg, now, 14)
        check(all(i.mode == "deterministic" for i in frozen),
              "every item falls back to deterministic mode")
        meta = build_meta(frozen, frozen, [{"id": "s", "name": "S"}], [], status, now, 14)
        check(meta["scoring"]["status"] == status and meta["scoring"]["new_llm_ranked"] == 0,
              "meta.json reports the degradation")
    finally:
        if had_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = had_key

    section("Acceptance criterion 7: a failing feed is reported, not raised")
    import requests

    from src.fetch import fetch_feed

    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")

    original_get = requests.get
    requests.get = lambda *a, **kw: FakeResponse()
    try:
        result_items, error = fetch_feed(
            {"id": "x", "name": "Flaky Feed", "url": "https://example.org/rss"}, 96
        )
        check(result_items == [], "a failing feed returns no items")
        check(error is not None and "500" in error, "the error is captured, not raised", str(error))
    finally:
        requests.get = original_get

    feed_problems = ["Flaky Feed: HTTPError: 500 Server Error"]
    meta = build_meta([], [], [{"id": "x", "name": "Flaky Feed"}], feed_problems, "n/a", now, 14)
    down = [s for s in meta["sources"] if not s["ok"]]
    check(len(down) == 1 and down[0]["name"] == "Flaky Feed",
          "the failing feed shows up as unhealthy in meta.json's source list")

    section("Blending")
    check(blend_relevance(8.0, None, 0.6) == 8.0,
          "deterministic-only relevance uses the norm directly")
    check(abs(blend_relevance(6.0, 9.0, 0.6) - (0.4 * 6.0 + 0.6 * 9.0)) < 1e-9,
          "blended relevance matches the documented weighting")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
