"""End-to-end offline build, then checks against its output: acceptance
criteria 6 (LLM partial-failure fallback), 9 (a quiet run never empties the
site), 10 (the read-only guarantee), 11 (API shape), and 12 (ReadStore is
defensively wrapped). No network access required. Run with:

    python tests/test_readonly.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build  # noqa: E402
from src.fetch import parse_feed  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""))
        failures.append(label)


def section(title: str) -> None:
    print(f"\n{title}")


def _offline_fetch_all(feeds, window_hours):
    """Stand-in for src.fetch.fetch_all: parses local fixtures instead of
    hitting the network, so the full build pipeline can run offline."""
    by_id = {f["id"]: f for f in feeds if f.get("enabled", True)}
    items, problems = [], []
    seen = set()
    for path in sorted(FIXTURES.glob("*.xml")):
        cfg = by_id.get(path.stem)
        if not cfg:
            continue
        parsed, error = parse_feed(path.read_bytes(), cfg, window_hours)
        if error:
            problems.append(f"{cfg['name']}: {error}")
            continue
        for item in parsed:
            if item.uid in seen:
                continue
            seen.add(item.uid)
            items.append(item)
    return items, problems


def _run_offline_build(tmp_root: Path) -> None:
    original_fetch_all = build.fetch_all
    build.fetch_all = _offline_fetch_all
    try:
        code = build.main(output_root=tmp_root)
    finally:
        build.fetch_all = original_fetch_all
    assert code == 0


class _FakeMessages:
    def __init__(self, texts, prefix_thinking_block=False):
        self._texts = iter(texts)
        # Real Sonnet-5 responses run adaptive thinking by default and put a
        # ThinkingBlock (type="thinking", no .text) ahead of the TextBlock --
        # see src/llm.py's _score_batch. Reproduced here on demand so this
        # fake actually exercises the same "find the text block" code path
        # instead of a shape the real API never returns.
        self._prefix_thinking_block = prefix_thinking_block

    def create(self, **kwargs):
        text = next(self._texts)
        content = []
        if self._prefix_thinking_block:
            content.append(types.SimpleNamespace(type="thinking"))
        content.append(types.SimpleNamespace(type="text", text=text))
        return types.SimpleNamespace(content=content, stop_reason="end_turn")


class _FakeAnthropicClient:
    def __init__(self, texts, *a, prefix_thinking_block=False, **kw):
        self.messages = _FakeMessages(texts, prefix_thinking_block=prefix_thinking_block)


def _with_fake_client(texts, run, *, prefix_thinking_block=False):
    """Monkeypatch anthropic.Anthropic for the duration of `run()`, feeding
    it canned response texts in order, then restore it -- shared by every
    two-stage pipeline test below so each one only has to state what
    responses it wants, not the setup/teardown around getting them there."""
    import os

    import anthropic

    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
    original_cls = anthropic.Anthropic
    try:
        anthropic.Anthropic = lambda *a, **kw: _FakeAnthropicClient(
            texts, prefix_thinking_block=prefix_thinking_block
        )
        try:
            return run()
        finally:
            anthropic.Anthropic = original_cls
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def main() -> int:
    import os

    from datetime import datetime, timedelta, timezone

    from src import classify as classify_mod
    from src import llm as llm_mod
    from src.corpus import CorpusItem, recompute_rank
    from src.fetch import Item

    now = datetime.now(timezone.utc)

    def item(uid, title="T"):
        return Item(uid=uid, title=title, summary="", url=f"u{uid}", source_id="s",
                    source_name="S", published=now)

    section("Acceptance criterion 6: whole-batch response failures fall back cleanly")
    # Two items, both classified OTHER (universals only, no type scale) so
    # the fake response shape doesn't need a type-specific field. Batch size
    # 1, so each gets its own API call: the first call returns unparseable
    # text, the second a truncated (incomplete) JSON body -- two different
    # ways a whole response can fail to parse. Per-item omission within an
    # otherwise-valid response is no longer a reachable failure mode now
    # that output_config's json_schema requires "verdicts" as an object
    # keyed by exactly the batch's own ids (required + additionalProperties
    # false) -- see the note on llm_score_batch_size in scoring.yml -- so
    # that scenario isn't tested here any more; a batch either satisfies
    # the schema or it doesn't.
    item_a, item_b = item("a1"), item("b2")
    classify_ab = {
        "a1": {"type": "OTHER", "locality": 0, "confidence": "high"},
        "b2": {"type": "OTHER", "locality": 0, "confidence": "high"},
    }
    cfg = {"llm_score_batch_size": 1}
    verdicts, status = _with_fake_client(
        ["not json at all", '{"verdicts": {"b2": {"u1": 4, "u2": 4'],
        lambda: llm_mod.rerank_new_items([item_a, item_b], classify_ab, cfg),
    )
    check(verdicts == {}, "neither item gets a verdict: both batches failed to parse",
          str(verdicts))
    check("degraded" in status or "failed" in status,
          "the status message reflects the degradation", status)

    section("Regression: a leading thinking block must not break text extraction")
    # Production bug (21 Aug 2026): Sonnet 5 runs adaptive thinking by
    # default, so response.content[0] is a ThinkingBlock with no .text --
    # indexing [0] instead of searching for type=="text" raised
    # AttributeError on every batch, silently dropping to deterministic
    # scoring. This reproduces that exact response shape.
    item_c = item("c3")
    classify_c = {"c3": {"type": "OTHER", "locality": 0, "confidence": "high"}}
    cfg = {"llm_score_batch_size": 1}
    verdicts, status = _with_fake_client(
        ['{"verdicts": {"c3": {"u1": 4, "u2": 4, "why": "directly relevant"}}}'],
        lambda: llm_mod.rerank_new_items([item_c], classify_c, cfg),
        prefix_thinking_block=True,
    )
    # universals = 0.6*4 + 0.4*4 = 4.0; relevance = 4.0 * 2.5 = 10.0
    check(verdicts.get("c3", (None, None))[0] == 10.0,
          "the verdict behind a leading thinking block is still extracted", str(verdicts))
    check("failed" not in status and "AttributeError" not in status,
          "no AttributeError surfaces in the status message", status)

    section("Two-stage pipeline: a classification error routes to OTHER, never discards")
    item_x = item("x1")
    results, status = _with_fake_client(
        ["not json at all"],
        lambda: classify_mod.classify_items([item_x], {"llm_classify_batch_size": 10}),
    )
    check(results.get("x1", {}).get("type") == "OTHER",
          "a failed classify batch routes its items to OTHER, not IRRELEVANT", str(results))
    check("x1" in results, "the item still gets an entry -- classify never silently drops one")

    section("Two-stage pipeline: an explicit IRRELEVANT verdict discards from stage 2")
    item_y = item("y1")
    classify_y = {"y1": {"type": "IRRELEVANT", "locality": 0, "confidence": "high"}}
    # No responses queued at all: if the item were sent to stage 2 despite
    # being IRRELEVANT, the fake client's iterator would raise
    # StopIteration, failing loudly rather than silently passing.
    verdicts, status = _with_fake_client(
        [], lambda: llm_mod.rerank_new_items([item_y], classify_y, {"llm_score_batch_size": 10}),
    )
    check(verdicts == {}, "an IRRELEVANT item gets no verdict -- never sent to stage 2", str(verdicts))
    check("irrelevant" in status.lower() and "discarded" in status.lower(),
          "the status names the discard", status)

    section("Two-stage pipeline: SECTOR/OTHER score on universals alone and still rank")
    item_z = item("z1")
    classify_z = {"z1": {"type": "SECTOR", "locality": 0, "confidence": "high"}}
    verdicts, status = _with_fake_client(
        ['{"verdicts": {"z1": {"u1": 4, "u2": 2, "why": "test"}}}'],
        lambda: llm_mod.rerank_new_items([item_z], classify_z, {"llm_score_batch_size": 10}),
    )
    # universals = 0.6*4 + 0.4*2 = 3.2; relevance = 3.2 * 2.5 = 8.0
    check(abs(verdicts.get("z1", (0.0, ""))[0] - 8.0) < 1e-9,
          "a SECTOR item's relevance comes purely from universals, matching the documented formula",
          str(verdicts))

    section("Two-stage pipeline: one type's scoring failure leaves other types fully scored")
    item_cur, item_hist = item("cur1"), item("hist1")
    classify_ch = {
        "cur1": {"type": "CURRICULUM", "locality": 0, "confidence": "high"},
        "hist1": {"type": "HISTORY", "locality": 0, "confidence": "high"},
    }
    verdicts, status = _with_fake_client(
        [
            "not json at all",  # CURRICULUM's one batch (grouped/sent first)
            '{"verdicts": {"hist1": {"u1": 4, "u2": 4, "h1": 4, "h2": 4, "h3": 4, "h4": 4, "why": "ok"}}}',  # HISTORY's
        ],
        lambda: llm_mod.rerank_new_items([item_cur, item_hist], classify_ch, {"llm_score_batch_size": 10}),
    )
    check("cur1" not in verdicts and "hist1" in verdicts,
          "CURRICULUM's malformed batch doesn't stop HISTORY from being fully scored", str(verdicts))

    section("Two-stage pipeline: truncated JSON in one batch degrades that batch only")
    item_p, item_q = item("p1"), item("q1")
    classify_pq = {
        "p1": {"type": "SECTOR", "locality": 0, "confidence": "high"},
        "q1": {"type": "SECTOR", "locality": 0, "confidence": "high"},
    }
    verdicts, status = _with_fake_client(
        [
            '{"verdicts": {"p1": {"u1": 4, "u2": 4',  # truncated, no closing bracket
            '{"verdicts": {"q1": {"u1": 2, "u2": 2, "why": "ok"}}}',
        ],
        lambda: llm_mod.rerank_new_items([item_p, item_q], classify_pq, {"llm_score_batch_size": 1}),
    )
    check("p1" not in verdicts and "q1" in verdicts,
          "the truncated batch's item falls back; the other batch is unaffected", str(verdicts))

    section("Two-stage pipeline: the stage-2 schema pins exact verdict coverage via object keys")
    # Omitted or hallucinated ids used to be defended against by per-item
    # validation after parsing; that's gone now that output_config's
    # json_schema requires "verdicts" as an OBJECT keyed by exactly the
    # batch's own ids (required + additionalProperties: false) -- see the
    # note on llm_score_batch_size in scoring.yml. NOT an array with
    # minItems/maxItems: the first version of this schema used that and
    # failed every single request in production, since Anthropic's
    # structured outputs only supports array minItems of 0 or 1, and
    # doesn't support maxItems or minimum/maximum at all. A fake test
    # client can't exercise real schema enforcement (it returns canned
    # text directly, bypassing the API), so what's checkable offline is
    # that the schema this code actually SENDS is built the corrected way,
    # and doesn't regress back to the keywords that broke production.
    schema = llm_mod._stage2_schema(["hist1", "hist2"], llm_mod.TYPE_SCORE_KEYS["HISTORY"])
    verdicts_schema = schema["properties"]["verdicts"]
    check(verdicts_schema["type"] == "object" and set(verdicts_schema["required"]) == {"hist1", "hist2"},
          "verdicts is an object requiring exactly the batch's own ids as keys")
    check(verdicts_schema["additionalProperties"] is False,
          "no key beyond the batch's own ids is allowed")
    check(set(verdicts_schema["properties"]["hist1"]["required"]) == {"u1", "u2", "h1", "h2", "h3", "h4", "why"},
          "HISTORY's per-item schema requires u1/u2, all four H-scale fields, and why")

    def _contains_key(obj, key) -> bool:
        if isinstance(obj, dict):
            return key in obj or any(_contains_key(v, key) for v in obj.values())
        if isinstance(obj, list):
            return any(_contains_key(v, key) for v in obj)
        return False

    unsupported = [k for k in ("minItems", "maxItems", "minimum", "maximum", "maxLength") if _contains_key(schema, k)]
    check(unsupported == [],
          "the schema uses none of the array/numeric-bound keywords that caused the "
          "2026-08-22 production BadRequestError incident", str(unsupported))

    classify_schema = classify_mod._classify_schema(["a1", "b2"])
    classify_verdicts_schema = classify_schema["properties"]["verdicts"]
    check(classify_verdicts_schema["type"] == "object" and set(classify_verdicts_schema["required"]) == {"a1", "b2"},
          "classify's schema is the same object-keyed-by-id shape as stage 2's")
    classify_unsupported = [
        k for k in ("minItems", "maxItems", "minimum", "maximum", "maxLength")
        if _contains_key(classify_schema, k)
    ]
    check(classify_unsupported == [], "classify's schema also avoids the unsupported keywords",
          str(classify_unsupported))

    section("Two-stage pipeline: the locality floor is a build-time percentile, not a fixed number")
    rank_items = [
        CorpusItem(id=f"loc{k}", title="t", url="u", summary="", source_id="s", source_name="S",
                   published=now, first_seen=now, expires=now + timedelta(days=14),
                   relevance=rel, locality=loc)
        for k, (rel, loc) in enumerate(
            [(9.0, 0), (8.0, 0), (7.0, 0), (6.0, 0), (1.0, 4), (0.5, 2)]
        )
    ]
    recompute_rank(rank_items, {"locality": {"floor_percentile": 85, "floor_min_score": 3},
                                 "tiers": {}, "rank_score": {}}, now)
    by_id = {i.id: i for i in rank_items}
    # 85th percentile of [0.5,1,6,7,8,9] (linear interpolation): k=0.85*5=4.25
    # -> between index 4 (8.0) and 5 (9.0) -> 8.0 + (9.0-8.0)*0.25 = 8.25
    check(abs(by_id["loc4"].rank_score - 8.25) < 1e-6,
          "an item with locality >= floor_min_score is floored to this build's 85th percentile",
          str(by_id["loc4"].rank_score))
    check(by_id["loc5"].rank_score == 0.5,
          "an item with locality below the threshold (2 < 3) is never floored, even as the lowest score",
          str(by_id["loc5"].rank_score))

    section("Golden test: a fixture set produces stable type/locality data on each card")
    # The list groups by TIER (score strength), same as it always has --
    # type moved to a per-card pill (see card() in site.py) after grouping
    # by type buried high-scoring items under whichever category happened
    # to render lower on the page. Pill rendering itself runs client-side
    # in JS -- there's no headless browser in this offline suite to execute
    # it, the same honest limit AC12 already notes for ReadStore. What IS
    # directly checkable: every item's {type, locality} survives into the
    # embedded ITEMS payload unchanged, and TYPE_LABELS (what the pill text
    # actually says) is present and correct in the rendered page. Fixed
    # input -> fixed output, checked by direct string comparison, is the
    # "golden" part.
    from src import site as site_mod

    golden_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    golden_items = [
        CorpusItem(id="g-cur", title="Curriculum story", url="uc", summary="", source_id="s",
                   source_name="S", published=golden_now, first_seen=golden_now,
                   expires=golden_now + timedelta(days=14), relevance=8.0, rank_score=8.0,
                   tier="lead", item_type="CURRICULUM", locality=0),
        CorpusItem(id="g-notts", title="Notts history story", url="un", summary="", source_id="s",
                   source_name="S", published=golden_now, first_seen=golden_now,
                   expires=golden_now + timedelta(days=14), relevance=6.0, rank_score=6.0,
                   tier="worth", item_type="HISTORY", locality=4),
        CorpusItem(id="g-sector", title="Sector story", url="us", summary="", source_id="s",
                   source_name="S", published=golden_now, first_seen=golden_now,
                   expires=golden_now + timedelta(days=14), relevance=3.0, rank_score=3.0,
                   tier="rest", item_type="SECTOR", locality=0),
        CorpusItem(id="g-other", title="Uncategorised story", url="uo", summary="", source_id="s",
                   source_name="S", published=golden_now, first_seen=golden_now,
                   expires=golden_now + timedelta(days=14), relevance=2.0, rank_score=2.0,
                   tier="rest", item_type="OTHER", locality=0),
    ]
    payload = site_mod._payload(golden_items)
    by_id_payload = {p["id"]: p for p in payload}
    check(
        [by_id_payload[i]["type"] for i in ["g-cur", "g-notts", "g-sector", "g-other"]]
        == ["CURRICULUM", "HISTORY", "SECTOR", "OTHER"],
        "every item's type survives into the payload unchanged and in order",
    )
    check(by_id_payload["g-notts"]["locality"] == 4,
          "locality survives into the payload unchanged")
    with tempfile.TemporaryDirectory() as tmp:
        golden_root = Path(tmp)
        meta_stub = {"scoring": {"status": "test"}, "retention_days": 14, "sources": []}
        site_mod.write_site(golden_items, [], meta_stub, golden_root, golden_now, {})
        golden_html = (golden_root / "docs" / "index.html").read_text("utf-8")
    check(
        '"CURRICULUM": "Curriculum"' in golden_html and '"OTHER": "Other"' in golden_html,
        "TYPE_LABELS (the pill text for every type, including OTHER) reaches the rendered page",
    )
    check('class="type-pill"' in golden_html and 'class="notts-pill"' in golden_html,
          "the card template emits both the type pill and the Notts pill")
    check('"type": "HISTORY", "locality": 4' in golden_html,
          "the Nottinghamshire-eligible item's type and locality both reach the rendered page")

    section("Full offline build")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        _run_offline_build(tmp_root)

        docs = tmp_root / "docs"
        api_dir = docs / "api" / "v1"
        check((docs / "index.html").exists(), "index.html was written")
        check((docs / "archive.html").exists(), "archive.html was written")
        check((tmp_root / "data" / "corpus.json").exists(), "corpus.json was written")
        check(api_dir.exists(), "docs/api/v1 was written")

        items_doc = json.loads((api_dir / "items.json").read_text("utf-8"))
        first_run_count = items_doc["count"]
        check(first_run_count > 0, "the first run produced live items", str(first_run_count))

        tags_html = (docs / "tags.html").read_text("utf-8") if (docs / "tags.html").exists() else ""
        check(bool(tags_html), "tags.html was written")

        section("RSS feed")
        feed_path = docs / "feed.xml"
        check(feed_path.exists(), "feed.xml was written")
        feed_text = feed_path.read_text("utf-8")
        try:
            root_el = ET.fromstring(feed_text)
            feed_parses = True
        except ET.ParseError as exc:
            root_el, feed_parses = None, False
            failures.append(f"feed.xml is not well-formed XML: {exc}")
        check(feed_parses, "feed.xml is well-formed XML")
        if feed_parses:
            channel = root_el.find("channel")
            check(root_el.tag == "rss" and channel is not None,
                  "feed.xml has an <rss><channel> structure")
            check(channel.find("title") is not None and channel.find("link") is not None,
                  "the channel has a title and link")
            feed_items = channel.findall("item")
            tiers_in_feed = {
                it.find("description").text or "" for it in feed_items
            }
            check(all(("Everything else" not in t and "Low relevance" not in t) for t in tiers_in_feed),
                  "the feed only includes lead/worth tier items, not the full corpus")
        check("sub-history" in tags_html and "<td>" in tags_html,
              "tags.html lists at least one known tag with a description")

        section("Acceptance criterion 9: a quiet second run doesn't empty the site")
        _run_offline_build(tmp_root)  # same fixtures again -> nothing new
        items_doc_2 = json.loads((api_dir / "items.json").read_text("utf-8"))
        index_html = (docs / "index.html").read_text("utf-8")
        check(items_doc_2["count"] == first_run_count,
              "a run finding zero new items still reports the full live corpus",
              f"before={first_run_count} after={items_doc_2['count']}")
        check(len(items_doc_2["items"]) > 0 and "ITEMS = []" not in index_html,
              "the rendered site still has a non-empty item list")

        section("Acceptance criterion 10: the read-only guarantee")
        banned_token_patterns = [
            re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
            re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
        ]
        external_fetch = re.compile(r"""fetch\(\s*["'](?!\.)(?!/)(?!data:)""")
        offenders: list[str] = []
        for path in docs.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except UnicodeDecodeError:
                continue
            rel = path.relative_to(docs)
            if "api.github.com" in text:
                offenders.append(f"{rel}: api.github.com")
            if external_fetch.search(text):
                offenders.append(f"{rel}: external fetch(")
            for pattern in banned_token_patterns:
                if pattern.search(text):
                    offenders.append(f"{rel}: token-shaped string")
        check(offenders == [], "no api.github.com, external fetch(, or token-shaped "
              "string anywhere under docs/", "; ".join(offenders))

        section("Acceptance criterion 11: every API file is valid JSON with the documented shape")
        all_json = sorted(api_dir.rglob("*.json"))
        check(len(all_json) > 0, "at least one API file was written")
        parsed_ok = True
        for path in all_json:
            try:
                json.loads(path.read_text("utf-8"))
            except json.JSONDecodeError:
                parsed_ok = False
                failures.append(f"{path.relative_to(api_dir)} is not valid JSON")
        check(parsed_ok, "every API file under docs/api/v1 parses as JSON")

        index_doc = json.loads((api_dir / "index.json").read_text("utf-8"))
        actual_endpoints = sorted(
            "/api/v1/" + p.relative_to(api_dir).as_posix()
            for p in all_json
            if p.name != "index.json"
        )
        check(
            sorted(index_doc["endpoints"]) == sorted(["/api/v1/index.json", *actual_endpoints]),
            "index.json's endpoint list matches every endpoint actually emitted",
        )

        meta_doc = json.loads((api_dir / "meta.json").read_text("utf-8"))
        check(all(k in meta_doc for k in ("counts", "scoring", "sources", "retention_days")),
              "meta.json has the documented top-level keys")
        tags_doc = json.loads((api_dir / "tags.json").read_text("utf-8"))
        check(all("tag" in t and "count" in t for t in tags_doc["tags"]),
              "tags.json rows carry tag and count")
        latest_doc = json.loads((api_dir / "latest.json").read_text("utf-8"))
        check("items" in latest_doc and "generated" in latest_doc,
              "latest.json has the documented shape")

        section("Acceptance criterion 12: ReadStore wraps localStorage access defensively")
        store_match = re.search(r"const ReadStore = \(\(\) => \{.*?\}\)\(\);", index_html, re.S)
        check(store_match is not None, "the ReadStore block is present in index.html")
        store_src = store_match.group(0) if store_match else ""
        check(store_src.count("try {") >= 2 and store_src.count("catch") >= 2,
              "every localStorage read and write is wrapped in try/catch",
              f"try count={store_src.count('try {')} catch count={store_src.count('catch')}")

    section("Regression: two runs on the same day accumulate, not overwrite")
    # With more than one scheduled run a day (e.g. 07:00 and 19:00), a second
    # run's day-scoped writes (the dated archive, the markdown brief) must
    # add to the first run's items, not replace them -- this is exactly the
    # bug that existed before today_items was introduced in build.py.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        all_stems = sorted(p.stem for p in FIXTURES.glob("*.xml"))
        half = len(all_stems) // 2
        first_stems, second_stems = set(all_stems[:half]), set(all_stems)

        def make_fetch(stems):
            def _fetch(feeds, window_hours):
                by_id = {f["id"]: f for f in feeds if f.get("enabled", True)}
                items, problems, seen = [], [], set()
                for path in sorted(FIXTURES.glob("*.xml")):
                    if path.stem not in stems:
                        continue
                    cfg = by_id.get(path.stem)
                    if not cfg:
                        continue
                    parsed, error = parse_feed(path.read_bytes(), cfg, window_hours)
                    if error:
                        problems.append(f"{cfg['name']}: {error}")
                        continue
                    for item in parsed:
                        if item.uid in seen:
                            continue
                        seen.add(item.uid)
                        items.append(item)
                return items, problems

            return _fetch

        original_fetch_all = build.fetch_all
        try:
            build.fetch_all = make_fetch(first_stems)
            assert build.main(output_root=tmp_root) == 0
            build.fetch_all = make_fetch(second_stems)  # dedup narrows this to the remaining stems
            assert build.main(output_root=tmp_root) == 0
        finally:
            build.fetch_all = original_fetch_all

        from datetime import datetime, timezone

        iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        api_dir = tmp_root / "docs" / "api" / "v1"
        archive_doc = json.loads((api_dir / "archive" / f"{iso}.json").read_text("utf-8"))
        items_doc = json.loads((api_dir / "items.json").read_text("utf-8"))
        brief_text = (tmp_root / "briefs" / f"{iso}.md").read_text("utf-8")

        check(
            archive_doc["count"] == items_doc["count"] == len(archive_doc["items"]),
            "today's archive file covers items from both runs, not just the second",
            f"archive={archive_doc['count']} live={items_doc['count']}",
        )
        first_run_titles = [i["title"] for i in archive_doc["items"]][: len(first_stems)]
        check(
            any(t in brief_text for t in first_run_titles),
            "the markdown brief still contains a first-run item after the second run wrote the file",
        )

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
