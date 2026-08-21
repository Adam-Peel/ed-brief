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


def main() -> int:
    section("Acceptance criterion 6: LLM malformed JSON and omitted ids fall back cleanly")
    import os

    from src import llm as llm_mod

    fixture_items, cfg = None, None
    with tempfile.TemporaryDirectory() as tmp:
        # Two items, batch size 1, so each gets its own API call: the first
        # call returns unparseable text (total batch failure), the second
        # returns valid JSON that omits the requested id (partial failure).
        from src.fetch import Item
        from datetime import datetime, timezone

        item_a = Item(uid="a1", title="A", summary="", url="ua", source_id="s", source_name="S",
                      published=datetime.now(timezone.utc))
        item_b = Item(uid="b2", title="B", summary="", url="ub", source_id="s", source_name="S",
                      published=datetime.now(timezone.utc))
        cfg = {"llm_batch_size": 1}

        os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
        try:
            import anthropic

            original_cls = anthropic.Anthropic
            anthropic.Anthropic = lambda *a, **kw: _FakeAnthropicClient(
                ["not json at all", '[{"id": "zzz-not-requested", "score": 7, "why": "ok"}]']
            )
            try:
                verdicts, status = llm_mod.rerank_new_items([item_a, item_b], cfg)
            finally:
                anthropic.Anthropic = original_cls
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

        check(verdicts == {}, "neither item gets a verdict: one batch was malformed, "
              "the other omitted the requested id", str(verdicts))
        check("degraded" in status or "failed" in status,
              "the status message reflects the degradation", status)

    section("Regression: a leading thinking block must not break text extraction")
    with tempfile.TemporaryDirectory() as tmp:
        # Production bug (21 Aug 2026): Sonnet 5 runs adaptive thinking by
        # default, so response.content[0] is a ThinkingBlock with no .text --
        # indexing [0] instead of searching for type=="text" raised
        # AttributeError on every batch, silently dropping to deterministic
        # scoring. This reproduces that exact response shape.
        from src.fetch import Item
        from datetime import datetime, timezone

        item_c = Item(uid="c3", title="C", summary="", url="uc", source_id="s", source_name="S",
                      published=datetime.now(timezone.utc))
        cfg = {"llm_batch_size": 1}

        os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
        try:
            import anthropic

            original_cls = anthropic.Anthropic
            anthropic.Anthropic = lambda *a, **kw: _FakeAnthropicClient(
                ['[{"id": "c3", "subject": 4, "substance": 4, "actionability": 2, '
                 '"locality": 0, "why": "directly relevant"}]'],
                prefix_thinking_block=True,
            )
            try:
                verdicts, status = llm_mod.rerank_new_items([item_c], cfg)
            finally:
                anthropic.Anthropic = original_cls
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

        check(verdicts.get("c3", (None, None))[0] == 8.0,
              "the verdict behind a leading thinking block is still extracted", str(verdicts))
        check("failed" not in status and "AttributeError" not in status,
              "no AttributeError surfaces in the status message", status)

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
