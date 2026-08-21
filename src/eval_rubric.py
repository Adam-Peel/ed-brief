"""Standalone diagnostic: re-score every already-scored item in
data/corpus.json against the CURRENT rubric, and compare the fresh LLM
score to the one already frozen in the corpus.

Never touches data/corpus.json, docs/, or commits anything -- read-only,
manually triggered (see .github/workflows/eval-rubric.yml), not part of
the build pipeline. Answers "did the last rubric edit actually change
anything, and by how much" with a number instead of a guess: rerun this
after every future rubric change to see the delta on the same real
stories, rather than judging a diff by eye.

Usage:
    python -m src.eval_rubric
"""

from __future__ import annotations

import sys

from . import classify, llm
from .build import ROOT, load_config
from .corpus import load
from .fetch import Item


def main() -> int:
    _, cfg = load_config()
    corpus = load(ROOT / "data" / "corpus.json")

    scored_before = [c for c in corpus if c.llm_score is not None]
    if not scored_before:
        print(
            "No items in the corpus have a prior LLM score to compare "
            "against -- nothing to evaluate.",
            file=sys.stderr,
        )
        return 1

    if not llm.available():
        print(
            "ANTHROPIC_API_KEY not set (or the anthropic package isn't "
            "installed) -- nothing to compare against.",
            file=sys.stderr,
        )
        return 1

    items = [
        Item(
            uid=c.id,
            title=c.title,
            summary=c.summary,
            url=c.url,
            source_id=c.source_id,
            source_name=c.source_name,
            published=c.published,
        )
        for c in scored_before
    ]
    old_score_by_id = {c.id: c.llm_score for c in scored_before}
    title_by_id = {c.id: c.title for c in scored_before}

    # Classified fresh, not reused from the corpus's own stored type --
    # existing items may predate the two-stage pipeline entirely (defaulting
    # to OTHER via CorpusItem's backward-compat default), and the whole
    # point of this run is testing the CURRENT pipeline end to end, not just
    # stage 2 in isolation.
    print(f"Classifying {len(items)} already-scored items against the current pipeline...")
    classify_results, classify_status = classify.classify_items(items, cfg)
    print(f"{classify_status}\n")

    print(f"Re-scoring {len(items)} items against the current rubric...\n")
    verdicts, status = llm.rerank_new_items(items, classify_results, cfg)
    print(f"{status}\n")

    if not verdicts:
        print("No verdicts came back -- nothing to compare.", file=sys.stderr)
        return 1

    rows = [
        (new_score - old_score_by_id[item_id], old_score_by_id[item_id], new_score, item_id, why)
        for item_id, (new_score, why) in verdicts.items()
    ]
    rows.sort(key=lambda r: -abs(r[0]))

    # locality is a build-time FLOOR on rank_score (corpus.recompute_rank),
    # never summed into relevance -- so a locality-eligible item's "new"
    # score here is deliberately its PRE-floor value, and a drop is not
    # automatically a regression the way it would be for anything else.
    # Flagged explicitly rather than left to look like an unexplained
    # exception to every other row.
    print(f"{'delta':>6}  {'old':>5}  {'new':>5}  type          title")
    for delta, old_score, new_score, item_id, why in rows:
        title = title_by_id.get(item_id, item_id)[:60]
        item_type = classify_results.get(item_id, {}).get("type", "?")
        locality = classify_results.get(item_id, {}).get("locality", 0)
        local_flag = f" [locality={locality}, floored in production]" if locality >= 3 else ""
        print(f"{delta:+6.1f}  {old_score:5.1f}  {new_score:5.1f}  {item_type:<12}  {title}{local_flag}")
        if why:
            print(f"{'':>29}  now: {why}")

    deltas = [r[0] for r in rows]
    mean_delta = sum(deltas) / len(deltas)
    mean_abs_delta = sum(abs(d) for d in deltas) / len(deltas)
    moved_a_lot = sum(1 for d in deltas if abs(d) >= 2.0)
    missing = len(items) - len(rows)

    print(
        f"\n{len(rows)} compared"
        + (f" ({missing} not re-scored this run)" if missing else "")
        + f". mean delta {mean_delta:+.2f}, mean |delta| {mean_abs_delta:.2f}, "
        f"{moved_a_lot} moved >= 2.0 points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
