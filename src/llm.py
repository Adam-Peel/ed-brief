"""LLM ranking pass, run once per item during ingestion, never at page-view
time.

Dormant unless ANTHROPIC_API_KEY is set in the environment. Every failure path
-- no key, missing package, client construction, a network/API error,
malformed JSON, or a response that omits some of the requested ids -- falls
back to deterministic scoring for exactly the affected items, never the whole
run. The build must never fail because of a billing or API problem.

To switch it on:
  1. Add ANTHROPIC_API_KEY as a repository secret (Settings → Secrets and
     variables → Actions → New repository secret).
  2. Uncomment the `ANTHROPIC_API_KEY` line in .github/workflows/brief.yml.
"""

from __future__ import annotations

import json
import os
import re

from .fetch import Item

DEFAULT_MODEL = "claude-sonnet-5"

# Anchors are load-bearing: the model is asked to score each story against a
# fixed rubric, never against the other stories in the batch. Batch-relative
# ranking is exactly the bug the fixed-ceiling deterministic normalisation
# (score.py) also exists to avoid -- see BUILD-SPEC.md's "scoring must be
# absolute" section.
RUBRIC = """You are helping a career-changer in England who is moving into \
secondary HISTORY teaching. They are at the point of entering initial teacher \
training, so they care most about:

- anything specific to history as a school subject (curriculum, GCSE, pedagogy)
- new historical research or scholarship that could inform how they teach or \
sequence their subject content -- not just teaching methodology, but new \
findings, interpretations, or discoveries about periods they might actually \
teach. This has real standing value for a history teacher's own subject \
knowledge, not just narrow classroom-practice tips.
- routes into teaching: ITT, PGCE, bursaries, QTS, recruitment
- the early career framework, induction, mentoring, workload, retention
- national curriculum and assessment reform, including the Francis Review
- Ofsted inspection, since it shapes what they will walk into
- evidence and cognitive science that changes classroom practice
- Nottinghamshire local HISTORY, heritage, or archaeology specifically --
they have a deliberate regional interest in their own area's history. This
is a genuine premium, not a generic "local news" bump: a story that is
ITSELF about Nottinghamshire's history, heritage, or archaeology should
score noticeably higher than an equivalent story about somewhere else, even
though neither is curriculum news. It does not need to connect to teaching
at all to score well here. Be strict about what counts, though: a story
about a modern event that merely takes place AT a historic Nottinghamshire
site -- a cinema night, a concert, a market -- is not a history story and
gets no premium at all, even though it will likely mention the site by
name. The test is what the story is ABOUT, not which building it namedrops.

They care much less about higher education, early years, further education \
colleges, and school sport. They teach in ENGLAND, so Scottish and Welsh \
policy is background only.

The secondary history curriculum in England runs from roughly 1066 (the \
Norman Conquest) to the present day. A story about history set before 1066 \
-- ancient civilisations, prehistory, the classical world -- is very \
unlikely to connect to what they will actually teach, however well-written \
the history journalism is: score these low (0-2) UNLESS the story is \
explicitly framed around teaching, curriculum, or a documented KS3/GCSE/ \
A-level topic that reaches back that far (e.g. an exam board's ancient- \
history option). The immediate pre-Conquest period (Anglo-Saxon and Viking \
England) is a genuine judgement call, not an automatic low score -- some \
GCSE options cover it as direct context for 1066 (e.g. "Anglo-Saxon and \
Norman England, c1060-88"); use your judgement on whether a given story sits \
close enough to that boundary to matter. The same story set after 1066 \
should be judged purely on its own merits, with no date penalty at all.

Score each story on its own merits against this fixed rubric. Do not rank or \
compare stories against each other -- judge each one in isolation:

  0  = irrelevant to this reader
  5  = useful to any teacher generally
  8  = directly relevant to secondary history ITT
  10 = drop everything, must read today

For each story below, return a relevance score from 0 to 10 on that scale, \
and a single short clause (under 15 words, no full stop) saying why it \
matters to them specifically. If a story is irrelevant, score it low and say \
so plainly.

Return ONLY a JSON array, no prose, no code fence:
[{{"id": "...", "score": 0-10, "why": "..."}}]

Stories:
{payload}"""


def available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _extract_json(text: str) -> list[dict]:
    """Pull the JSON array out of a response that may be wrapped in prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    return json.loads(text[start : end + 1])


def _score_batch(batch: list[Item], client, model: str) -> dict[str, tuple[float, str]]:
    """One API call for up to llm_batch_size items.

    Raises on total failure (network error, malformed JSON) -- the caller
    treats that as "none of this batch got a verdict". A clean return with
    some ids missing means the response parsed fine but omitted them, so only
    those specific items fall back; everything else in the batch still used
    the LLM.
    """
    payload = json.dumps(
        [
            {
                "id": item.uid,
                "title": item.title,
                "summary": item.summary[:320],
                "source": item.source_name,
            }
            for item in batch
        ],
        ensure_ascii=False,
        indent=1,
    )
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": RUBRIC.format(payload=payload)}],
    )
    verdicts = _extract_json(response.content[0].text)
    valid_ids = {item.uid for item in batch}

    results: dict[str, tuple[float, str]] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        item_id = verdict.get("id")
        # Only accept ids that were actually in this batch -- a hallucinated
        # or stale id from the model should be ignored, not stored.
        if not item_id or item_id not in valid_ids:
            continue
        try:
            score = max(0.0, min(10.0, float(verdict.get("score", 0))))
        except (TypeError, ValueError):
            continue
        why = str(verdict.get("why", "")).strip()[:160]
        results[item_id] = (score, why)
    return results


def rerank_new_items(
    items: list[Item], cfg: dict
) -> tuple[dict[str, tuple[float, str]], str]:
    """Score every new item against the absolute rubric, batched.

    Returns (verdicts_by_id, status). Only ids present in verdicts_by_id got
    an LLM score; the caller falls back to deterministic scoring for
    everything else, item by item -- never as an all-or-nothing decision for
    the whole run.
    """
    if not items:
        return {}, "no new items"
    if not available():
        return {}, "deterministic only (no API key set)"

    import anthropic

    # BRIEF_LLM_MODEL overrides scoring.yml's llm_model for quick local
    # testing, matching the same env-var-overrides-config pattern as
    # BRIEF_WINDOW_HOURS in build.py.
    model = os.environ.get("BRIEF_LLM_MODEL") or cfg.get("llm_model", DEFAULT_MODEL)

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 -- must not break the run
        return {}, f"deterministic only (client init failed: {type(exc).__name__})"

    batch_size = max(1, int(cfg.get("llm_batch_size", 45)))
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    verdicts: dict[str, tuple[float, str]] = {}
    errors: list[str] = []
    for n, batch in enumerate(batches, start=1):
        try:
            verdicts.update(_score_batch(batch, client, model))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad
            errors.append(f"batch {n}/{len(batches)}: {type(exc).__name__}")

    if not verdicts:
        detail = "; ".join(errors) or "no verdicts returned"
        return {}, f"deterministic only (LLM pass failed: {detail})"

    missing = len(items) - len(verdicts)
    if errors or missing:
        detail = "; ".join(errors) if errors else f"{missing} item(s) omitted from responses"
        return verdicts, (
            f"LLM re-ranked with {model} ({len(verdicts)}/{len(items)}; degraded: {detail})"
        )
    return verdicts, f"LLM re-ranked with {model}"
