"""Stage 1 of the two-stage LLM pipeline: triage every new item into a TYPE
before stage 2 (src/llm.py) scores it. Runs once per item during ingestion,
never at page-view time -- same dormant-unless-ANTHROPIC_API_KEY-is-set
contract as llm.py, since it uses the same client.

Discarding is deliberately conservative: only an explicit, high-confidence
IRRELEVANT verdict discards an item (by excluding it from what stage 2 scores
-- see rerank_new_items in llm.py). Every other failure mode -- a failed
batch, a malformed response, an unknown type string, low confidence -- routes
to OTHER, which still gets scored. An article is never dropped because of an
error; only because the model was confident it doesn't belong here.
"""

from __future__ import annotations

import json

from . import llm
from .fetch import Item

# NEW — add to Tag Index (per rule 5.2 of the filing convention, same
# convention already used for the deterministic tags in scoring.yml's
# `topics:` list -- see the comment there). These eight type codes and the
# locality concept are a genuinely different kind of vocabulary from that
# list (a single classification per item from stage 1, not multiple
# keyword-matched tags), so they're not folded into the `topics:` tags
# themselves -- but they're new, owner-facing vocabulary all the same, and
# claude/filing-convention.md lives outside this repo, so this can only be
# marked pending here, not actually added.
VALID_TYPES = {
    "CURRICULUM",
    "HISTORY",
    "PEDAGOGY",
    "PUPILS",
    "CAREER",
    "SECTOR",
    "OTHER",
    "IRRELEVANT",
}

# Anchors are load-bearing: the model classifies each story against a fixed
# rubric, never relative to the other stories in the batch -- same absolute-
# not-batch-relative requirement as the scoring rubric in llm.py, and for
# the same reason (BUILD-SPEC.md's "scoring must be absolute" section).
RUBRIC = """You are triaging education and history news for one specific \
reader: a career-changer in England entering secondary HISTORY teaching. \
They are currently {reader_stage}.

For each article below, assign exactly ONE type, state your confidence, and \
score how far the story is about Nottinghamshire.

TYPES

CURRICULUM   What is taught and how it is examined. National curriculum, \
GCSE and A-level specifications, exam board options and their \
withdrawal, qualifications reform, curriculum reviews.
HISTORY      Historical scholarship and material. New research, discoveries, \
reinterpretations, books, archives, collections, digitisations, \
heritage, exhibitions, anniversary material - anything that is \
about the past itself or makes the past usable.
PEDAGOGY     How to teach. Classroom practice, learning evidence, cognitive \
science, behaviour management, assessment and feedback practice, \
professional conduct in the classroom.
PUPILS       Who they will teach. Adolescent development, pupil wellbeing and \
mental health, motivation, behaviour, attendance, disadvantage \
and the attainment gap - what the 11-18s are like and what school \
does to them.
CAREER       Routes in and working life. ITT, PGCE and SCITT, bursaries, QTS, \
provider accreditation, recruitment and teacher supply, workload, \
pay, pensions, retention, induction and mentoring, school \
inspection, accountability measures and statutory duties.
SECTOR       General education news they skim but do not act on, including all \
exam results and attainment statistics.
OTHER        Plainly relevant to a secondary teacher in England but fits none \
of the above - or you are not confident which type applies.
IRRELEVANT   Higher education, further education and apprenticeships, early \
years, obituaries, human interest, departmental administration, \
and Ofsted's children's social care work. These are discarded and \
never scored.

TIE-BREAKS - this is where classification actually goes wrong

- Exam results and attainment statistics are SECTOR, never CURRICULUM, however \
prominent the coverage.
- A study of how pupils EXPERIENCE a policy is PUPILS. The policy change \
itself is CURRICULUM or CAREER.
- Regulatory enforcement against a provider or exam board is SECTOR.
- Inspection, funding and governance are CAREER when they change what a \
teacher is accountable for or whether jobs exist; SECTOR otherwise.
- Approaches to teaching a historical topic are PEDAGOGY, not HISTORY.
- Ofsted also regulates children's homes, fostering and early years, and the \
DfE covers early years and higher education. Ofsted or DfE appearing in a \
headline does not make a story about secondary schools. Judge what the story \
is ABOUT, not whose name is in it.
- Cross-subject curriculum change is still CURRICULUM even when history is \
never mentioned, because measures like the EBacc and Progress 8 determine \
history's timetable share and uptake.
- A staff profile, a personal travelogue, or a conference listing is not \
scholarship. If nothing about the past is actually conveyed, it is \
IRRELEVANT.

CONFIDENCE
  high - the type is clear from what the story is about
  low  - it could plausibly sit in two or more types, or you cannot tell from \
the title and summary given

LOCALITY, scored 0-4 for every article:
To what extent is this story ITSELF about Nottinghamshire - its history, \
heritage or archaeology, or a Nottinghamshire school, multi-academy trust, ITT \
provider, or the local teaching job market?
  4  the county is the subject of the story
  3  substantially about the county alongside other material
  2  one of several places or institutions covered
  1  a passing mention only
  0  not about the county
A modern event merely HELD AT a historic Nottinghamshire site - a concert, a \
market, a screening, a wedding fair - scores 0 however prominently the site is \
named. The test is what the story is about, not which building it names. A \
story merely PUBLISHED BY a Nottinghamshire organisation scores 0 unless the \
county is its subject.

Return ONLY a JSON array, no prose, no code fence:
[{{"id": "...", "type": "...", "confidence": "high|low", "locality": 0-4}}]

Articles:
{payload}"""


def _classify_batch(batch: list[Item], client, model: str, reader_stage: str, max_tokens: int) -> dict[str, dict]:
    """One API call for up to llm_classify_batch_size items. Raises on total
    failure (network error, malformed JSON) -- the caller routes every item
    in the batch to OTHER, never drops them. A clean return with some ids
    missing means those specific items get OTHER too, via the same
    caller-side default."""
    payload = json.dumps(
        [
            {"id": item.uid, "title": item.title, "summary": item.summary[:500]}
            for item in batch
        ],
        ensure_ascii=False,
        indent=1,
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": RUBRIC.format(payload=payload, reader_stage=reader_stage),
        }],
    )
    text_block = llm._extract_text_block(response)
    verdicts = llm._extract_json(text_block.text)
    valid_ids = {item.uid for item in batch}

    results: dict[str, dict] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        item_id = verdict.get("id")
        if not item_id or item_id not in valid_ids:
            continue
        try:
            locality = max(0, min(4, int(float(verdict.get("locality", 0)))))
        except (TypeError, ValueError):
            locality = 0
        item_type = verdict.get("type")
        confidence = verdict.get("confidence")
        # Unknown type, missing type, or low confidence -> OTHER. A low-
        # confidence IRRELEVANT is exactly the case this guards: the model
        # wasn't sure, so the conservative outcome is "keep it, score it",
        # not "discard it".
        if item_type not in VALID_TYPES or confidence != "high":
            item_type = "OTHER"
        results[item_id] = {"type": item_type, "locality": locality, "confidence": confidence or "low"}
    return results


def classify_items(items: list[Item], cfg: dict) -> tuple[dict[str, dict], str]:
    """Classify every new item, batched. Returns (results_by_id, status) in
    the same shape as llm.rerank_new_items, so build.py's degradation
    handling doesn't need to know classify and score are different calls.

    Every item that goes in gets an entry out, even under total failure --
    unlike stage 2 (where a missing verdict just means "fall back to
    deterministic for this one item"), stage 1's output is an input to
    stage 2's own grouping, so every item needs *some* type to be grouped
    by. A failed batch defaults every item in it to OTHER rather than
    omitting them, which is the one place this deliberately differs from
    the "omit on failure" pattern the rest of this pipeline uses.
    """
    if not items:
        return {}, "no new items"
    if not llm.available():
        return {}, "deterministic only (no API key set)"

    import anthropic

    model = llm._resolve_model(cfg)
    reader_stage = cfg.get("llm_reader_stage", "at the point of entering initial teacher training")
    max_tokens = int(cfg.get("llm_max_tokens", 12000))

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 -- must not break the run
        return {}, f"deterministic only (client init failed: {type(exc).__name__})"

    batch_size = max(1, int(cfg.get("llm_classify_batch_size", 40)))
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    results: dict[str, dict] = {}
    errors: list[str] = []
    for n, batch in enumerate(batches, start=1):
        try:
            batch_results = _classify_batch(batch, client, model, reader_stage, max_tokens)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad
            errors.append(f"batch {n}/{len(batches)}: {type(exc).__name__}")
            batch_results = {}
        for item in batch:
            results[item.uid] = batch_results.get(
                item.uid, {"type": "OTHER", "locality": 0, "confidence": "low"}
            )

    irrelevant = sum(1 for r in results.values() if r["type"] == "IRRELEVANT")
    if errors:
        detail = "; ".join(errors)
        return results, (
            f"classified {len(results)}/{len(items)} with {model} "
            f"({irrelevant} irrelevant, discarded; degraded: {detail})"
        )
    return results, f"classified with {model} ({irrelevant}/{len(items)} irrelevant, discarded)"
