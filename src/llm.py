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
RUBRIC = """You are helping a career-changer in England moving into secondary \
HISTORY teaching. They are currently {reader_stage} -- that stage is a \
config value (scoring.yml), not fixed, because their priorities shift as it \
changes.

Score against this fixed rubric only. Do not rank or compare stories \
against each other -- judge each one in isolation, on its own merits.

WHAT THEY CARE ABOUT, roughly in descending order:

- History as a school subject: curriculum, GCSE and A-level specs, exam \
board choices, spec changes, option withdrawals, entry numbers -- Ofqual \
and the awarding bodies (AQA, Edexcel, OCR, WJEC) matter here as much as \
Ofsted does.
- New historical research or scholarship that could inform how they teach \
or sequence their subject content -- not just teaching methodology, but \
new findings, interpretations, or discoveries about periods they might \
actually teach. This has real standing value for a history teacher's own \
subject knowledge, not just narrow classroom-practice tips. The Historical \
Association and Teaching History are exactly this reader's trade press.
- Contested history and curriculum politics as they play out IN schools: \
empire and decolonisation in the curriculum, Holocaust education, \
political-impartiality guidance, textbook and exam-board controversies. \
Don't file this as generic culture-war news -- if it's about what gets \
taught, or how, it's a history-teaching story.
- Routes into teaching and the profession's shape: ITT, PGCE, bursaries, \
QTS, recruitment, national curriculum and assessment reform including the \
Francis Review, and -- once past this stage -- the ITTECF (the Initial \
Teacher Training and Early Career Framework, which merged the old ITT Core \
Content Framework and Early Career Framework from September 2025), \
induction, mentoring.
- Classroom practice broadly, not just "cognitive science" narrowly: \
retrieval and spaced practice, cognitive load and working memory, but \
equally behaviour management, questioning, modelling, adaptive teaching \
and differentiation, oracy, disciplinary literacy, formative assessment \
and marking. Score a solid piece on any of these as you would a \
cognitive-science piece -- it's the same category of story.
- SEND and inclusion: EHCPs, the SEND reform agenda, disadvantage and the \
attainment gap. Daily classroom reality, not a niche interest.
- AI in schools: generative AI in the classroom or in marking, academic \
integrity and plagiarism, edtech. Sharp for history specifically, since \
source analysis and independent coursework are exactly where AI misuse \
shows up.
- Safeguarding and statutory duties: KCSIE updates (revised most \
Septembers), attendance and behaviour guidance with legal force.
- Workload, retention, pay, pensions, the STRB, and industrial action -- \
material to someone actually deciding whether to enter or stay in the \
profession, not just abstract sector news.
- Ofsted inspection, since it shapes what they will walk into.
- Pupil wellbeing and mental health as shaped by school policy or \
research -- see the separate paragraph below; it needs its own scoring \
guidance.
- Nottinghamshire local HISTORY, heritage, or archaeology specifically -- \
also see the separate paragraph below; a real premium, but capped lower \
than the categories above.

They care much less about higher education, early years, further \
education colleges, and school sport. They teach in ENGLAND, so Scottish \
and Welsh policy is background only.

PUPIL WELLBEING AS A CATEGORY, not a single story: any research or \
reporting on how an exam, assessment, curriculum design, or school policy \
affects pupils' mental health, wellbeing, motivation, behaviour, or \
academic outcomes belongs here, whatever the specific angle -- resits, \
homework load, marking practices, attendance policy, screening checks, \
uniform rules, whatever. Treat these as one type of story and score the \
type consistently, not just whichever specific angle is in front of you. \
This is real classroom-relevant evidence, but it is not automatically \
top-tier: score it 5-7, higher toward 7 when it's squarely about \
secondary-age pupils' academic experience (e.g. exam-linked stress), \
lower toward 5 when it's more general wellbeing news not tied to a \
specific school-age policy.

NOTTINGHAMSHIRE LOCAL HISTORY PREMIUM: a genuine regional interest, not a \
generic "local news" bump. A story ITSELF about Nottinghamshire's history, \
heritage, or archaeology scores noticeably higher than an equivalent story \
about somewhere else, even though it isn't curriculum news and doesn't \
need to connect to teaching at all -- but cap it at 6: high enough to \
reliably surface, never higher than genuine curriculum, ITT, or career \
news. Be strict about what counts: a modern event that merely takes place \
AT a historic Nottinghamshire site -- a cinema night, a concert, a market \
-- is not a history story and gets no premium at all, however prominently \
it names the site. The test is what the story is ABOUT, not which \
building it namedrops. This premium is exempt from the pre-1066 guidance \
below -- Nottinghamshire's prehistoric and Roman sites are exactly what a \
local history study can legitimately cover.

DIRECT vs INCIDENTAL relevance: Ofsted and the DfE cover far more than \
secondary schools -- Ofsted also regulates children's social care \
(children's homes, fostering, adoption) and early years; the DfE covers \
early years and higher education too. A press release about Ofsted \
prosecuting an illegal children's home, or Ofsted's annual report as a \
general administrative document, is not a school-inspection story just \
because "Ofsted" is in the headline -- score it as you would any other \
social-care story, near the bottom, not as if it touched the inspection \
framework this reader will actually walk into. The test throughout this \
whole rubric is what a story is ABOUT, not which name it namedrops.

PERIOD: the secondary history curriculum in England runs roughly 1066 to \
the present, but this is not a hard cutoff. The KS3 programme of study \
explicitly includes a local-history study (which can predate 1066) and a \
thematic study "that consolidates and extends pupils' chronological \
knowledge from before 1066"; several GCSE thematic papers run from c1000 \
or earlier; OCR offers Ancient History at GCSE and A-level; and KS2 \
Romans/Anglo-Saxons/Vikings is the prior knowledge a Year 7 teacher builds \
on. So: the further a story sits from what a KS3-GCSE class actually \
studies, the lower it scores -- but don't apply a flat pre-1066 penalty. \
Score pre-1066 content on its own merits, same as any other period, when \
it is local (see the Nottinghamshire premium above), connects to a named \
thematic study or exam-board option, or is explicitly framed around \
teaching or curriculum. Reserve a low score (0-2) for what this guidance \
actually targets: disconnected ancient-world or prehistory journalism with \
no link to England, no local angle, and no curriculum framing -- \
interesting history writing, but not this reader's subject.

The `source` field in each story below is for your own disambiguation \
only (e.g. telling a specialist history publication apart from a general \
news outlet) -- do not weight a story up or down for which source it came \
from. Source reputation is already handled separately, deterministically, \
elsewhere in this pipeline; factor it in here too and it gets \
double-counted.

Score each story 0-10 against these anchors:

  0  = irrelevant to this reader
  2  = distantly related to education but outside this reader's remit \
(e.g. early years, HE, another subject)
  5  = useful to any teacher generally, not specific to this reader
  6  = solid interest to this reader specifically, or exceptional history \
  content
  8  = directly relevant to secondary history, ITT, or this reader's own \
subject knowledge/career
  10 = essential reading for THIS reader specifically -- not just strong \
journalism generally

This is a rolling 14-day digest, not a same-day alert -- score for lasting \
relevance to this reader, not literal urgency.

For each story below, return a relevance score from 0 to 10, and a single \
short clause (under 25 words, no full stop) saying why it matters to them \
specifically -- or, for a low score, why it doesn't.

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


def _score_batch(
    batch: list[Item], client, model: str, reader_stage: str
) -> dict[str, tuple[float, str]]:
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
        # Generous, not tuned to typical usage: adaptive thinking (on by
        # default -- see the text_block search below) spends part of this
        # budget before a single verdict token is written, so a tight cap
        # risks truncating the JSON array mid-batch. A truncated array has
        # no closing "]", _extract_json raises, and the WHOLE batch (up to
        # llm_batch_size items) falls back to deterministic -- the exact
        # all-or-nothing failure this function's docstring says it avoids.
        # Raising the ceiling doesn't cost anything unless it's actually
        # used: billing is by tokens generated, not by max_tokens itself.
        max_tokens=12000,
        messages=[{
            "role": "user",
            "content": RUBRIC.format(payload=payload, reader_stage=reader_stage),
        }],
    )
    # NOT response.content[0] -- current-generation models (Sonnet 5 among
    # them) run adaptive thinking by default even with no `thinking` param
    # set at all, which puts a thinking block ahead of the text block, so
    # content[0] is a ThinkingBlock with no .text attribute. Explicitly
    # disabling thinking is a documented pitfall on this model family
    # (occasional tool-call-in-visible-text / thinking-tag leakage), so this
    # finds the actual text block instead of fighting the default off.
    text_block = next((b for b in response.content if getattr(b, "type", None) == "text"), None)
    if text_block is None:
        raise ValueError(f"no text block in response (stop_reason={response.stop_reason!r})")
    verdicts = _extract_json(text_block.text)
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
    reader_stage = cfg.get("llm_reader_stage", "at the point of entering initial teacher training")

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
            verdicts.update(_score_batch(batch, client, model, reader_stage))
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
