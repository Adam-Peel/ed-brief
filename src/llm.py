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
HISTORY teaching. They are currently {reader_stage}.

Score against this fixed rubric only. Do not rank or compare stories \
against each other -- judge each one in isolation, on its own merits.

Score each story on FOUR SEPARATE dimensions, then the final score is \
their sum -- score each dimension independently. A story that feels \
generally important should not inflate SUBJECT just because it matters; \
that belongs in SUBSTANCE instead. This split exists specifically so one \
strong dimension (e.g. "this affects lots of pupils") can't single-\
handedly carry a story that's weak on the others (e.g. not actually about \
history, or not new information).

SUBJECT (0-4): how directly is this about history as a school subject, or \
this reader's own training and career, as opposed to secondary education \
generally?
  4 = history curriculum, GCSE/A-level specs and exam boards (AQA, \
Edexcel, OCR, WJEC), new historical scholarship or primary-source \
releases (The National Archives' periodic openings, digitisation, newly \
available collections count as much as academic papers) on a period they \
might teach, contested history and curriculum politics (empire and \
decolonisation in the curriculum, Holocaust education, political-\
impartiality guidance -- not generic culture-war news, but what gets \
taught and how), a history-specific teaching resource or museum/heritage \
schools programme, and history-specific ITT/CPD -- including a source \
that IS a history-teacher-training blog or department, even when an \
individual post's own wording reads as generic practice advice (see the \
source note below). Also 4: national curriculum/qualifications reform \
and accountability measures (the Francis Review, EBacc, Progress 8, KS3/ \
GCSE content and structure) that shape which subjects get timetable and \
option-block space, ITT providers themselves (accreditation, inspection, \
placement supply), and the structural shape of the secondary job market \
they're entering (falling rolls, a contracting market for new posts) -- \
these all determine whether and how much history gets taught, or whether \
this reader has a career to enter, even when they never name history \
specifically. Global/non-British history (Empire, the Cold War, Russia, \
China, apartheid) scores here too, at full value -- this is history \
taught IN England, not only history OF England.
  3 = real secondary-teaching professional substance that isn't subject- \
specific: SEND and inclusion (the substance of classroom inclusion, not \
just the word "SEND" appearing in a council notice or a community event \
badged SEND-friendly), Ofsted inspection, safeguarding and statutory \
duties (KCSIE, attendance/behaviour guidance with legal force), AI in \
schools, workload/pay/pensions/industrial action, school funding and MAT/ \
academisation stories that describe an actual staffing or curriculum \
consequence (a routine administrative or financial-oversight bulletin \
that merely mentions a trust or funding is NOT this -- see the 1 band \
below), classroom practice broadly (retrieval practice and cognitive \
load, but equally behaviour management, questioning, modelling, adaptive \
teaching, oracy, disciplinary literacy, formative assessment), and the \
induction/early-career phase of the ITTECF (relevant background, since \
they haven't reached it yet).
  2 = a story about a pupil-affecting policy (resit rules, uniform cost, \
attendance enforcement, exam-day logistics) that describes a real effect \
on pupils generally, without touching what or how much gets taught -- \
this reader teaches those pupils, so it's not irrelevant, but it isn't \
about their subject or their specific training either.
  1 = education-adjacent but weak ties to secondary teaching: higher \
education, early years, further education, school sport; a routine \
administrative/financial-oversight bulletin that only namedrops a trust \
or funding; Ofsted/DfE stories that are actually about children's social \
care or early years, not school inspection (these bodies cover far more \
than secondary schools, and a story is not about the part of their work \
that touches this reader just because their name is in the headline); \
disconnected ancient-world or prehistory journalism with no England link, \
no local angle, and no curriculum framing (see the period note below); \
Scottish/Welsh policy.
  0 = not education-relevant at all.

Period note for SUBJECT: the secondary history curriculum runs roughly \
1066 to the present, but this is a TIME rule, not a hard cutoff or a \
geography one -- the KS3 programme of study includes a local-history \
study and a thematic study that can predate 1066, several GCSE thematic \
papers run from c1000, and OCR offers Ancient History at GCSE/A-level. \
Score pre-1066 content at full SUBJECT value when it's local (see \
LOCALITY below), tied to a named thematic study or exam-board option, or \
explicitly framed around teaching -- otherwise it scores 1, per the band \
above.

Source note: `source` is not a reputation shortcut -- don't score a story \
up for coming from a trusted outlet, that's handled separately and \
deterministically elsewhere in this pipeline, and doing it here too \
double-counts it. But a source's own declared identity IS legitimate \
evidence of subject-specificity, which is a different thing: a post from \
a named history-teacher-training blog or a university history department \
is evidence its content is history- and training-specific even when its \
own wording doesn't say so.

SUBSTANCE (0-3): is this genuinely NEW, or is it commentary, narrative, \
or explanation of already-established facts?
  3 = a primary discovery, finding, or reform EVENT actually happening -- \
new archival or archaeological findings, a genuinely new historical \
interpretation, an exam board confirming a real spec change, a policy \
actually being enacted, not just studied or proposed.
  2 = solid reporting or analysis of a real, specific development that \
isn't itself new information -- a study's results being reported \
(distinct from being the study itself), a genuine consultation or \
proposal not yet enacted, an analysis piece with real evidence behind it. \
A story about research showing that a policy harms pupil wellbeing sits \
here, not at 3: the finding is real, but the story is reporting ON \
research, not presenting a reform event or a discovery itself.
  1 = narrative, explainer, or summary journalism about facts that are \
already well established -- well-written and still worth reading, but \
not telling this reader, or the field, anything new.
  0 = trivia, puzzles, listicles, or no real content.

ACTIONABILITY (0-2): could this reader actually DO something with it -- \
use it as a lesson hook or resource, cite it in planning, change a \
practice because of it?
  2 = directly usable: a primary source, a teaching resource, a case \
study concrete enough to build a lesson around, a practice change they \
could make tomorrow.
  1 = informs their thinking or context, but nothing to directly act on.
  0 = no practical application for this reader.

LOCALITY (0-1): the Nottinghamshire premium -- deliberately small and \
structurally capped, so it can nudge a borderline story up but never \
carry one on its own. Award 1 when the story is ITSELF about \
Nottinghamshire's history, heritage, or archaeology, OR about a \
Nottinghamshire school, multi-academy trust, ITT provider, or the local \
teaching job market. A story framed as covering a wider region (e.g. "the \
East Midlands") that is substantively about Nottinghamshire content still \
counts -- don't withhold this point on a technicality of exact wording. \
Be strict about the history facet specifically: a modern event that \
merely takes place AT a historic Nottinghamshire site -- a cinema night, \
a concert, a market -- gets 0 here, however prominently it names the \
site; score what the story is ABOUT, not which building it namedrops.

This is a rolling 14-day digest, not a same-day alert -- score SUBSTANCE \
for lasting value, not literal urgency.

For each story below, return all four dimension scores and a single \
short clause (under 25 words, no full stop) saying why it matters to \
them specifically -- or, for a low total, why it doesn't.

Return ONLY a JSON array, no prose, no code fence:
[{{"id": "...", "subject": 0-4, "substance": 0-3, "actionability": 0-2, "locality": 0-1, "why": "..."}}]

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
                # Raised from 320, 2026-08-21: real corpus data showed some
                # feeds front-load this with boilerplate that eats the old
                # budget before any real content -- Visit Nottinghamshire's
                # venue/ticket details ahead of what an exhibit is actually
                # about, in exactly the cases where the DIRECT vs ABOUT test
                # above matters most. Doesn't fix every feed (Cambridge
                # Faculty of History leads with an author byline that just
                # continues past a larger budget too), but checked against
                # real data rather than guessed.
                "summary": item.summary[:500],
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
        # Additive, 2026-08-21: the model returns four independent
        # dimensions (see RUBRIC) instead of one holistic 0-10 judgement,
        # and the final score is their sum -- computed here, not asked of
        # the model directly, so a single dominant dimension (e.g. "this
        # affects lots of pupils") can't inflate the total by leaking into
        # a dimension it doesn't belong in. Each is clamped to its own
        # range independently; a missing field defaults to 0 rather than
        # failing the whole verdict, same tolerance as the old single-score
        # parsing had for a missing "score" key.
        try:
            subject = max(0.0, min(4.0, float(verdict.get("subject", 0))))
            substance = max(0.0, min(3.0, float(verdict.get("substance", 0))))
            actionability = max(0.0, min(2.0, float(verdict.get("actionability", 0))))
            locality = max(0.0, min(1.0, float(verdict.get("locality", 0))))
        except (TypeError, ValueError):
            continue
        score = subject + substance + actionability + locality
        # Raised from 160, 2026-08-21: the rubric asks for "under 25 words",
        # which averages 150-170 characters -- 160 was clipping the longest
        # ones mid-word. 220 gives real headroom above that average.
        why = str(verdict.get("why", "")).strip()[:220]
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
