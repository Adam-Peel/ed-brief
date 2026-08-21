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

Score each item below 0-4, anchored at 0/2/4 -- the same scale \
throughout, so every item is judged the same mental way. Each item \
measures ONE thing only. How the items combine into a final score is \
handled separately, after your answers -- not something to weigh in \
yourself.

=== DOMAIN 1: SUBJECT (history as a school subject) ===

1.1 Curriculum, qualifications and assessment: to what extent is this \
about what school history contains or how it's examined -- KS3 content, \
GCSE/A-level specs, exam board options (AQA, Edexcel, OCR, WJEC), entry \
numbers, or curriculum/assessment reform as it reaches history?
  4 = what school history teaches or how it's assessed IS the subject of \
the story
  2 = school history is one substantial strand of a broader curriculum or \
qualifications story
  0 = no bearing on what history classrooms teach or how it's assessed
A cross-curricular reform story (the Francis Review, EBacc, Progress 8, a \
structural Ofqual change) scores at least 2 here even when history is \
never named, because it determines history's content, timetable share, \
or uptake.

1.2 Subject knowledge for periods they might teach: to what extent does \
this add to the reader's own historical knowledge of a period, place or \
theme on the secondary curriculum -- new research, interpretations, \
discoveries, or newly released archive material (The National Archives' \
periodic openings and digitisation projects count as much as academic \
papers)?
  4 = substantive new historical knowledge about a period, event or theme \
taught at KS3, GCSE or A-level
  2 = real historical content, but a period, place or scale they'd rarely \
teach
  0 = not about the past
Period is not a hard gate: pre-1066 content scores on its merits when \
it's local (see LOCALITY), tied to a thematic study running from c.1000, \
or maps to a named exam-board option (e.g. OCR Ancient History). Reserve \
0-1 for ancient-world or prehistory writing with no England, local, or \
curricular connection.

1.3 How young people learn history: to what extent is this about pupils' \
historical thinking specifically -- progression in ideas about evidence, \
causation, chronology, significance, or interpretation, or how \
adolescents reason about the past?
  4 = research or practice squarely on pupils' historical thinking
  2 = generic pedagogy or learning science with an explicit history \
application
  0 = no connection to how history in particular is learned

SUBJECT = the HIGHEST of 1.1-1.3, not their average -- alternative routes \
into the subject, not converging signs. A pure curriculum story shouldn't \
be marked down for containing no new scholarship; scoring strongly on ONE \
of these is what matters.

=== DOMAIN 2: CAREER (their route in, and their working life once in) ===

2.1 Training and qualifying: to what extent does this affect how they \
train and gain QTS -- ITT routes, PGCE and SCITT, bursaries and funding, \
provider accreditation or inspection, placement supply, or the ITTECF as \
it governs their training year?
  4 = directly changes the terms, cost, availability or content of their \
training
  2 = affects ITT generally, or a route/subject other than theirs
  0 = no bearing on training in England

2.2 Getting and keeping a post: to what extent does this affect whether \
secondary history posts exist and whether they can get one -- \
recruitment and vacancies, subject-level demand, falling rolls, school \
funding, staffing cuts, trust restructuring, closures?
  4 = materially changes the number or nature of posts they could apply \
for
  2 = affects the sector's labour market broadly, or with a long lag
  0 = no bearing on secondary teaching employment

2.3 Working conditions in post: to what extent does this affect what the \
job is like to do -- workload, pay, pensions, the STRB, industrial \
action, retention, induction, mentoring, or ECT entitlement?
  4 = directly changes pay, workload or entitlements for teachers in \
England
  2 = evidence or comment about conditions without a change attached
  0 = no bearing on the terms of the job

2.4 Accountability and statutory duties: to what extent does this change \
what they'll be held accountable for -- school inspection, accountability \
measures, or statutory duties such as KCSIE, behaviour and attendance \
guidance?
  4 = changes the framework, measures or duties a secondary teacher works \
under
  2 = commentary on inspection or accountability without changing it
  0 = no bearing on secondary school accountability
Strict: Ofsted also regulates children's social care and early years, and \
the DfE covers early years and HE. A story about Ofsted prosecuting an \
illegal children's home, or an annual report as an administrative \
document, scores 0 here -- the test is what the story is about, not whose \
name is in the headline.

CAREER = the HIGHEST of 2.1-2.4 -- alternative routes again: a bursary \
story and a pay story are both fully career-relevant, by different \
mechanisms.

=== DOMAIN 3: SUBSTANCE (is there anything actually here, and does it matter) ===

3.1 Is anything new? To what extent does this report something that's \
actually happened, been decided, been found, or been released, rather \
than discussing something already established?
  4 = reports a new decision, finding, dataset, release, or event
  2 = a small new development wrapped in substantial recap
  0 = entirely commentary, opinion, or retelling of known facts

3.3 Magnitude: if what this describes is true or goes ahead, to what \
extent does it change things?
  4 = reshapes the system, the curriculum, or a whole cohort's experience
  2 = a real but bounded change, affecting some schools, subjects or \
pupils
  0 = nothing changes as a result
A story about research showing that a policy harms pupil wellbeing is \
real (score 3.1 accordingly) but reports a STUDY, not an enacted change -- \
score magnitude for what the story itself describes happening, not the \
scale of the underlying problem it studies.

3.4 Durability: to what extent would this still matter to the reader in \
a month's time?
  4 = still material at the end of term
  2 = matters for a few weeks
  0 = a one-day story
Tuned to a 14-day rolling digest, not a same-day alert -- this replaces \
"must read today" urgency, which this reader's corpus can't support \
anyway.

SUBSTANCE = the AVERAGE of 3.1, 3.3 and 3.4, not the highest -- unlike \
the other domains, these genuinely converge: an item strong on all three \
is genuinely substantial, and real weakness on any one should pull the \
whole domain down, not be masked by strength elsewhere.

=== DOMAIN 4: ACTIONABILITY (could they do something with it) ===

4.1 Classroom material: to what extent could this go into a lesson more \
or less as it stands -- a source, image, dataset, site, story, archive \
release, or anniversary hook?
  4 = usable in a lesson this term with little preparation
  2 = usable after real work, or for a topic they may not teach
  0 = nothing lesson-usable

4.2 Planning and sequencing: to what extent would this change a planning \
decision -- what to teach, in what order, which option or specification \
to choose?
  4 = would change a concrete planning or option decision
  2 = worth knowing when planning, without forcing a change
  0 = no planning implication

4.3 Practice in the room: to what extent would this change something \
they actually do when teaching -- a technique, an assessment or feedback \
practice, or how they support a particular pupil?
  4 = a specific, adoptable change to practice
  2 = a principle they'd need to translate into practice themselves
  0 = nothing to change

ACTIONABILITY = the HIGHEST of 4.1-4.3 -- any one route to use is enough; \
a National Archives release is fully actionable as material even though \
it changes no one's teaching practice.

=== DOMAIN 5: LOCALITY (Nottinghamshire) ===

5.1 The county's past: to what extent is the story itself about \
Nottinghamshire's past -- its history, heritage, archaeology, archives, \
or historic landscape?
  4 = the county's past is the subject of the story
  2 = Nottinghamshire is one of several places covered
  0 = not about the county's past
Strict: a modern event merely held AT a historic Nottinghamshire site -- \
a concert, a market, a screening, a wedding fair -- scores 0, however \
prominently the site is named. Score what the story is ABOUT, not which \
building it namedrops.

5.2 Local schools and providers: to what extent is this about a \
Nottinghamshire school, multi-academy trust, or ITT provider?
  4 = a named local school, trust, or provider is the subject
  2 = a regional story covering Nottinghamshire among other areas
  0 = no local institution involved

5.3 Local market and policy: to what extent is this about the \
Nottinghamshire or East Midlands teaching job market, or local authority \
education policy?
  4 = directly about local demand, vacancies, or local education policy
  2 = regional coverage including the area
  0 = no local labour-market or policy content

LOCALITY = the HIGHEST of 5.1-5.3. This is scored like the others but is \
NOT summed into the final total with the domains above -- it's near zero \
for most of the corpus, so it works as a guaranteed floor instead: a \
story scoring 4 on any locality item is guaranteed a place in the \
"worth a look" range regardless of the other domains, so genuinely local \
stories reliably surface without needing to outrank real curriculum or \
career news to do it.

Source note: `source` is not a reputation shortcut for any item above -- \
don't score a story up for coming from a trusted outlet, that's handled \
separately and deterministically elsewhere in this pipeline, and doing it \
here too double-counts it. But a source's own declared identity IS \
legitimate evidence for SUBJECT specifically: a post from a named \
history-teacher-training blog or a university history department is \
evidence its content is history- and training-specific even when its own \
wording doesn't say so.

For each story below, return every item score (0-4 each) using these \
exact keys -- 1_1, 1_2, 1_3, 2_1, 2_2, 2_3, 2_4, 3_1, 3_3, 3_4, 4_1, 4_2, \
4_3, 5_1, 5_2, 5_3 -- and a single short clause (under 25 words, no full \
stop) saying why it matters to them specifically, or for a low total, why \
it doesn't.

Return ONLY a JSON array, no prose, no code fence:
[{{"id": "...", "1_1": 0-4, "1_2": 0-4, "1_3": 0-4, "2_1": 0-4, "2_2": 0-4, "2_3": 0-4, "2_4": 0-4, "3_1": 0-4, "3_3": 0-4, "3_4": 0-4, "4_1": 0-4, "4_2": 0-4, "4_3": 0-4, "5_1": 0-4, "5_2": 0-4, "5_3": 0-4, "why": "..."}}]

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
        # Item bank, 2026-08-21: the model answers 16 single-concept "to what
        # extent" items (see RUBRIC), each 0-4 on the same scale for scoring
        # consistency -- a missing key defaults to 0 rather than failing the
        # whole verdict, same tolerance the old single-score parsing had for
        # a missing "score" key.
        try:
            def item(key: str) -> float:
                return max(0.0, min(4.0, float(verdict.get(key, 0))))

            subject_items = [item("1_1"), item("1_2"), item("1_3")]
            career_items = [item("2_1"), item("2_2"), item("2_3"), item("2_4")]
            substance_items = [item("3_1"), item("3_3"), item("3_4")]
            actionability_items = [item("4_1"), item("4_2"), item("4_3")]
            locality_items = [item("5_1"), item("5_2"), item("5_3")]
        except (TypeError, ValueError):
            continue

        # Two different aggregation rules, not one: SUBJECT/CAREER/
        # ACTIONABILITY/LOCALITY take the MAX of their items, because those
        # items are alternative routes into the domain, not converging
        # signs -- a pure curriculum story shouldn't be marked down for
        # containing no new scholarship. SUBSTANCE takes the MEAN, because
        # its three items (is anything new / magnitude / durability)
        # genuinely converge on one question -- real weakness on any one
        # should pull the whole domain down, not be masked by the others.
        subject = max(subject_items)
        career = max(career_items)
        substance = sum(substance_items) / len(substance_items)
        actionability = max(actionability_items)
        locality = max(locality_items)

        # Weighted average of the four summed domains (weights sum to 3.0),
        # rescaled from the shared 0-4 item scale to the final 0-10 scale
        # (x2.5). Weights are never shown to the model -- it can't discount
        # a domain it knows "counts less", or start doing its own mental
        # arithmetic instead of just answering "to what extent" each time.
        # These weights are placeholders, not a calibrated result: they
        # reproduce the previous four-dimension design's relative-importance
        # ratio (1 / .75 / .75 / .5) rather than anything empirically fit.
        raw = subject * 1.0 + career * 0.75 + substance * 0.75 + actionability * 0.5
        score = (raw / 3.0) * 2.5

        # LOCALITY is deliberately excluded from that sum -- it's 0 for most
        # of the corpus, so any weight small enough not to distort the
        # ranking is also too small to ever change one. Used as a gate and a
        # floor instead, which can never both fire on the same item (the
        # gate requires locality == 0, the floor requires locality == 4):
        if subject <= 1 and career <= 1 and locality == 0:
            # Nothing irrelevant floats up on substance/actionability alone
            # -- a hugely important, highly actionable story about
            # something outside this reader's subject and career is still
            # not for this reader.
            score = min(score, 3.0)
        if locality == 4:
            # A genuinely local story always reaches "worth a look",
            # without needing to outrank real curriculum/career news to
            # get there.
            score = max(score, 6.0)
        score = max(0.0, min(10.0, score))
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
