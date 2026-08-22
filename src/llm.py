"""Stage 2 of the two-stage LLM pipeline, run once per item during
ingestion, never at page-view time. Stage 1 (src/classify.py) triages every
new item into a TYPE and discards IRRELEVANT ones, unscored; this module
scores each survivor against universals plus its own type's Likert scale,
batched by type so every batch only ever carries the questions that
actually apply to what's in it.

Dormant unless ANTHROPIC_API_KEY is set in the environment. Every failure
path -- no key, missing package, client construction, or a network/API/
schema-validation error that fails a whole batch -- falls back to
deterministic scoring for exactly that batch's items, never the whole run.
Per-item omission within an otherwise-successful response isn't a separate
failure path any more: output_config's json_schema (see _stage2_schema)
pins each batch's response to exactly one verdict per item, so a response
either satisfies that or the call fails outright. The build must never fail
because of a billing or API problem.

To switch it on:
  1. Add ANTHROPIC_API_KEY as a repository secret (Settings → Secrets and
     variables → Actions → New repository secret).
  2. Uncomment the `ANTHROPIC_API_KEY` line in .github/workflows/brief.yml.
"""

from __future__ import annotations

import json
import os

from .fetch import Item

DEFAULT_MODEL = "claude-sonnet-5"


def _resolve_model(cfg: dict) -> str:
    """BRIEF_LLM_MODEL overrides scoring.yml's llm_model for quick local
    testing, matching the same env-var-overrides-config pattern as
    BRIEF_WINDOW_HOURS in build.py. Shared with classify.py so both stages
    of the pipeline resolve the model the same way."""
    return os.environ.get("BRIEF_LLM_MODEL") or cfg.get("llm_model", DEFAULT_MODEL)


def available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _extract_text_block(response):
    """Find the text block in a response's content list rather than
    indexing content[0] -- current-generation models (Sonnet 5 among them)
    run adaptive thinking by default even with no `thinking` param set at
    all, which puts a thinking block ahead of the text block, so content[0]
    is a ThinkingBlock with no .text attribute. Explicitly disabling
    thinking is a documented pitfall on this model family (occasional
    tool-call-in-visible-text / thinking-tag leakage), so this finds the
    actual text block instead of fighting the default off. Shared with
    classify.py -- both stages hit the exact same response shape."""
    text_block = next((b for b in response.content if getattr(b, "type", None) == "text"), None)
    if text_block is None:
        raise ValueError(f"no text block in response (stop_reason={response.stop_reason!r})")
    return text_block


# ---------------------------------------------------------------------------
# Stage 2 prompts. Every batch is one call's worth of one TYPE (assigned by
# stage 1) -- the shared preamble plus universals always applies; the type
# scale is appended only for the five types that have one. SECTOR and OTHER
# get universals alone, no type scale, no "s" items -- see scoring.yml's
# `weights` comment for how the two combine into a final 0-10 relevance.
# ---------------------------------------------------------------------------

# Anchors are load-bearing: the model scores each story against a fixed
# rubric, never against the other stories in the batch. Batch-relative
# ranking is exactly the bug the fixed-ceiling deterministic normalisation
# (score.py) also exists to avoid -- see BUILD-SPEC.md's "scoring must be
# absolute" section. Batching by type makes this a sharper risk than the
# single-pass rubric this replaced: every batch looks alike (same type), so
# the preamble says so explicitly -- a weak fortnight of CURRICULUM stories
# must produce low scores, not a re-ranked set of high ones.
STAGE2_PREAMBLE = """You are scoring education and history news for one \
specific reader: a career-changer in England entering secondary HISTORY \
teaching. They are currently {reader_stage}.

Every article below has already been classified as {type_name}. Do not \
re-judge the type. Score each one against the fixed anchors below.

Score each article ON ITS OWN MERITS against these anchors. Do NOT rank or \
compare the articles against each other. These articles have been grouped by \
type, so they will resemble one another; that is expected and must not push \
scores up. If every article in this batch is weak, every score should be \
low. There is no quota of high scores.

ALWAYS ANSWER THESE TWO:

U1 CONSEQUENCE
If this is true or goes ahead, how much changes for this reader, or for the \
pupils and schools they will work in?
  4  reshapes something whole - a curriculum, a cohort's experience, a system
  2  a real but bounded change affecting some schools, subjects or pupils
  0  nothing changes as a result

U2 DISCOVERY VALUE
How likely is it that this reader would MISS this without a curated brief?
  4  obscure or specialist - a small archive post, a single study, a local \
find; they would almost certainly never see it
  2  they might catch it if they were looking in the right place
  0  unavoidable headline news, or the latest of several near-identical \
reports of the same event
Score this on how hard the item is to discover, not on how good it is."""

_CURRICULUM_SCALE = """C1 HISTORY DIRECTNESS
  4  about the history curriculum, a history specification, or a history \
qualification
  3  a cross-subject change that demonstrably moves history's content, \
timetable share or uptake - the EBacc, Progress 8, KS3 structure, \
option-block policy
  2  a change to how all subjects are examined, history affected as one of \
many
  1  about another subject's curriculum or qualifications
  0  not about what is taught or examined

C2 STAGE
  4  a decision made - a specification published, an option withdrawn, a \
statutory change confirmed
  3  a formal proposal or consultation with a named timetable
  2  an official review or inquiry under way
  1  comment or speculation about possible change
  0  no change is involved

C3 REACH
  4  changes a subject's content or structure across KS3, GCSE or A-level
  3  changes one qualification, one key stage, or a major option
  2  changes assessment mechanics or a single component
  1  marginal or administrative
  0  no effect on what is taught"""

_HISTORY_SCALE = """H1 CURRICULUM PROXIMITY
  4  squarely on a period, theme or place taught at KS3, GCSE or A-level -- \
including history set outside Britain: Empire and its legacy, migration, \
the Cold War, Russia, China, the USA, apartheid South Africa and \
comparable options are standard, growing GCSE/A-level content. This is \
history taught IN England, not only history OF England -- don't discount \
a story for being set outside Britain.
  3  adjacent to taught content, or within the range of a thematic study \
running from c.1000
  2  British history outside the periods usually taught
  1  world history with no curricular foothold
  0  not about the past
English or British medieval material (Norman Conquest to roughly 1500) \
does not need a named curricular link the way pre-1066 material does --
score it 3-4 by default, even from a bare title and a thin summary with no \
named monarch or event, unless something marks it as OUTSIDE an English/ \
British context. "Medieval" alone, from an English institution (a national \
or cathedral archive, an English university, English heritage body), is \
enough: this is squarely "the range of a thematic study running from \
c.1000", not a vague gesture toward antiquity the way an undated "ancient \
world" claim is. Be strict about the SEPARATE pre-1066 exception below --
it exists for a NAMED curricular link, not a general gesture toward \
antiquity. The KS3 programme of study \
includes a local history study and a thematic strand that explicitly \
reaches back before 1066; OCR offers Ancient History at GCSE and A-level. \
Score pre-1066 material on its merits ONLY when it names that specific \
link (a documented thematic paper, OCR's Ancient History option, the \
local-history study) or is local (see LOCALITY) -- a vague invocation of \
"early civilisations" or "the ancient world" is not a named link. \
Mythology, literature, and "what was life like" cultural pieces about the \
classical world -- the Odyssey, ancient sexuality, daily life in \
Mesopotamia -- are Classical Civilisation, a DIFFERENT subject from \
History, not covered by this exception even when Greece or Rome is in the \
headline: the test is whether the story is actually about a documented \
history curriculum topic, not whether it's set in the ancient world. \
Score disconnected ancient-world or prehistory content, named link or \
not, at 0-1.

H2 SUBSTANCE
  4  new research, a discovery, or a substantive reinterpretation
  2  a competent retelling that contains something fresh
  0  recycled narrative, a staff or student profile, an accreditation \
notice, or a personal travelogue

H3 CLASSROOM USABILITY
  4  a source, image, dataset, site or collection that could go in front of \
a class with little adaptation
  2  usable after real preparatory work
  0  nothing a class could use

H4 DISCIPLINARY PURCHASE
  4  raises a genuine historical question, or shows how knowledge of the \
past was constructed from evidence -- this includes archival/curatorial \
"how we know" research (conservation science on an object, provenance or \
authentication work, a newly catalogued item), which is disciplinary \
methodology made visible, not just content about the object
  2  illustrates interpretation or contested accounts implicitly
  0  pure content, or nothing to interrogate"""

_PEDAGOGY_SCALE = """G1 EVIDENCE STRENGTH
  4  a synthesis, replication, or well-designed trial
  2  a single study, or a careful and specific practitioner account
  0  assertion, opinion, or a vendor claim

G2 SETTING FIT
  4  secondary classrooms
  2  primary, FE, or mixed-age settings
  0  adult, undergraduate or laboratory findings presented as classroom \
evidence

G3 ACTIONABILITY
  4  could be tried next term substantially as described
  2  a principle the reader would have to translate into practice \
themselves
  0  nothing to do differently

G4 HISTORY TRANSFER
  4  bears directly on teaching history - extended writing, source work, \
causal explanation, chronology, disciplinary vocabulary
  2  general but clearly applicable to a history classroom
  0  specific to another subject"""

_PUPILS_SCALE = """P1 AGE FIT
  4  about 11-18s specifically
  2  about school-age children generally
  0  early years, adults, or an undifferentiated "young people"

P2 EVIDENCE QUALITY
  4  research findings or substantive data
  2  reported experience or professional testimony
  0  opinion or anecdote

P3 CLASSROOM BEARING
  4  would change how the reader reads or responds to a pupil in front of \
them
  2  useful background on the cohort they will teach
  0  no bearing on the classroom"""

_CAREER_SCALE = """K1 STAGE FIT
  4  bites during initial teacher training or induction
  2  bites within their first few years in post
  0  distant, hypothetical, or applies to a career stage they will not \
reach soon

K2 MATERIALITY
  4  changes money, workload, entitlement, job availability, or what a \
teacher is held accountable for
  2  evidence or comment about conditions with no change attached
  0  no bearing on the terms of the job

K3 EXPOSURE
  4  hits history teachers, or East Midlands schools, specifically
  2  hits secondary teachers in England generally
  0  another phase, sector or nation
Judge what the finding or technique is ABOUT and who it transfers to, not \
which phase or sector its research population or setting happened to be \
drawn from -- a reflective-practice or mentoring technique studied with FE \
trainees that plainly generalises to teacher training generally is exposure \
2, not 0, the same way a study of secondary pupils doesn't need to specify \
"history pupils" to count. Score 0 only when the finding itself is specific \
to that other phase or sector (FE funding, FE curriculum, primary-only \
practice) and wouldn't transfer."""

TYPE_NAMES = {
    "CURRICULUM": "curriculum and qualifications news",
    "HISTORY": "historical scholarship and resources",
    "PEDAGOGY": "pedagogy and learning evidence",
    "PUPILS": "pupils and adolescence",
    "CAREER": "career, conditions and accountability",
    "SECTOR": "general sector news",
    "OTHER": "news that doesn't fit a specific category",
}

# SECTOR and OTHER deliberately absent here -- no type scale, universals
# alone carry the score (see _score_batch_typed's assembly formula).
TYPE_SCALES = {
    "CURRICULUM": _CURRICULUM_SCALE,
    "HISTORY": _HISTORY_SCALE,
    "PEDAGOGY": _PEDAGOGY_SCALE,
    "PUPILS": _PUPILS_SCALE,
    "CAREER": _CAREER_SCALE,
}

# Lowercase, matching the labels already used in each _*_SCALE string above
# (C1/C2/C3, H1-H4, ...) -- reusing them as the JSON keys the model must
# answer with means the scale text already explains what each key means,
# with nothing extra to keep in sync.
TYPE_SCORE_KEYS = {
    "CURRICULUM": ["c1", "c2", "c3"],
    "HISTORY": ["h1", "h2", "h3", "h4"],
    "PEDAGOGY": ["g1", "g2", "g3", "g4"],
    "PUPILS": ["p1", "p2", "p3"],
    "CAREER": ["k1", "k2", "k3"],
    "SECTOR": [],
    "OTHER": [],
}


def _build_stage2_prompt(item_type: str, reader_stage: str, payload: str) -> str:
    """Assemble one type's scoring prompt: shared preamble (formatted with
    this type's name), its Likert scale if it has one, then the output
    contract. Response shape (one verdict per article, none omitted) is
    enforced by the json_schema passed alongside this prompt in
    _score_batch_typed -- schema-invalid is a harder constraint than a
    prose instruction the model can silently under-comply with, which is
    exactly what happened before this was schema-enforced (see the note on
    llm_score_batch_size in scoring.yml)."""
    type_name = TYPE_NAMES.get(item_type, item_type)
    scale = TYPE_SCALES.get(item_type)
    score_keys = TYPE_SCORE_KEYS.get(item_type, [])

    parts = [STAGE2_PREAMBLE.format(reader_stage=reader_stage, type_name=type_name)]
    if scale:
        parts.append(scale)

    fields = ", ".join(f'"{f}"' for f in ["u1", "u2", *score_keys, "why"])
    parts.append(
        "Return one entry under \"verdicts\" for every article below, keyed "
        f"by that article's id exactly as given. Each entry has: {fields} "
        "-- \"why\" is one clause of at most 20 words, no full stop, saying "
        "why it matters to this reader, or why it does not."
    )
    parts.append(f"Articles:\n{payload}")
    return "\n\n".join(parts)


def _stage2_schema(batch_ids: list[str], score_keys: list[str]) -> dict:
    """json_schema for one stage-2 batch: "verdicts" is an OBJECT keyed by
    the batch's own ids, not an array -- Anthropic's structured outputs only
    supports array minItems of 0 or 1 (not an arbitrary exact count) and
    doesn't support maxItems, minimum/maximum or maxLength at all (see the
    note on llm_score_batch_size in scoring.yml for how the first version of
    this schema, built on those unsupported keywords, actually failed every
    single request in production). `required` + `additionalProperties:
    False` on an OBJECT has no such limit, so "verdicts" requiring exactly
    batch_ids as keys gets the same one-verdict-per-item guarantee through a
    mechanism the API actually supports. u/s values can no longer be
    range-bounded by the schema itself (no minimum/maximum) -- clamped in
    _score_batch_typed instead, same as before schema enforcement existed."""
    field_names = ["u1", "u2", *score_keys, "why"]
    verdict_schema = {
        "type": "object",
        "properties": {
            f: ({"type": "string"} if f == "why" else {"type": "number"}) for f in field_names
        },
        "required": field_names,
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "object",
                "properties": {bid: verdict_schema for bid in batch_ids},
                "required": batch_ids,
                "additionalProperties": False,
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _score_batch_typed(
    batch: list[Item], item_type: str, client, model: str, reader_stage: str, cfg: dict
) -> dict[str, tuple[float, str]]:
    """One API call for up to llm_score_batch_size items, all the same TYPE
    (grouped by the caller). Computes each item's final 0-10 relevance from
    the assembly formula in scoring.yml's `weights` block:

        universals = u1_consequence * U1 + u2_discovery * U2         # 0-4
        type_score = mean(s items) if the type has any, else None    # 0-4
        relevance = (type_score * .6 + universals * .4) * 2.5        # 0-10
                    -- or universals * 2.5 alone when type_score is None
                    (SECTOR, OTHER)

    Raises on total failure (network error, a response that can't satisfy
    the schema, missing text block) -- the caller treats that as "none of
    this batch got a verdict", and because batching is per-type, a failure
    here can never affect another type's batches.
    """
    weights = cfg.get("weights", {})
    u1_w = float(weights.get("u1_consequence", 0.6))
    u2_w = float(weights.get("u2_discovery", 0.4))
    type_w = float(weights.get("type_score", 0.6))
    universals_w = float(weights.get("universals", 0.4))
    max_tokens = int(cfg.get("llm_max_tokens", 12000))
    score_keys = TYPE_SCORE_KEYS.get(item_type, [])

    payload = json.dumps(
        [
            {
                "id": item.uid,
                "title": item.title,
                "summary": item.summary[:500],
                "source": item.source_name,
            }
            for item in batch
        ],
        ensure_ascii=False,
        indent=1,
    )
    prompt = _build_stage2_prompt(item_type, reader_stage, payload)
    schema = _stage2_schema([item.uid for item in batch], score_keys)
    response = client.messages.create(
        model=model,
        # Generous, not tuned to typical usage: adaptive thinking spends
        # part of this budget before a single verdict token is written, so
        # a tight cap risks truncating the response mid-batch -- see the
        # comment on llm_max_tokens in scoring.yml.
        max_tokens=max_tokens,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = _extract_text_block(response)
    verdicts = json.loads(text_block.text)["verdicts"]

    # The schema guarantees exactly one verdict per batch id (verdicts is
    # keyed by them, required, additionalProperties False) and the right
    # fields on each -- but NOT their numeric range, since minimum/maximum
    # aren't schema features this API supports (see _stage2_schema). u/s
    # are clamped here for that reason, same as before schema enforcement
    # existed.
    results: dict[str, tuple[float, str]] = {}
    for item_id, verdict in verdicts.items():
        u1 = max(0.0, min(4.0, float(verdict["u1"])))
        u2 = max(0.0, min(4.0, float(verdict["u2"])))
        s_vals = [max(0.0, min(4.0, float(verdict[k]))) for k in score_keys]

        universals = u1_w * u1 + u2_w * u2  # 0-4
        if s_vals:
            # Plain mean, no per-item weighting yet -- see the placeholder
            # note on `weights` in scoring.yml.
            type_score = sum(s_vals) / len(s_vals)  # 0-4
            relevance = (type_w * type_score + universals_w * universals) * 2.5
        else:
            # SECTOR / OTHER: no type scale, universals alone carry it.
            relevance = universals * 2.5
        # Guards against scoring.yml's weights not summing to 1.0, on top of
        # the u/s clamping above.
        relevance = max(0.0, min(10.0, relevance))

        # maxLength isn't schema-enforceable either (see _stage2_schema) --
        # truncated here instead. 220, not the 20-word limit's naive char
        # estimate: 20 words averages 120-140 characters, but a few
        # genuinely long words push past that.
        why = str(verdict["why"]).strip()[:220]
        results[item_id] = (relevance, why)
    return results


def rerank_new_items(
    items: list[Item], classify_results: dict[str, dict], cfg: dict
) -> tuple[dict[str, tuple[float, str]], str]:
    """Score every new item that survived stage 1 classification (see
    classify.classify_items), batched by type. Returns (verdicts_by_id,
    status). Only ids present in verdicts_by_id got an LLM score; the
    caller falls back to deterministic scoring for everything else, item by
    item -- never as an all-or-nothing decision for the whole run.

    Items classified IRRELEVANT are excluded here, not deleted anywhere --
    they simply never get a verdict, so score_new_items() in build.py falls
    them back to deterministic scoring exactly like any other missing
    verdict (no API key, a failed batch, ...). A genuinely irrelevant story
    scores near-zero deterministically anyway, and publish_floor keeps it
    off every published surface while still remembering it for dedup -- the
    same mechanism already used for below-floor items, not a new discard
    path.
    """
    if not items:
        return {}, "no new items"
    if not available():
        return {}, "deterministic only (no API key set)"

    survivors = [
        item for item in items
        if classify_results.get(item.uid, {}).get("type") != "IRRELEVANT"
    ]
    discarded = len(items) - len(survivors)
    if not survivors:
        return {}, f"no items to score ({discarded} irrelevant, discarded)"

    import anthropic

    model = _resolve_model(cfg)
    reader_stage = cfg.get("llm_reader_stage", "at the point of entering initial teacher training")

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 -- must not break the run
        return {}, f"deterministic only (client init failed: {type(exc).__name__})"

    # Group by type -- every batch carries only the Likert scale that
    # actually applies to what's in it.
    by_type: dict[str, list[Item]] = {}
    for item in survivors:
        item_type = classify_results.get(item.uid, {}).get("type", "OTHER")
        by_type.setdefault(item_type, []).append(item)

    batch_size = max(1, int(cfg.get("llm_score_batch_size", 20)))
    total_batches = sum(
        (len(type_items) + batch_size - 1) // batch_size for type_items in by_type.values()
    )

    verdicts: dict[str, tuple[float, str]] = {}
    errors: list[str] = []
    batch_n = 0
    for item_type, type_items in by_type.items():
        for i in range(0, len(type_items), batch_size):
            batch_n += 1
            batch = type_items[i : i + batch_size]
            try:
                # Each type's batches are tried independently -- one type's
                # failure can never affect another's, since this update()
                # only ever touches the ids from this one batch.
                verdicts.update(_score_batch_typed(batch, item_type, client, model, reader_stage, cfg))
            except Exception as exc:  # noqa: BLE001 -- deliberately broad
                # The message, not just the class name -- see the identical
                # comment in classify.py's batch loop.
                errors.append(f"{item_type} batch {batch_n}/{total_batches}: {type(exc).__name__}: {exc}")

    discard_note = f"{discarded} irrelevant, discarded" if discarded else ""

    if not verdicts:
        detail = "; ".join(errors) or "no verdicts returned"
        suffix = f" ({discard_note})" if discard_note else ""
        return {}, f"deterministic only (LLM pass failed: {detail}){suffix}"

    missing = len(survivors) - len(verdicts)
    if errors or missing:
        detail = "; ".join(errors) if errors else f"{missing} item(s) omitted from responses"
        bits = "; ".join(b for b in [f"degraded: {detail}", discard_note] if b)
        return verdicts, f"LLM re-ranked with {model} ({len(verdicts)}/{len(survivors)}; {bits})"

    suffix = f" ({discard_note})" if discard_note else ""
    return verdicts, f"LLM re-ranked with {model}{suffix}"
