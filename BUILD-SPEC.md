# Build spec — ed-brief production version

*Paste this whole file into Claude Code as the opening prompt, in a directory
containing the proof-of-concept repo.*

---

You are building the production version of **ed-brief**, a twice-weekly
education news brief for a career-changer entering secondary **history**
teaching in England.

A working proof of concept is in this directory. It fetches eight feeds, scores
them against a controlled vocabulary, and publishes a static page. Read it
first — `README.md`, then `src/`, then `config/` — and run
`python tests/test_pipeline.py` to confirm it passes before changing anything.
The proof of concept establishes the scoring approach and the source list. It
is not the architecture you are shipping.

**Before writing code, produce a short plan** covering the data model, the
build pipeline stages, and the API surface, and check it against the acceptance
criteria at the end of this document.

---

## What changes from the proof of concept

The PoC publishes a snapshot per run: each brief contains only stories new
since the last run, and nothing persists. Four requirements change that.

1. **LLM ranking happens during ingestion, in the same build step as sourcing,
   and always before anything is pushed.** The published site and API are fully
   pre-ranked static data. Nothing is scored, sorted, or fetched at page-view
   time.
2. **Items persist for 14 days** from first sighting, or until the reader marks
   them read. The site becomes one rolling list rather than a series of
   snapshots.
3. **A public read-only JSON API** exposes every sourced and ranked item, so a
   mobile app can be built against it later.
4. **The published site is strictly read-only.** No path exists from the web
   page back to the repository — no writes, no pushes, no workflow triggers, no
   tokens. This is a hard constraint to protect against abuse of Actions
   minutes and API spend, not a preference.

---

## Architecture

### Data model

Replace `state/seen.json` with a rolling corpus at `data/corpus.json`, the
single source of truth for everything the system knows about:

```jsonc
{
  "schema": 1,
  "generated": "2026-08-24T06:02:11Z",
  "retention_days": 14,
  "items": [
    {
      "id": "a1b2c3d4e5f6a7b8",          // stable, never reused
      "title": "...",
      "url": "https://...",
      "summary": "...",
      "source": { "id": "schools-week", "name": "Schools Week" },
      "published": "2026-08-23T14:20:00Z",
      "first_seen": "2026-08-24T06:02:11Z",
      "expires":    "2026-09-07T06:02:11Z",
      "relevance": 8.4,                   // FROZEN at ingest, 0-10
      "rank_score": 7.9,                  // relevance minus age penalty, recomputed each build
      "tier": "lead",
      "tags": ["sub-history", "pol-national-curriculum"],
      "why": "changes how KS3 history is sequenced",
      "scoring": {
        "mode": "llm",                    // or "deterministic"
        "deterministic_raw": 21.3,
        "deterministic_norm": 8.5,
        "llm": 8.0
      }
    }
  ]
}
```

`id` is the identity contract for the whole system, including the future mobile
app. Derive it as the PoC does — SHA-256 of the URL with query strings stripped,
truncated — so the same story arriving with different tracking parameters is one
item. Never change the derivation without bumping `schema`.

### Scoring must be absolute, not batch-relative

**This is the subtlest requirement in the build; get it wrong and the rolling
list silently misranks.**

The PoC normalises deterministic scores against the minimum and maximum of the
current batch. That is fine for a snapshot and wrong for a corpus, because
items scored in different runs then sit on different scales and cannot be
compared in one list.

- Normalise the deterministic score against a **fixed** ceiling from config
  (`deterministic_scale_max`, default 25.0): `min(raw, max) / max * 10`.
- Have the LLM return an **absolute** 0–10 relevance against a written rubric
  with anchors — 0 irrelevant, 5 useful to any teacher, 8 directly relevant to
  history ITT, 10 drop-everything. Never ask it to rank items against each
  other.
- Blend to a final `relevance` on 0–10 (default 60% LLM, 40% deterministic,
  configurable). Tier cuts are then fixed thresholds on that scale.

`relevance` is frozen at ingest and never recomputed — it cost money to produce
and the significance of a story doesn't change.

`rank_score` is `relevance` minus a small age penalty, **recomputed for the
whole corpus at every build**. This costs nothing, keeps fresher items above
equally relevant stale ones, and still satisfies requirement 1, because it
happens in the build before the push. The browser never computes it.

### Build pipeline

One command, `python -m src.build`, running these stages in order:

1. Load `data/corpus.json` (empty corpus on first run).
2. Fetch all enabled feeds concurrently, with per-feed timeouts and errors
   captured rather than raised.
3. Normalise and deduplicate: within the run, then against the existing corpus
   by `id`.
4. Deterministically score **only the new items**.
5. LLM-rank **only the new items**, batched, in this same step. Existing items
   keep their frozen `relevance`. This is both the cost control and requirement 1.
6. Merge new items into the corpus with `first_seen` and `expires`.
7. Drop items past `expires`.
8. Recompute `rank_score` and `tier` across the whole corpus; sort.
9. Emit the API files, the site, and a dated markdown brief of that run's new
   items to `briefs/`.
10. Commit `data/`, `docs/`, and `briefs/`.

Every LLM failure path — no key, missing package, API error, malformed
response, partial response — must fall back to deterministic scoring for the
affected items, record `"mode": "deterministic"` on them, and surface the
degradation in `meta.json` and the page footer. **The build must never fail
because of a billing or API problem.**

### API

Static JSON under `docs/api/v1/`, served by Pages. Public Pages sites send
`access-control-allow-origin: *`, so cross-origin fetches from a mobile app
work without configuration. Custom headers cannot be set on Pages, so the
client must use the `generated` timestamp for change detection.

| Endpoint | Contents |
|---|---|
| `/api/v1/index.json` | Discovery document: schema version, endpoint list, `generated` |
| `/api/v1/items.json` | Every live (unexpired) item, ranked |
| `/api/v1/latest.json` | Only items added by the most recent run |
| `/api/v1/meta.json` | Run metadata, per-source health, counts, scoring mode, retention |
| `/api/v1/tags.json` | Tags with item counts |
| `/api/v1/archive/YYYY-MM-DD.json` | Per-run snapshot, retained indefinitely |

Version the path from day one and treat `v1` as a contract: the mobile app will
depend on it. Additive changes only within a version.

The API always returns the full live corpus. Read-state is a presentation
concern and never filters the API.

### Site

One rolling list at `docs/index.html`, unread first, ranked by `rank_score`,
grouped into the three tiers. Retain the dated archive as a separate page.

- **Read-state in `localStorage`, behind a small `ReadStore` interface** with
  `isRead`, `markRead`, `markUnread`, `clear`. Isolating it means a sync
  backend could replace it later without touching the rest of the site. Wrap
  every read and write in try/catch: private windows and blocked site data
  throw on access, and the page must render correctly with no stored value.
- Read items are hidden by default with a toggle to reveal them, plus "mark all
  read". Read-state is per-browser — say so plainly in the footer, so nobody
  expects it to follow them to their phone.
- Filters for tier, tag, source, and free-text search, **reflected in the URL
  query string** so a filtered view can be bookmarked and shared.
- Self-contained: inline CSS and JS, no external requests, no CDN. Theme-aware
  for light and dark.

Keep the presentation layer honest about provenance: show the tier, the tags,
and whether an item was LLM-ranked or fell back to deterministic scoring.

### Cohort-readiness (structure only, do not build the UI)

The default profile is tuned to one person: history, career-changer, ITT. If
this is later shared with a cohort, other trainees need different orderings.

Prepare for that without building it: keep the scoring profile entirely in
config, and expose per-item `tags` and the `scoring` component breakdown in the
API. That is enough for a future client-side re-ranker to reorder in the
browser from the same static files. **Do not build a subject picker now.**

---

## The read-only guarantee

Requirement 4 is a security property, so make it verifiable rather than
assumed:

- The published page makes **no network requests** other than for its own
  static assets. No calls to `api.github.com`, no third-party analytics, no
  remote fonts.
- No token, PAT, or secret appears in any file under `docs/`.
- The workflow grants `permissions: contents: write` and nothing else.
- `workflow_dispatch` stays enabled for convenience — GitHub only allows users
  with repository write access to trigger it, so it is not a public surface —
  but note this explicitly in the README so the reasoning survives.
- Add a test that greps the built `docs/` tree for `api.github.com`, `fetch(`
  against non-relative URLs, and token-shaped strings, and fails the build if
  any appear.

---

## Platform constraints

Carry these forward from the proof of concept; they are all load-bearing.

- **GitHub cron is UTC and DST-blind.** Schedule both `0 6 * * 1,4` and
  `0 7 * * 1,4`, with a guard step that proceeds only when `TZ=Europe/London
  date +%H` is `07`, or when the run is a manual dispatch.
- **Scheduled workflows are disabled after 60 days without commits.** Each run
  commits, which resets the clock. This is why the corpus and briefs are
  committed rather than only published.
- **Actions needs write permission**, set at Settings → Actions → General →
  Workflow permissions. Without it the build succeeds and the push 403s.
- **Pages limits:** 1 GB site, 100 GB/month soft bandwidth, 10-minute deploy
  timeout. A 14-day corpus is a few hundred kilobytes; archives accumulate
  slowly. Not a concern, but keep the archive JSON minified.
- **Tes has no public RSS.** Ship it disabled with an empty URL and the
  newsletter-to-feed workaround documented, exactly as the PoC does. Do not
  guess at a URL.

---

## Tag vocabulary

Tags come from the Tag Index in the owner's filing convention, so a tag in the
brief is the same string as the tag in their Zotero library. Preserve this
exactly. Two tags in `config/scoring.yml` are marked `NEW` —
`prof-routes-into-teaching` and `ped-ai` — and are pending addition to the Tag
Index. Do not invent further tags without marking them the same way. The
`disc-` set is a fixed canon; do not extend it.

---

## Reuse and replace

**Keep, largely as-is:** `config/feeds.yml` and `config/scoring.yml` — the
vocabulary and weights are the accumulated value here. The word-boundary term
matching in `score.py` (it correctly stops `ect` matching "collected"). The
feed normalisation and `id` derivation in `fetch.py`. The offline fixture
approach in `tests/`.

**Replace:** `state/seen.json` with the corpus. `render.py` — it renders
snapshots, not a rolling list, and emits no API. `llm.py` — its batch-relative
normalisation is exactly the bug described above. `main.py` — the stage order
changes.

---

## Acceptance criteria

Build tests for these. They should run offline against fixtures, as the PoC's
do, with no network access.

1. An item first seen on day 0 is present in `items.json` on day 13 and absent
   on day 15.
2. An item present in the corpus is not re-scored on a subsequent run: its
   `relevance` and `scoring.llm` are unchanged, and no LLM call is made for it.
3. Two items ingested in different runs with the same deterministic raw score
   receive the same `deterministic_norm`. (This is the batch-relative bug.)
4. `rank_score` decreases as an item ages while `relevance` stays fixed.
5. With no `ANTHROPIC_API_KEY`, the build completes, every item carries
   `"mode": "deterministic"`, and `meta.json` reports the degradation.
6. When the LLM returns malformed JSON, or omits some requested ids, the build
   completes and the affected items fall back cleanly.
7. A feed returning a 500 is reported in `meta.json` and the footer, and does
   not fail the build.
8. The same story from two overlapping feeds produces one item.
9. A run finding no new items leaves the existing site and corpus intact rather
   than publishing an empty page.
10. `docs/` contains no `api.github.com`, no external `fetch(`, and no
    token-shaped strings.
11. Every API file parses as JSON and matches its documented shape;
    `index.json` lists every endpoint actually emitted.
12. The page renders correctly when `localStorage` throws on access.

---

## Non-goals

Do not build: user accounts, a server or database, a sync backend, comments,
push notifications, a subject picker, scraping of paywalled content, or any
write path from the site to the repository.

---

## Working style

Plan first and show the plan. Build in stages — corpus and pipeline, then API,
then site — running the tests at each stage. Keep configuration in YAML, not in
Python. Comment the non-obvious decisions, especially the absolute-scoring one,
because the person maintaining this is learning the codebase as they go.
