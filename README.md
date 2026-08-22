# ed-brief

A twice-daily education news brief, ranked for someone moving into secondary
history teaching in England.

Runs at 07:00 and 19:00 London time via GitHub Actions, aimed at the two
commutes -- 07:00 catches the overnight trickle before the morning journey,
19:00 catches the day's main volume (measured at ~91% of daily items from
the real feed data) before the evening one. Pulls the feeds, scores every
story against a vocabulary you control, and publishes a rolling, filterable
list to GitHub Pages — plus a public read-only JSON API and a dated markdown
copy committed to `briefs/`, so you have a searchable archive that outlives
any hosting decision.

Every story is pre-scored and pre-ranked at ingest time, before anything is
published. The site and API are static output: nothing is fetched, scored, or
sorted when you load the page. Items stay in the list for 14 days from when
they're first seen, or until you mark them read.

No newsletters. No inbox. A rolling list you can skim in five minutes.

---

## Setup

Five steps, about ten minutes.

### 1. Create the repository

```bash
cd ed-brief
git init -b main
git add .
git commit -m "Initial commit"
gh repo create ed-brief --public --source=. --push
```

No `gh`? Create an empty public repo on github.com, then:

```bash
git remote add origin https://github.com/YOUR-USERNAME/ed-brief.git
git push -u origin main
```

### 2. Let the workflow write back to the repo

**Settings → Actions → General → Workflow permissions → Read and write
permissions → Save.**

This is the step people miss. Without it the run succeeds, produces the
brief, and then fails on the final push with a 403.

### 3. Turn on Pages

**Settings → Pages → Source: Deploy from a branch → Branch: `main`, folder:
`/docs` → Save.**

Your site appears at `https://YOUR-USERNAME.github.io/ed-brief/` within a
minute or two of the first run being committed.

### 4. Run it once by hand

**Actions → Build education brief → Run workflow.**

Manual runs skip the time guard, so you don't have to wait for the next
scheduled 07:00 or 19:00. Watch the log: the "Check feed health" step
reports any source that has
moved or died, which is the single most useful thing to glance at on a first
run.

### 5. Check the result

The page should be live, `data/corpus.json` should exist in the repo, and
`briefs/YYYY-MM-DD.md` should exist alongside it. If the top tier looks
wrong, go straight to [tuning](#tuning-the-ranking) — the defaults are a
starting guess about what matters to you, not a finished answer.

---

## How ranking works

Every story gets a **deterministic score** — the sum of a source weight
(`config/feeds.yml`), matched topic groups (`config/scoring.yml`), a recency
bonus, and any mute penalties — normalised onto a fixed 0–10 scale. If an
`ANTHROPIC_API_KEY` is configured, a **two-stage LLM pass** also judges the
story:

1. **Classify** (`src/classify.py`) triages every new story into a type --
   CURRICULUM, HISTORY, PEDAGOGY, PUPILS, CAREER, SECTOR, OTHER, or
   IRRELEVANT (discarded, unscored) -- plus a 0–4 Nottinghamshire locality
   score.
2. **Score** (`src/llm.py`) batches surviving stories by type and scores each
   against universals (how much changes, how easy to discover) plus that
   type's own small Likert item bank (never against the other stories in the
   batch — see [below](#why-the-scoring-has-to-be-absolute)).

The two stages produce one 0–10 LLM score, which is then blended with the
deterministic one (60% LLM, 40% deterministic by default) into a single
**relevance** score, 0–10.

That blend happens exactly once, when a story is first seen, and is then
**frozen forever** — it cost money to compute and the significance of a story
doesn't change. What you see ranked highest on a given day is relevance minus
a small, ever-growing age penalty (`rank_score`), recomputed for the whole
list on every run so fresher stories edge out equally-relevant stale ones
without ever re-touching the frozen score. A Nottinghamshire-eligible story
also gets a **percentile floor** on `rank_score` at this same recompute step
(see `locality` in `config/scoring.yml`) -- not summed into relevance, so
locality can surface a story without ever letting it outrank real curriculum
or career news on its own.

Stories land in four tiers — **Read these**, **Worth a look**, **Everything
else**, **Low relevance** — set by fixed thresholds on the relevance scale
(`tiers` in `config/scoring.yml`). The site organises stories primarily by
**type**, not tier -- Curriculum, History, Pupils, Pedagogy, Career, Sector
(background, collapsed by default) -- with tier staying as a filter and
Nottinghamshire as a cross-cutting section.

A topic group scores **once** no matter how many of its terms match, so an
article that repeats a keyword twenty times gains nothing over one that says
it once. A match in the **headline** is worth 1.6× a match in the summary,
because headlines are where publications put what a story is actually about.

### Why the scoring has to be absolute

An earlier, snapshot-only version of this brief normalised the deterministic
score against the min/max of whatever ran that day, and asked the LLM to rank
stories against each other. Both are fine for a one-off page and both are
wrong for a rolling list: today's batch and tomorrow's batch are different
batches, so the same story could land at a different point on the scale
depending on what else happened to be scored alongside it that day. A
14-day rolling list needs everything on one fixed scale, so:

- The deterministic score is normalised against a **fixed ceiling**
  (`deterministic_scale_max` in config), not the current run's own min/max.
- The LLM scores each story on a fixed 0–4 Likert scale per item (0 = not at
  all, 4 = entirely), explicitly told never to compare stories to each
  other and never to let a batch of similar stories (they're grouped by
  type) push scores up on that basis alone.

You can read exactly why anything ranked where it did in `deterministic_raw`
and the tags surfaced per item — everything is in `data/corpus.json` and the
API.

---

## The public API

Every ranked item is available as static JSON under `docs/api/v1/`, so a
future mobile client (or anything else) can be built against it without
touching this repo. It's fully public and read-only — the API is just build
output.

| Endpoint | Contents |
|---|---|
| `/api/v1/index.json` | Discovery document: schema version, every endpoint actually emitted, `generated` |
| `/api/v1/items.json` | Every live (unexpired) item, already ranked |
| `/api/v1/latest.json` | Only items added by the most recent run |
| `/api/v1/meta.json` | Run metadata: per-source health, counts, scoring mode, retention |
| `/api/v1/tags.json` | Every tag currently in use, with item counts |
| `/api/v1/archive/YYYY-MM-DD.json` | That run's new items, retained indefinitely |

`v1` is a contract from here on: changes within it are additive only. GitHub
Pages sends `access-control-allow-origin: *` on public sites, so cross-origin
fetches work with no server-side configuration — but Pages doesn't support
custom response headers, so a client should diff on the `generated` timestamp
rather than expect an ETag.

Each item's `tags` and `scoring` breakdown (`deterministic_raw`,
`deterministic_norm`, `llm`) are exposed specifically so a future client
could re-rank the same static data for a different reader profile (e.g. a
different subject or career stage) without this repo needing a subject
picker or per-user config.

---

## The RSS feed

`docs/feed.xml` — a standard RSS 2.0 feed, for reading in Feeder,
NetNewsWire, Feedly, or any other feed reader, the same way you'd read any
other source. The site's `<head>` advertises it via the normal
`<link rel="alternate">` autodiscovery tag, so most readers pick it up from
just the site URL; otherwise point a reader directly at
`https://YOUR-USERNAME.github.io/ed-brief/feed.xml`.

This is deliberately a different thing from the JSON API above: the API is a
bespoke contract for a future purpose-built client that can be taught its
exact fields, RSS is for interoperating with reader software this repo
doesn't control. Static output either way — a reader polling `feed.xml` is
an ordinary file fetch, generated fresh at every build like everything else.

The feed only includes **"Read these" and "Worth a look"** items, not the
full corpus — the same bar the site itself defaults to showing. Everything
else is still reachable through the JSON API; a reader's inbox is a worse
place than a browsable list for items that are borderline by design. Each
entry's description carries the LLM's "why this matters" (when there is
one), the summary, the tier, and the score, so you don't lose that context
just because you're reading it somewhere else.

---

## Saving to Zotero

Every card has an **Add to Zotero** link. If you have the [Zotero
Connector](https://www.zotero.org/download/) browser extension installed,
clicking it opens the item's `docs/ris/{id}.ris` file, which the Connector
intercepts and offers to import directly — carrying the item's Tag Index
tags (`scoring.yml`'s `topics:` vocabulary, the same strings used in
`tags.html`) onto the saved item as real Zotero tags, not just the bare
article link a normal "save this page" would give you. Without the
Connector, the link still downloads a plain, valid RIS file.

This is what turns the feed into something you can actually keep: items
here disappear after `retention_days` (14 by default) or once you mark
them read, so anything worth citing later needs saving somewhere durable —
Zotero already IS that somewhere, if you use it, and this skips retyping
the tags by hand.

Field mappings (which RIS tag becomes which Zotero field, `TY` as `ELEC`
for Zotero's own "Web Page" type, `KW` per tag rather than comma-separated)
are verified against
[Zotero's own RIS translator source](https://github.com/zotero/translators/blob/master/RIS.js),
not just the general RIS spec — see `src/ris.py`'s module docstring.

---

## Weekly digest (unlisted)

The Sunday run (London time) that has an `ANTHROPIC_API_KEY` writes
`docs/digest.html`: a flowing prose round-up of the last 7 days — written
for *listening*, not reading, since the eventual point is a script for a
weekly audio briefing. Started out daily and scoped to just that day's new
items; real data killed that design in one afternoon — a quiet Saturday
had six new items, the best scoring below this project's own "worth a
look" bar. Not enough material, and not good enough material either.
Weekly fixes both, and incidentally fixes a second problem for free:
today-scoped digests never covered anything not first-seen on the exact
day they ran, so a slow trickle of arrivals could sit uncovered forever.
A 7-day window covers everything from the last week regardless of which
specific day it arrived, with no separate tracking needed, since the
coverage window and the episode cadence are the same period.

Content is deliberately gated, not "everything new": full coverage for
anything that cleared `scoring.yml`'s `worth` tier cut (read from config,
so it can't drift out of sync with the site's own bar), plus a brief
mention for any HISTORY-typed item that didn't clear that bar — history
content matters to this reader even when the general rubric doesn't rate
it highly. Everything else below the bar is left out entirely, not
grouped into a passing mention.

Deliberately **not linked from anywhere on the published site** —
`index.html`, `archive.html`, the API endpoint index all omit it entirely
— and marked `noindex` so it won't turn up in search results either.
Reachable only if you already have the URL:
`https://YOUR-USERNAME.github.io/ed-brief/digest.html`. It's a single
always-current page, not a dated archive (yet) — regenerated each Sunday
that has qualifying items and a key, silently skipped otherwise (including
every non-Sunday run, by design).

Text-to-speech isn't wired up yet — Claude doesn't do TTS natively (even
Anthropic's own consumer Voice Mode uses ElevenLabs under the hood, per
their own terms of service), so that would mean a separate vendor
(Google Cloud, Amazon Polly, OpenAI, or ElevenLabs, roughly $0.50–$14/month
at this volume depending on voice quality) and a podcast RSS feed on top
of this. The prose digest here is the first step toward that, usable on
its own in the meantime.

---

## The read-only guarantee

The published page and API are load-bearing on being **strictly read-only**:
there's no path from the site back to this repository, which is what
protects you from a bug or a bad actor burning through your Actions minutes
or API spend. Concretely:

- The page makes no network requests other than for its own static assets —
  no calls to `api.github.com`, no analytics, no remote fonts. `docs/index.html`
  embeds its item list inline rather than fetching it, so it also works
  opened directly from disk.
- No token or secret appears anywhere under `docs/`.
- The workflow's `permissions:` block grants `contents: write` and nothing
  else.
- `workflow_dispatch` (the manual "Run workflow" button) stays enabled —
  GitHub only allows people with write access to this repository to trigger
  it, so it isn't a public surface, just a convenience for you.
- `tests/test_readonly.py` greps the entire built `docs/` tree for
  `api.github.com`, non-relative `fetch(` calls, and token-shaped strings,
  and fails if it finds any. It runs in CI on every push.

---

## Tuning the ranking

Everything lives in `config/feeds.yml` and `config/scoring.yml`. No Python to
touch.

**A source is too noisy** — lower its `weight` in `config/feeds.yml`, or set
`enabled: false`.

**A subject keeps getting missed** — add terms to the relevant group in
`config/scoring.yml`. Terms match on word boundaries, so `ect` matches "ECT
mentors" but not "collected". Stems work: `decolonis` catches
"decolonising".

**The top tier is crowded** — lower `tiers.lead`. Too sparse — raise it.
Both `tiers.lead` and `tiers.worth` are thresholds on the final 0–10
`relevance` scale, not the raw deterministic sum.

**The LLM and the keyword weights disagree too much, or too little** — adjust
`llm_weight` (default 0.8). Higher trusts the LLM's judgement more; lower
leans on your own weights.

**A different model for the LLM pass** — change `llm_model` (default
`claude-sonnet-5`). `BRIEF_LLM_MODEL` overrides it for a one-off local test
without touching the config.

**What each tag actually means** — `/tags.html` on the published site lists
every tag from `config/scoring.yml`'s `description` field alongside its live
item count; that field is the single source of truth for the glossary, so
editing a topic's description there keeps the page in sync automatically.

**Something keeps appearing that you never want** — add it to a `mutes`
group in `config/scoring.yml`. Muting sinks an item rather than deleting it,
so you can still find it under "Everything else" if you were wrong.

**Items should stick around longer or shorter** — change `retention_days`.

---

## Adding Tes

Tes retired its public RSS feed; the old `/magazine/rss.xml` now serves a
migration notice, which is why `config/feeds.yml` ships it disabled with an
empty URL rather than a guess that would fail silently.

Two ways to get it back:

1. **Newsletter to feed.** Subscribe to a Tes newsletter using an address
   from [Kill the Newsletter](https://kill-the-newsletter.com/) or Inoreader.
   Both hand you a feed URL for the resulting emails. Paste it into the `tes`
   entry and set `enabled: true`. This also works for any other newsletter
   you'd rather read here than in your inbox.

2. **An RSS bridge.** Point the entry at an RSS-Bridge instance scraping
   `tes.com/magazine/news`. More fragile, no signup.

Either way, run `python -m src.validate` afterwards to confirm it resolves.

---

## Turning on LLM re-ranking

Off by default. It adds an absolute relevance judgement and a one-line "why
this matters" to each new story, which the deterministic scorer can't
produce on its own.

1. Get an API key from [console.anthropic.com](https://console.anthropic.com).
   A non-expiring key is fine here: it's stored in a real secrets manager
   (below) and used by an unattended schedule, so an expiring key would just
   risk silently degrading back to deterministic-only scoring if nobody
   happened to notice and rotate it in time.
2. **Settings → Secrets and variables → Actions → New repository secret**,
   named `ANTHROPIC_API_KEY`. The workflow already reads it — nothing else
   to uncomment.

Cost is a few pence a month at two runs a day, since only *new* items are
ever sent to the model — existing items keep their frozen score forever.

Every failure path falls back to deterministic scoring for exactly the
affected items and says so in `data/corpus.json` (`scoring.mode`),
`docs/api/v1/meta.json`, and the page footer: no key, missing package, or a
network/API/schema-validation error that fails a whole batch. Both LLM
stages use the Claude API's structured outputs (`output_config`'s
`json_schema`) to pin each batch's response to exactly one verdict per item
it was sent, so a response that would omit or invent an item is
schema-invalid rather than merely against instructions — a single bad
response degrades that one batch, not the whole run, and never fails the
build.

---

## Running locally

```bash
pip install -r requirements.txt

python -m src.build            # fetches real feeds, writes data/docs/briefs
python -m src.validate         # check every feed URL, enabled or not
python tests/test_pipeline.py  # offline unit/acceptance tests against fixtures
python tests/test_readonly.py  # offline end-to-end build + read-only/API checks
```

`python -m src.build` hits the real internet and, unlike the tests, really
does update `data/corpus.json` — that's what makes it a faithful local
preview, but it also means running it twice will show "0 new since last run"
the second time. To reset and try again: `git checkout -- data docs briefs`
(once you've committed a clean state), or just delete `data/corpus.json` to
start from an empty corpus.

Open `docs/index.html` in a browser to see the result — it's self-contained,
so this works straight off disk.

---

## Layout

```
config/feeds.yml         sources, weights, on/off
config/scoring.yml       topic groups, weights, mutes, tier cuts, scoring scale
src/fetch.py             HTTP + feed normalisation, item identity (id derivation)
src/score.py             deterministic scorer + fixed-ceiling normalisation
src/classify.py          stage 1: type + locality triage, batched, dormant without a key
src/llm.py               stage 2: typed Likert scoring, batched, dormant without a key
src/corpus.py            the rolling corpus: load/save/expire/rank/locality floor
src/api.py               docs/api/v1/*.json writer
src/site.py              docs/index.html, docs/archive.html, dated snapshots
src/ris.py               docs/ris/{id}.ris writer -- one per item, for Add to Zotero
src/rss.py               docs/feed.xml writer (lead/worth tiers only)
src/brief.py             briefs/YYYY-MM-DD.md writer
src/digest.py            docs/digest.html writer -- unlisted weekly prose round-up
src/build.py             entry point: fetch, dedupe, score, rank, publish
src/validate.py          feed health check
data/corpus.json         the single source of truth (committed)
briefs/                  dated markdown archive (committed)
docs/                    the published site + API (committed, served by Pages)
tests/                   offline fixtures and acceptance-criteria tests
```

---

## Things worth knowing

**Identity is by URL, not headline.** Query strings are stripped, so the
same story arriving with different tracking parameters is recognised as one
item, and its id is stable for the item's whole 14-day life (and forever in
the archive). The derivation only ever changes alongside a `schema` bump in
`data/corpus.json`.

**The schedule keeps itself alive.** GitHub disables scheduled workflows in
repos with no commits for 60 days. Because `rank_score` is recomputed for the
whole corpus on every run, `data/corpus.json`'s `generated` timestamp changes
even on a run with zero new stories — so every run produces a commit, and the
clock resets every day regardless of how quiet the feeds are.

**A quiet run never publishes an empty page.** Unlike a snapshot-per-run
design, the rolling corpus means there's always something to show as long as
it has ever had one item in it — a run with nothing new just republishes the
same list with a fresher `generated` timestamp and slightly decayed
`rank_score`s.

**Dead feeds degrade rather than break.** A source that stops responding is
reported in the workflow log, in `docs/api/v1/meta.json`, and in a collapsed
note in the page footer. It never fails the run. Check that footer
occasionally — a feed can rot quietly and the only symptom is fewer new
items.

**Read state is per-browser.** It lives in `localStorage`, behind a small
`ReadStore` interface (`src/site.py`) so a sync backend could replace it
later without touching the rest of the site. It does not follow you to
another device, and the footer says so.
