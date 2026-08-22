"""Generates one static .ris file per published item (docs/ris/{id}.ris),
so a card's "Add to Zotero" link can hand the reader's browser a real
bibliographic record instead of just the bare article URL -- this is what
carries the Tag Index vocabulary (scoring.yml's `topics:`, already synced
with Zotero per the tags.html page) onto the saved item as real Zotero
tags, rather than the reader having to add them by hand after the fact.

Field mappings and structural rules below are verified against Zotero's
own RIS import/export translator source
(https://github.com/zotero/translators/blob/master/RIS.js), not just the
general RIS spec, since a generic-but-technically-valid RIS file is not
the same thing as one Zotero's own parser handles the way intended -- see
the note on TY below for why that distinction matters here. Checked
2026-08-22, after the structured-outputs incident made "verify against
what's actually there, not training-data recall" a hard rule for this
project.

Deliberately flat, one record per file, no batching: this is unrelated to
the classify/score pipeline's own batching concerns (LLM cost, grammar
size) -- these are just static text files, and Zotero's connector expects
one file per save action anyway.
"""

from __future__ import annotations

from pathlib import Path

from .corpus import CorpusItem

# Zotero's translator canonically pairs its "Web Page" item type with TY
# "ELEC" on EXPORT (the round-trip value Zotero itself produces); "WEB" is
# only recognised on import as an EndNote-compatibility shim, "not in spec"
# per the translator's own comment. ELEC is the correct choice for a file
# WE generate for Zotero to import, not the lenient fallback. Every ed-
# brief source -- newspaper, government release, personal blog -- becomes
# "Web Page" uniformly rather than guessing a more specific type (BLOG,
# NEWS, ...) per source: it's the one type that's honestly accurate for
# all of them (a link to a web page), and Zotero's Web Page fields (Title,
# Website Title, Date, URL) already match what ed-brief actually has.
_TY = "ELEC"

# RIS requires CRLF line endings (confirmed in Zotero's own translator:
# newLineChar = "\r\n") -- writing with plain "\n" is a common enough
# mistake across RIS generators that it's worth this being impossible to
# get wrong by accident: every line is built through this, never a bare
# f-string with an assumed newline.
_EOL = "\r\n"


def _line(tag: str, value: str) -> str:
    """One RIS tag line. Flattened to a single line always, even for
    fields (AB, KW, ...) that RIS's continuation-line rules would let span
    multiple lines -- ed-brief's summaries are already short single-
    paragraph RSS text, so nothing real is lost, and it removes an entire
    class of "did I reproduce the continuation rule correctly" risk from a
    file format with no escaping mechanism of its own."""
    flat = " ".join(value.split())
    return f"{tag}  - {flat}"


def _record(item: CorpusItem) -> str:
    lines = [
        _line("TY", _TY),
        _line("TI", item.title),
        _line("T2", item.source_name),
        _line("UR", item.url),
        _line("PY", item.published.strftime("%Y")),
        _line("DA", item.published.strftime("%Y/%m/%d")),
    ]
    if item.summary:
        lines.append(_line("AB", item.summary))
    # KW maps directly to Zotero's tags array on import (confirmed in the
    # translator source) -- one KW line per tag, not comma-separated on a
    # single line, since Zotero's own splitting logic treats commas within
    # one KW value as "more problematic" than a clean line-per-tag.
    lines.extend(_line("KW", tag) for tag in item.tags)
    if item.why:
        lines.append(_line("N1", f"ed-brief ({item.relevance:.1f}/10): {item.why}"))
    lines.append("ER  - ")
    return _EOL.join(lines) + _EOL


def write_ris_files(items: list[CorpusItem], docs_dir: Path) -> None:
    """One file per item in `items` (the same live/published set the site
    itself renders cards for), at docs/ris/{id}.ris. Regenerated in full
    every build like every other output here, but content is purely a
    function of each item's own frozen fields -- unchanged items produce
    byte-identical files, so this doesn't manufacture a diff on a quiet
    run the way a timestamp in the content would."""
    ris_dir = docs_dir / "ris"
    ris_dir.mkdir(parents=True, exist_ok=True)
    live_ids = {item.id for item in items}
    for item in items:
        # newline="" -- otherwise Python's universal-newline text-mode
        # writing would translate the CRLFs already in the record back
        # down to the platform default, undoing the point of _EOL.
        (ris_dir / f"{item.id}.ris").write_text(_record(item), encoding="utf-8", newline="")
    # Prunes files for items that have expired or dropped below
    # publish_floor since the last build -- otherwise docs/ris/ would only
    # ever grow, serving "Add to Zotero" links for articles no longer on
    # the page at all.
    for path in ris_dir.glob("*.ris"):
        if path.stem not in live_ids:
            path.unlink()
