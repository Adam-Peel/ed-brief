"""Static JSON API under docs/api/v1/, served by GitHub Pages.

Every file here is fully pre-computed at build time. Public Pages sites send
access-control-allow-origin: *, so a mobile app can fetch these cross-origin
with no server-side configuration; Pages doesn't support custom response
headers, so a client diffs on the `generated` timestamp rather than an ETag.

v1 is a contract from here on: additive changes only. Bump the top-level
`schema` and the path (v2) for anything that isn't additive.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .corpus import SCHEMA, CorpusItem, format_dt


def _dump(obj: dict) -> str:
    # Minified: the archive accumulates one file per calendar day, retained
    # indefinitely, so keeping every API file compact costs nothing and adds
    # up over a year of runs.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_all(
    live: list[CorpusItem],
    new_items: list[CorpusItem],
    today_items: list[CorpusItem],
    meta: dict,
    root: Path,
    now: datetime,
) -> None:
    api_dir = root / "docs" / "api" / "v1"
    archive_dir = api_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    generated = format_dt(now)

    def emit(relpath: str, obj: dict) -> None:
        (api_dir / relpath).write_text(_dump(obj), encoding="utf-8")

    emit(
        "items.json",
        {
            "schema": SCHEMA,
            "generated": generated,
            "count": len(live),
            "items": [item.to_dict() for item in live],
        },
    )
    emit(
        "latest.json",
        {
            "schema": SCHEMA,
            "generated": generated,
            "count": len(new_items),
            "items": [item.to_dict() for item in new_items],
        },
    )
    emit("meta.json", meta)

    tag_counts: dict[str, int] = {}
    for item in live:
        for tag in item.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    emit(
        "tags.json",
        {
            "schema": SCHEMA,
            "generated": generated,
            "tags": [
                {"tag": tag, "count": count}
                for tag, count in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
        },
    )

    emit(
        f"archive/{now:%Y-%m-%d}.json",
        {
            "schema": SCHEMA,
            "date": now.strftime("%Y-%m-%d"),
            "count": len(today_items),
            "items": [item.to_dict() for item in today_items],
        },
    )

    # Built by walking what's actually on disk, not hardcoded, so this can
    # never drift from what was really emitted (BUILD-SPEC.md acceptance
    # criterion 11) -- and it picks up every previously-archived date, not
    # just today's, since archives are retained indefinitely.
    endpoints = sorted(
        "/api/v1/" + p.relative_to(api_dir).as_posix()
        for p in api_dir.rglob("*.json")
        if p.name != "index.json"
    )
    index_obj = {
        "schema": SCHEMA,
        "generated": generated,
        "endpoints": ["/api/v1/index.json", *endpoints],
    }
    (api_dir / "index.json").write_text(_dump(index_obj), encoding="utf-8")
