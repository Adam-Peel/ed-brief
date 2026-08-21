"""Check every configured feed URL, including the disabled ones.

    python -m src.validate

Run this after editing config/feeds.yml, or any time the brief looks thin —
publishers move and retire feeds without notice, and a rotted URL otherwise
just means quietly fewer stories.
"""

from __future__ import annotations

import sys

import feedparser
import requests

from .build import load_config
from .fetch import TIMEOUT, USER_AGENT


def check(url: str) -> tuple[bool, str]:
    if not url:
        return False, "no URL set"
    try:
        response = requests.get(
            url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException as exc:
        return False, f"{type(exc).__name__}"

    if response.status_code != 200:
        return False, f"HTTP {response.status_code}"

    parsed = feedparser.parse(response.content)
    if not parsed.entries:
        return False, "parsed but empty — probably not a feed"

    title = parsed.feed.get("title", "untitled")
    return True, f"{len(parsed.entries)} entries — “{title}”"


def main() -> int:
    feeds, _ = load_config()
    failures = 0

    for feed in feeds:
        state = "on " if feed.get("enabled", True) else "off"
        ok, detail = check(feed.get("url", ""))
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] [{state}] {feed['name']:<28} {detail}")
        if not ok and feed.get("enabled", True):
            failures += 1

    if failures:
        print(f"\n{failures} enabled feed(s) failing.", file=sys.stderr)
    else:
        print("\nAll enabled feeds reachable.")
    return 0  # never fail the build over a flaky publisher


if __name__ == "__main__":
    raise SystemExit(main())
