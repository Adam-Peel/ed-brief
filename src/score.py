"""Deterministic relevance scoring against the Tag Index vocabulary."""

from __future__ import annotations

import math
import re

from .fetch import Item


def _compile(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    """Compile each term to a word-boundary-ish pattern.

    Plain substring matching produces embarrassing false positives — "ect"
    inside "collected", "ai" inside "said". Anchoring on non-word characters
    fixes that while still matching multi-word phrases and letting prefixes
    like "decolonis" catch their own suffixes.
    """
    compiled = []
    for term in terms:
        escaped = re.escape(term.lower().strip())
        # Trailing word boundary is omitted for stem-style terms (no trailing
        # space and clearly truncated), so "decolonis" matches "decolonising".
        pattern = re.compile(rf"(?<!\w){escaped}", re.IGNORECASE)
        compiled.append((term, pattern))
    return compiled


class Scorer:
    def __init__(self, cfg: dict):
        self.title_multiplier = float(cfg.get("title_multiplier", 1.5))
        recency = cfg.get("recency", {})
        self.half_life = float(recency.get("half_life_hours", 60))
        self.max_recency_bonus = float(recency.get("max_bonus", 2.5))
        self.scale_max = float(cfg.get("deterministic_scale_max", 25.0))

        self.topics = [
            {
                "tag": group["tag"],
                "weight": float(group.get("weight", 1.0)),
                "patterns": _compile(group.get("terms", [])),
            }
            for group in cfg.get("topics", [])
        ]
        self.mutes = [
            {
                "name": group["name"],
                "weight": float(group.get("weight", -3.0)),
                "patterns": _compile(group.get("terms", [])),
            }
            for group in cfg.get("mutes", [])
        ]

    def _recency_bonus(self, age_hours: float) -> float:
        """Exponential decay, so a three-day-old story is worth about half."""
        if self.half_life <= 0:
            return 0.0
        return self.max_recency_bonus * math.exp(
            -math.log(2) * age_hours / self.half_life
        )

    @staticmethod
    def _first_match(
        patterns: list[tuple[str, re.Pattern]], title: str, body: str
    ) -> tuple[bool, bool, str]:
        """Does this group match, was it in the title, and on which term?

        Scans for a title hit in preference to a body hit, because a title
        match earns the multiplier and we want the best available evidence
        rather than whichever term happens to come first in the config.
        """
        body_hit: str | None = None
        for term, pattern in patterns:
            if pattern.search(title):
                return True, True, term
            if body_hit is None and pattern.search(body):
                body_hit = term
        if body_hit:
            return True, False, body_hit
        return False, False, ""

    def score_item(self, item: Item) -> Item:
        """Compute the raw deterministic score and stamp tags/reasons on the
        item. Returns the item for convenience; the caller reads
        `item.deterministic_raw`. Kept separate from `normalize()` so
        normalisation stays a pure function of one number, independent of
        anything else in the batch."""
        title, body = item.haystack()
        total = item.source_weight
        tags: list[str] = []
        reasons: list[str] = []

        if item.source_weight:
            reasons.append(f"source {item.source_name} ({item.source_weight:+.1f})")

        for group in self.topics:
            matched, in_title, term = self._first_match(
                group["patterns"], title, body
            )
            if not matched:
                continue
            weight = group["weight"]
            if in_title:
                weight *= self.title_multiplier
            total += weight
            tags.append(group["tag"])
            where = "title" if in_title else "body"
            reasons.append(f"{group['tag']} via “{term}” in {where} ({weight:+.1f})")

        for group in self.mutes:
            matched, _, term = self._first_match(group["patterns"], title, body)
            if matched:
                total += group["weight"]
                reasons.append(
                    f"muted: {group['name']} via “{term}” ({group['weight']:+.1f})"
                )

        bonus = self._recency_bonus(item.age_hours)
        total += bonus
        reasons.append(f"recency ({bonus:+.1f})")

        item.tags = tags
        item.reasons = reasons
        item.deterministic_raw = total
        return item

    def normalize(self, raw: float) -> float:
        """Fixed-ceiling normalisation onto 0-10.

        The proof of concept normalised against the min/max of the current
        run's batch, which is fine for a one-off snapshot but wrong for a
        rolling corpus: items scored in different runs would sit on different
        scales and couldn't be compared in one list (BUILD-SPEC.md calls this
        out as the subtlest bug to avoid). A fixed ceiling from config means
        the same raw score always normalises to the same value, this run or
        any other. Negative raw scores (heavily muted items) are floored at 0
        rather than allowed to go negative post-normalisation.
        """
        return max(0.0, min(raw, self.scale_max)) / self.scale_max * 10.0


def blend_relevance(
    deterministic_norm: float, llm_score: float | None, llm_weight: float
) -> float:
    """Combine the two 0-10 signals into the frozen `relevance` value.

    Deterministic-only (no LLM verdict for this item) uses deterministic_norm
    directly rather than blending with a missing value.
    """
    if llm_score is None:
        return deterministic_norm
    return (1 - llm_weight) * deterministic_norm + llm_weight * llm_score
