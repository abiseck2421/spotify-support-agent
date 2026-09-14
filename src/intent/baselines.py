"""Baselines for intent classification.

Baseline 1 (trivial): always predict the majority class from training labels.
Baseline 2 (keyword/regex): hand-written rules derived from the taxonomy
seed keywords. This is explicitly the DUMB baseline — it is never the final
classifier (that is src/intent/train.py). Matching order = taxonomy order,
so playback wins over downloads_offline when both hit (labels are priority
ordered in the taxonomy module).
"""

from __future__ import annotations

import re

from src.intent.taxonomy import INTENTS, LABELS


def keyword_intent(text: str) -> str:
    """Baseline-2 rule tagger. First intent whose seed keywords match."""
    t = (text or "").lower()
    for intent in INTENTS:
        if re.search(intent.seed_keywords, t):
            return intent.label
    return "other"  # taxonomy's last entry also matches some praise; fallback


class TrivialBaseline:
    """Predicts the single most frequent label seen in training."""

    def __init__(self) -> None:
        self.majority: str | None = None

    def fit(self, labels: list[str]) -> None:
        from collections import Counter

        self.majority = Counter(labels).most_common(1)[0][0]

    def predict(self, texts: list[str]) -> list[str]:
        assert self.majority is not None, "call fit() first"
        return [self.majority] * len(texts)


__all__ = ["TrivialBaseline", "keyword_intent", "LABELS"]