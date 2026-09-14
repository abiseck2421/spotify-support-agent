"""Intent-module sanity & regression tests.

Runnable two ways: `python tests/test_intent.py` (plain asserts) or pytest.

Key regression: the Day-1 'car inside SpotifyCares' bug. Every taxonomny
regex must be word-boundary safe.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.baselines import TrivialBaseline, keyword_intent
from src.intent.taxonomy import INTENTS, LABELS, normalize_label


def test_no_self_match_on_brand_handle():
    # real code lowercases before matching, so the regression is vs lowercased text.
    # 'spotifycares' ends with 'cares' -> 'car' must NOT match (needs a \b after it).
    handle = "spotifycares"
    for intent in INTENTS:
        if intent.label == "other":
            continue
        assert re.search(intent.seed_keywords, handle) is None, (
            f"{intent.label} matched inside the brand handle"
        )


def test_word_boundary_traps():
    assert re.search(INTENTS[5].seed_keywords, "business") is None          # 'ui' trap
    assert re.search(INTENTS[0].seed_keywords, "count") is None             # 'count' vs 'account'
    assert re.search(INTENTS[7].seed_keywords, "graphite") is None          # 'ios'? no
    assert keyword_intent("music praise nothing wrong") == "other"


def test_keyword_intent_returns_valid_labels():
    for text in [
        "i cant log in, forgot my password",
        "i was charged twice, refund please",
        "music keeps skipping and crashing",
        "downloads keep disappearing and it says offline",
        "my album was removed from the store",
    ]:
        label = keyword_intent(text)
        assert label in LABELS, (text, label)


def test_keyword_labels_specificity():
    cases = {
        "my account was hacked and my password changed": "account_login",
        "i want a refund for this month": "billing_subscription",
        "songs keep pausing on their own": "playback",
        "there is no drag handle to rearrange my playlist tracks": "playlist",
    }
    for text, expected in cases.items():
        got = keyword_intent(text)
        assert got == expected, f"{text!r}: got {got}, expected {expected}"


def test_trivial_baseline():
    b = TrivialBaseline()
    b.fit(["playback", "playback", "other"])
    assert b.predict(["anything", "else"]) == ["playback", "playback"]


def test_feature_matrix_shape():
    from scipy import sparse
    from src.intent.features import make_word_char_tfidf

    feats = make_word_char_tfidf()
    X = feats.fit_transform(["i cant log in", "songs keep pausing on my galaxy s8"])
    assert sparse.issparse(X) and X.shape[0] == 2 and X.shape[1] > 0


def test_normalize_label():
    assert normalize_label("Billing Subscription") == "billing_subscription"
    assert normalize_label("billing-subscription") == "billing_subscription"
    assert normalize_label("NOT_A_LABEL") == "unknown"


if __name__ == "__main__":
    fn = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fn:
        f()
        print(f"PASS {f.__name__}")
    print(f"\n{len(fn)} tests passed.")