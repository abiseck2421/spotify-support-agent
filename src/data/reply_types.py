"""Shared reply-type classification for SpotifyCares brand replies.

Used by src/data/analyze.py and src/data/build_pairs.py so the
"what kind of reply is this?" rule lives in exactly one place.
"""

import re

SUBSTANTIVE = (
    "(?:try|check|make sure|go to|settings|clear|restart|update|install|"
    "log out|log back|steps|open your|open the)"
)
DM_PUSH = "(?:dm us|send us a dm|private message|pls dm|please dm|dm me|direct message)"
CONTACT = "(?:get in touch|reach out|contact us|talk to|let us know|customer service)"
GENERIC = "(?:thanks for the feedback|we appreciate|glad to hear|hope this helps)"


def categorize(text: str) -> str:
    """Bucket a brand reply by whether it really helps or just deflects."""
    t = (text or "").lower()
    if re.search(SUBSTANTIVE, t):
        return "substantive_troubleshooting"
    if re.search(DM_PUSH, t):
        return "dm_push"
    if re.search(CONTACT, t):
        return "contact_us"
    if re.search(GENERIC, t):
        return "generic_thanks"
    return "other"


RAG_GROUNDING_TYPES = ("substantive_troubleshooting", "contact_us")
ESCALATION_TYPES = ("dm_push", "contact_us", "generic_thanks", "other")