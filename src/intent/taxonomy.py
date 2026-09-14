"""Single source of truth for the customer-tweet intent taxonomy.

A customer tweet gets exactly ONE primary intent (single-label).
"other" is a real class: praise, off-topic, or a complaint with no
actionable target. The seed keywords below are used ONLY by the
keyword/regex baseline (baseline 2) and by the stratified sampling of
training data. They are NOT the final classifier (that is TF-IDF+LogReg
trained on Gemini-drafted labels).

Boundary scheme: a leading \\b on the group anchors each alternative at a
word START (so tokens never match inside other words - the "car" inside
"SpotifyCares" Day-1 trap). There is intentionally NO trailing group
boundary: word stems like paus/play/download/skip must keep matching
inflected forms (pausing, playing, downloaded, skipping). Tokens that must
stand alone (ui, pc, tv, pay, card, bug) carry an explicit \\b of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    label: str
    definition: str
    seed_keywords: str
    examples: tuple[str, ...] = field(default_factory=tuple)


INTENTS = (
    Intent(
        label="account_login",
        definition=(
            "Problems logging in, password/email resets, locked or hacked "
            "accounts, wrong credentials, CSRF/verification errors. Actionable "
            "target = access to the account itself."
        ),
        seed_keywords=(
            r"\b(?:log\sin|login|sign\sin|password|hack|compromis|locked|"
            r"forgot|reset\s?(?:password|pw)|email\s?(?:change|confirm)|"
            r"csrf|credential|unable\s?to\s?(?:access|get\s?into)|"
            r"can'?t\s?(?:access|get\s?into))"
        ),
        examples=(
            "i am trying to reset my password but the page shows 'the csrf token is invalid.'",
            "i can't log in and when i try to change my password it says the code timed out",
            "my account was hacked/compromised and my email was changed",
        ),
    ),
    Intent(
        label="billing_subscription",
        definition=(
            "Charges, subscription plans, cancellations, refunds, payment "
            "errors, trial/free-then-charged issues. Actionable target = money."
        ),
        seed_keywords=(
            r"\b(?:bill|charg|refund|subscription|plan\b|premium|cancel|"
            r"trial|payment|pay\b|card\b|recurring|money|purchase|99\s?cents|"
            r"upgrade|downgrade|invoice)"
        ),
        examples=(
            "my card is being charged for premium even though i cancelled a day before",
            "i bought the $0.99 for 3 months but my account still says free",
            "i was charged twice for the same subscription",
        ),
    ),
    Intent(
        label="playback",
        definition=(
            "Music/video won't start, keeps skipping, pauses by itself, crashes "
            "while playing, freezes, buffers, black screen in the player. "
            "Actionable target = the act of playing."
        ),
        seed_keywords=(
            r"\b(?:won'?t\s?play|not\s?(?:play|working)|keeps?\s?(?:skip|paus|stop)|"
            r"skip(?:ping)?\s?(?:tracks|songs)?|paus(?:e|es|ing)|"
            r"freeze|frozen|crash(?:es|ing)?|black\s?screen|buffer(?:ing)?|"
            r"stuck|load(?:s|ing)?\s?forever|start(?:s)?\s?then\s?stops)"
        ),
        examples=(
            "your app keeps crashing in the middle of a song on my galaxy s8",
            "spotify is stuck loading and nothing will play",
            "songs pause by themselves every few seconds",
        ),
    ),
    Intent(
        label="downloads_offline",
        definition=(
            "Downloading songs for offline listening, offline mode not working, "
            "storage/memory limits for downloads. Actionable target = offline "
            "listening."
        ),
        seed_keywords=(
            r"\b(?:download(?:s|ed|ing)?|offline|storage|save\s?for\s?offline|"
            r"not\s?available\s?offline|can'?t\s?(?:download|listen\s?offline))"
        ),
        examples=(
            "it keeps switching to offline mode even though i have data",
            "my downloads disappeared after the update",
            "why can't i download playlists anymore",
        ),
    ),
    Intent(
        label="song_unavailable",
        definition=(
            "Specific songs/albums/artists missing, removed, greyed out, "
            "region-locked, or 'not available'. Actionable target = content."
        ),
        seed_keywords=(
            r"\b(?:unavailable|missing|gone|removed|not\s?(?:available|there)|"
            r"can'?t\s?(?:find|find\s?anymore)|disappeared|greyed|grey(?:ed)?\s?out|"
            r"empty\s?(?:playlist|album)?|doesn'?t\s?(?:exist|show))"
        ),
        examples=(
            "the album is missing from my library",
            "why was this song removed from spotify?",
            "i can't find the artist anywhere anymore",
        ),
    ),
    Intent(
        label="app_ui_feature",
        definition=(
            "Interface/design complaints, features not working the way users "
            "expect, feature requests, update changed the layout, dark mode, "
            "shuffle behavior, sorting. Actionable target = the app itself, "
            "not a specific song or device."
        ),
        seed_keywords=(
            r"\b(?:interface|ui\b|design|layout|dark\s?mode|update|new\s?update|"
            r"feature|shuffle|sorting|rollback|old\s?(?:interface|ui|version)|"
            r"app\s?(?:broke|crash|glitch)|glitch|bug\b)"
        ),
        examples=(
            "why can't i turn off the new interface and use the old one",
            "the update removed dark mode",
            "shuffle keeps picking the same few songs",
        ),
    ),
    Intent(
        label="playlist",
        definition=(
            "Creating/editing/sharing playlists, collaborative playlists, track "
            "ordering inside a playlist, playlist disappearing."
        ),
        seed_keywords=(
            r"\b(?:playlist|collab|together|edit(?:ing)?\s?(?:playlist|order)|"
            r"reorder|rearrange|move\s?tracks?)"
        ),
        examples=(
            "i want to rearrange tracks in my playlist but there is no drag handle",
            "my collaborative playlist disappeared",
            "songs keep getting removed from my playlist",
        ),
    ),
    Intent(
        label="device_platform",
        definition=(
            "Behavior specific to a device/platform (Android, iOS, web, desktop, "
            "TV, Playstation, car, speaker) or cross-device sync/connectivity "
            "that is not playback or downloads."
        ),
        seed_keywords=(
            r"\b(?:device|phone|android|ios|iphone|ipad|mac\b|windows|pc\b|linux|"
            r"desktop|web\b|browser|tv\b|ps4|playstation|xbox|car\b|speaker|"
            r"echo\b|alexa|sonos|chromecast|sync|connected\s?device)"
        ),
        examples=(
            "it works on my iphone but not on the web browser",
            "spotify on my tv keeps disconnecting from the app on my phone",
            "the desktop app can't see the devices to cast to",
        ),
    ),
    Intent(
        label="quality_sound",
        definition=(
            "Audio quality, volume weirdness, equalizer, loudness/quietness "
            "between songs, hiss/sound defects."
        ),
        seed_keywords=(
            r"\b(?:quality|bitrate|equaliz|volume|loud|quiet|hiss|static|"
            r"audio\s?(?:quality|is)|sound\s?(?:quality|is)|muffled|"
            r"distortion|too\s?(?:loud|quiet))"
        ),
        examples=(
            "why is the volume so much quieter on podcasts than music",
            "the audio quality is terrible on this new update",
            "the equalizer setting keeps resetting",
        ),
    ),
    Intent(
        label="other",
        definition=(
            "Everything else: praise/thanks, off-topic, memes, non-English, "
            "diffuse complaints with no actionable target, or not enough "
            "context to pick a primary problem."
        ),
        seed_keywords=(
            r"\b(?:love\s?(?:spotify|the\s?app|this)|thank|awesome|amazing|"
            r"great\b|compliment|best\s?app|worst\s?app)"
        ),
        examples=(
            "thank you for fixing the issue",
            "spotify is amazing",
            "why is spotify so bad now (no specific problem)",
        ),
    ),
)

LABEL_TO_INTENT = {i.label: i for i in INTENTS}
LABELS = tuple(i.label for i in INTENTS)
PRIORITY = tuple(i.label for i in INTENTS)

# The fallback label a Gemini-drafted label maps to when it is not in the
# taxonomy (e.g. Gemini invents a label).
UNKNOWN_LABEL = "unknown"


def normalize_label(label: str) -> str:
    """Map any detected/LLM label into the taxonomy, else UNKNOWN_LABEL."""
    if not label:
        return UNKNOWN_LABEL
    l = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    if l in LABEL_TO_INTENT:
        return l
    return UNKNOWN_LABEL