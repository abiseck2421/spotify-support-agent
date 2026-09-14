"""Golden-set slice builder (~40-50/day, test-only).

Draws unseen customer tweets from data/processed/pairs.csv (broad traffic,
i.e. answered messages) and asks Gemini to DRAFT an intent + escalation
hint. The human reviews/adjusts, then `--lock` promotes confirmed rows to
data/golden/golden.csv and adds them to the seen-ids tracker so each day
adds fresh examples. Golden is NEVER used for training or tuning.

Usage:
  python src/intent/build_golden.py                 # draft N=45 new examples
  python src/intent/build_golden.py --n 50          # draft 50
  python src/intent/build_golden.py --lock          # promote confirmed drafts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.intent.taxonomy import INTENTS
from src.llm import NoApiKeyError, batch_generate, jsonify

PAIRS = BASE_DIR / "data" / "processed" / "pairs.csv"
GOLDEN_DIR = BASE_DIR / "data" / "golden"
DRAFT = GOLDEN_DIR / "golden_draft.csv"
LOCKED = GOLDEN_DIR / "golden.csv"
SEEN = GOLDEN_DIR / "seen_ids.json"

_TAX = "\n".join(f"- {i.label}: {i.definition}" for i in INTENTS)
_PROMPT = (
    "You help build a test set for a Spotify support agent. For each tweet, "
    "output EXACTLY ONE primary intent from this taxonomy:\n" + _TAX + "\n\n"
    "Also decide escalation as yes|no: escalate when the reply cannot safely "
    "be automated (account security/hacked, money/billing errors, refunds). "
    "Reply with a JSON array of objects in the SAME order as the tweets:\n"
    '[{"id": <int>, "intent": "<label>", "escalate": "yes|no", "reason": "<10 words>"}]'
)


def seen_ids() -> set:
    if SEEN.exists():
        return set(json.loads(SEEN.read_text(encoding="utf-8")))
    return set()


def mk_row(tweet_id, text, reply, g):
    return {
        "tweet_id": tweet_id,
        "text": text,
        "reference_reply_clean": reply,
        "gemini_intent": str(g.get("intent", "")).strip().lower(),
        "gemini_escalate": str(g.get("escalate", "")).strip().lower(),
        "gemini_reason": str(g.get("reason", "")).strip(),
        "gold_intent": "",
        "gold_escalate": "",
        "verified": False,
    }


def draft(args) -> None:
    pairs = pd.read_csv(PAIRS)
    pairs["text"] = pairs["customer_text_clean"].astype(str)
    pairs = pairs[pairs["text"].str.len() >= 8]
    if args.seed is not None:
        rng = random.Random(args.seed)
        pool = pairs.apply(
            lambda r: (int(r["customer_tweet_id"]), r["text"], r["reply_text_clean"]),
            axis=1,
        ).tolist()
    else:
        pool = pairs.apply(
            lambda r: (int(r["customer_tweet_id"]), r["text"], r["reply_text_clean"]),
            axis=1,
        ).tolist()
        rng = random.Random()

    seen = seen_ids()
    fresh = [(tid, t, rp) for tid, t, rp in pool if tid not in seen and any(c in t for c in " ")]
    rng.shuffle(fresh)
    rows = []
    quota_hit = False
    for start in range(0, min(args.n, len(fresh)), 20):
        chunk = fresh[start : start + 20]
        prompt = _PROMPT + "\n\nTweets:\n" + "\n".join(f"{tid}: {t}" for tid, t, _ in chunk)
        try:
            raw = batch_generate([prompt], retries=3, model=args.model)[0]
            parsed = parse_golden(raw)
        except NoApiKeyError:
            raise
        except Exception as e:  # noqa: BLE001
            if is_quota_error(e):
                print(f"  QUOTA ({e}); draft partial and resume later with "
                      f"--model {args.model}")
                quota_hit = True
                break
            print(f"  batch failed ({e}); skipping chunk")
            parsed = {}
        for tid, t, rp in chunk:
            rows.append(mk_row(tid, t, rp, parsed.get(tid, {})))

    df = pd.DataFrame(rows)
    if DRAFT.exists():
        existing = pd.read_csv(DRAFT)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("tweet_id")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DRAFT, index=False)
    print(f"Drafted {len(df):,} example(s) -> {DRAFT}")
    print("Review columns: gold_intent, gold_escalate, set verified=True to promote.")


def parse_golden(raw: str) -> dict:
    import json as _json

    cleaned = jsonify(raw)
    try:
        data = _json.loads(cleaned)
    except _json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        data = _json.loads(cleaned[start : end + 1])
    out = {}
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("id") is not None:
            out[int(item["id"])] = item
    return out


def lock() -> None:
    if not DRAFT.exists():
        print("No draft file; nothing to lock.")
        return
    df = pd.read_csv(DRAFT)
    confirmed = df[df["verified"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if confirmed.empty:
        print("No confirmed (verified=True) rows in draft yet.")
        return

    # test-only guard: refuse to promote anything whose text is in the train labels
    train_file = BASE_DIR / "data" / "intent" / "train_labels.csv"
    if train_file.exists():
        tr_texts = set(
            pd.read_csv(train_file)["text"].astype(str).str.strip().str.lower()
        )
        confirmed = confirmed[
            ~confirmed["text"].astype(str).str.strip().str.lower().isin(tr_texts)
        ]
        if confirmed.empty:
            print("All confirmed rows collide with the training set; nothing locked.")
            return
        print(f"After test-only overlap guard: {len(confirmed):,} lockable rows.")

    confirmed = confirmed.copy()
    confirmed["verified"] = True
    cols = [
        "tweet_id", "text", "reference_reply_clean",
        "gemini_reason", "gold_intent", "gold_escalate",
    ]
    final = confirmed[cols]
    final.insert(0, "gemini_intent", confirmed["gemini_intent"])
    if LOCKED.exists():
        old = pd.read_csv(LOCKED)
        final = pd.concat([old, final], ignore_index=True).drop_duplicates("tweet_id")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(LOCKED, index=False)

    seen = seen_ids() | set(confirmed["tweet_id"].astype(int))
    SEEN.write_text(json.dumps(sorted(seen)), encoding="utf-8")

    # drop promoted rows from the draft
    remaining = df[df["tweet_id"].astype(int).isin(seen) == False]  # noqa: E712
    remaining.to_csv(DRAFT, index=False)
    print(f"Locked {len(confirmed):,} -> {LOCKED} (total {len(final):,}). "
          f"Draft now has {len(remaining):,} pending.")


def is_quota_error(e: Exception) -> bool:
    s = str(e).upper()
    return "RESOURCE_EXHAUSTED" in s or "429" in s or "QUOTA" in s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--model", default=None, help="Gemini model override")
    args = ap.parse_args()
    if args.lock:
        lock()
    else:
        draft(args)


if __name__ == "__main__":
    main()