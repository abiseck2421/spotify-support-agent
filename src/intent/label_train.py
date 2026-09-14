"""Draft intent labels for a ~600-tweet training sample via Gemini.

Sampling is stratified: each non-"other" intent gets quota from the
keyword-positive slice of pairs.csv, plus a random residual to catch
"other" and everything the keywords missed. Gemini drafts one clean
intent per tweet (single-label). Output: data/intent/train_labels.csv
with a verified flag (default False). A human re-checks a slice before
this file is trusted for training.

Idempotent: per-tweet results cached in data/intent/label_cache.json so a
crash or quota stall does not re-spend API calls.
"""

from __future__ import annotations

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
CACHE_FILE = BASE_DIR / "data" / "intent" / "label_cache.json"
OUT = BASE_DIR / "data" / "intent" / "train_labels.csv"

PER_INTENT = 70
RESIDUAL = 0
TARGET = 600
SEED = 7

_TAX = "\n".join(f"- {i.label}: {i.definition}" for i in INTENTS)
_SYS = (
    "You label short customer tweets about Spotify with exactly ONE intent "
    "from this taxonomy:\n" + _TAX + "\n"
    "Rules: single-label only; if the tweet has no clear actionable problem "
    "(praise, off-topic, vague rant) choose 'other'. Reply with a JSON array "
    "like [{\"id\": <int>, \"intent\": \"<label>\", \"reason\": \"<10 words>\"}] "
    "with one object per tweet, same order."
)


def sample_rows(seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pairs = pd.read_csv(PAIRS)
    all_pairs = pairs["customer_text_clean"].dropna()

    text_to_id = {
        t: int(i) for t, i in zip(all_pairs, pairs.loc[all_pairs.index, "customer_tweet_id"])
    }

    sampled: list[str] = []
    for intent in INTENTS:
        if intent.label == "other":
            continue
        rng.seed(seed)
        pool = all_pairs[all_pairs.str.lower().str.contains(
            intent.seed_keywords, regex=True)]
        sampled += rng.sample(list(pool), min(PER_INTENT, len(pool)))

    # residual keeps it near TARGET and collects "other"/keyword misses
    everything = set(all_pairs)
    used = set(sampled)
    candidates = list(everything - used)
    rng.seed(seed + 1)
    n_res = max(TARGET - len(sampled) - RESIDUAL * 0, 0)
    sampled += rng.sample(candidates, min(n_res, len(candidates)))

    df = pd.DataFrame({"text": list(dict.fromkeys(sampled))})  # dedup, keep order
    df["tweet_id"] = df["text"].map(text_to_id)
    return df


def cache() -> dict:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}  # json round-trip stringifies keys
    return {}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="override the Gemini model (e.g. gemini-flash-lite-latest)")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    sample = sample_rows(SEED)
    print(f"Sample size: {len(sample):,} tweets (stratified by keyword family)")

    store = cache()
    todo = [(i, txt) for i, (_, row) in enumerate(sample.iterrows()) if (txt := str(row["text"]))]

    # fill cache with per-tweet labels in small batches
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        owed = [(idx, txt) for idx, txt in chunk if idx not in store]
        if not owed:
            continue
        prompt = _SYS + "\n\nTweets:\n" + "\n".join(f"{idx}: {txt}" for idx, txt in owed)
        print(f"Batch {i // args.batch + 1}: {len(owed)} new tweets -> Gemini ...")
        try:
            raw = batch_generate([prompt], max_tokens=args.max_tokens,
                                 retries=3, model=args.model)[0]
            for idx, (intent, reason) in parse_labels(raw, owed):
                store[idx] = {"intent": intent, "reason": reason}
        except NoApiKeyError:
            raise
        except Exception as e:  # noqa: BLE001
            if is_quota_error(e):
                print(f"  QUOTA ({e}); cache saved, resume later with: "
                      f"python src/intent/label_train.py --model {args.model}")
                break
            print(f"  batch failed ({e}); falling back to per-tweet calls")
            for idx, txt in owed:
                if idx in store:
                    continue
                try:
                    raw = batch_generate(
                        [_SYS + "\n\nTweets:\n" + f"{idx}: {txt}"],
                        max_tokens=args.max_tokens, retries=3, model=args.model,
                    )[0]
                except NoApiKeyError:
                    raise
                except Exception as e2:  # noqa: BLE001
                    if is_quota_error(e2):
                        print(f"  QUOTA hit in per-tweet fallback; resuming later.")
                        break
                    continue
                parsed = parse_labels(raw, [(idx, txt)])
                if parsed:
                    store[idx] = {"intent": parsed[0][1][0], "reason": parsed[0][1][1]}
        CACHE_FILE.write_text(json.dumps(store, indent=1), encoding="utf-8")

    CACHE_FILE.write_text(json.dumps(store, indent=1), encoding="utf-8")

    id_by_idx = {i: tid for i, tid in enumerate(sample["tweet_id"])}
    rows = [
        {"tweet_id": id_by_idx.get(idx, idx), "text": txt,
         "intent": store.get(idx, {}).get("intent", ""),
         "reason": store.get(idx, {}).get("reason", ""), "verified": False}
        for idx, txt in todo if idx in store
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df):,} labeled rows -> {OUT}")
    print(df["intent"].value_counts().to_string())


def is_quota_error(e: Exception) -> bool:
    s = str(e).upper()
    return "RESOURCE_EXHAUSTED" in s or "429" in s or "QUOTA" in s


def parse_labels(raw: str, owed: list) -> list:
    """Parse Gemini's JSON array; each element -> (intent, reason)."""
    import json as _json

    cleaned = jsonify(raw)
    # find first '[' ... last ']' — robust to preamble
    try:
        data = _json.loads(cleaned)
    except _json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        data = _json.loads(cleaned[start : end + 1])
    out = []
    for i, (idx, _txt) in enumerate(owed):
        if i < len(data) and isinstance(data[i], dict):
            intent = str(data[i].get("intent", "")).strip().lower()
            reason = str(data[i].get("reason", "")).strip()
        else:
            intent, reason = "", ""
        out.append((idx, (intent, reason)))
    return out


if __name__ == "__main__":
    main()