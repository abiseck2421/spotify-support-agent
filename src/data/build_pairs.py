"""Step 2: build clean customer→reply pairs from data/processed/spotify.csv.

Reads spotify.csv (never twcs.csv directly) and writes data/processed/pairs.csv.
Each row = one customer tweet paired with the SpotifyCares reply that answered it.
Rows are tagged with reply_type (substantive / dm_push / contact_us / generic / other)
and is_rag_grounding (True for substantive+contact — useful for RAG grounding).
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.reply_types import RAG_GROUNDING_TYPES, categorize

SPOTIFY = BASE_DIR / "data" / "processed" / "spotify.csv"
OUT = BASE_DIR / "data" / "processed" / "pairs.csv"


def clean(text: str) -> str:
    """Strip noise from a tweet, leaving only the human message."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                             # @mentions
    text = text.replace("\u200b", "")                             # zero-width chars
    text = re.sub(r"\s+", " ", text).strip()                     # whitespace
    return text


print(f"Reading {SPOTIFY} ...")
df = pd.read_csv(SPOTIFY)

customer = df[df["inbound"] == True].copy()
brand    = df[df["inbound"] == False].copy()

# Build direct-pair index: brand reply whose in_response_to_tweet_id == a customer tweet
brand["parent_id"] = pd.to_numeric(brand["in_response_to_tweet_id"], errors="coerce")
cust_ids = set(customer["tweet_id"].astype(int))
direct   = brand[brand["parent_id"].isin(cust_ids)].copy()

print(f"Direct brand replies to a customer tweet: {len(direct):,}")

# Quick lookup: customer side
cust_lookup = customer.set_index("tweet_id").to_dict("index")

rows = []
for _, reply in direct.iterrows():
    cust_id = int(reply["parent_id"])
    cust_row = cust_lookup[cust_id]

    cust_orig  = str(cust_row["text"])
    reply_orig = str(reply["text"])

    rows.append({
        # --- customer ---
        "customer_tweet_id": cust_id,
        "customer_created_at": cust_row["created_at"],
        "customer_text_orig": cust_orig,
        "customer_text_clean": clean(cust_orig),
        # --- brand reply ---
        "brand_tweet_id": int(reply["tweet_id"]),
        "brand_created_at": reply["created_at"],
        "reply_text_orig": reply_orig,
        "reply_text_clean": clean(reply_orig),
        # --- tags ---
        "reply_type": categorize(reply_orig),
    })

pairs = pd.DataFrame(rows)
pairs["is_rag_grounding"] = pairs["reply_type"].isin(RAG_GROUNDING_TYPES)

# Dedup on cleaned customer text, keeping earliest pair by customer_created_at
pairs.sort_values("customer_created_at", inplace=True)
before = len(pairs)
pairs.drop_duplicates(subset="customer_text_clean", keep="first", inplace=True)
after = len(pairs)
print(f"Dedup: {before:,} → {after:,} unique customer messages")

# Summary
print("\nReply-type distribution after dedup:")
for k, v in pairs["reply_type"].value_counts().items():
    print(f"  {k:<28} {v:>6,}  ({v / len(pairs) * 100:.1f}%)")

n_ground = pairs["is_rag_grounding"].sum()
print(f"\nRAG-grounding pairs: {n_ground:,} ({n_ground / len(pairs) * 100:.1f}%)")

OUT.parent.mkdir(parents=True, exist_ok=True)
pairs.to_csv(OUT, index=False)
print(f"\nSaved {len(pairs):,} rows → {OUT}")
