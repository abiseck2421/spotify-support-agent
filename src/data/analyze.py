"""Step 1 (read-only): measure how much useful Spotify support data we have.

Reads data/processed/spotify.csv and prints a summary used to
decide the RAG corpus size. Never modifies any data file.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA = BASE_DIR / "data" / "processed" / "spotify.csv"

import sys  # noqa: E402

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.reply_types import RAG_GROUNDING_TYPES, categorize  # noqa: E402

df = pd.read_csv(DATA)
customer = df[df["inbound"] == True].copy()
brand = df[df["inbound"] == False].copy()

SEP = "=" * 72
print(SEP)
print("1. BASIC COUNTS")
print(SEP)
print(f"Total Spotify tweets:        {len(df):,}")
print(f"  Customer tweets (inbound): {len(customer):,}")
print(f"  Brand replies:             {len(brand):,}")
got_reply = customer["response_tweet_id"].notna()
print(f"Customer tweets w/ downstream reply: {got_reply.sum():,} "
      f"({got_reply.mean() * 100:.1f}%)")

print(SEP)
print("2. DIRECT CUSTOMER-REPLY PAIRS")
print(SEP)
brand["parent_id"] = pd.to_numeric(brand["in_response_to_tweet_id"], errors="coerce")
cust_ids = set(customer["tweet_id"].astype(int))
direct = brand[brand["parent_id"].isin(cust_ids)].copy()
print(f"Brand replies that directly answer a customer tweet: {len(direct):,}")
print(f"Unique customer tweets covered:                      {direct['parent_id'].nunique():,}")


direct["reply_type"] = direct["text"].fillna("").map(categorize)
direct["reply_len"] = direct["text"].str.len()
cust_len_map = customer.set_index("tweet_id")["text"].str.len().to_dict()
direct["cust_len"] = direct["parent_id"].map(cust_len_map).fillna(0).astype(int)
print("\nReply type distribution over direct pairs:")
tbl = direct["reply_type"].value_counts()
for k, v in tbl.items():
    print(f"  {k:<28} {v:>6,}  ({v / len(direct) * 100:.1f}%)")

usable = direct[direct["reply_type"].isin(RAG_GROUNDING_TYPES)]
print(f"\nPair candidates rich enough for RAG grounding (substantive+contact): {len(usable):,}")

print(SEP)
print("3. TEXT HEALTH: LENGTHS, DUPLICATES, NOISE")
print(SEP)
print("Customer tweet length (chars):")
print(f"  mean {direct['cust_len'].mean():.0f}  median {direct['cust_len'].median():.0f}  "
      f"min {direct['cust_len'].min()}  max {direct['cust_len'].max()}")
print("Brand reply length (chars):")
print(f"  mean {direct['reply_len'].mean():.0f}  median {direct['reply_len'].median():.0f}  "
      f"min {direct['reply_len'].min()}  max {direct['reply_len'].max()}")

uniq_cust = customer["text"].str.strip().str.lower()
dup_frac = 1 - (uniq_cust.nunique() / len(customer))
print(f"\nExact-duplicate customer texts: {uniq_cust.duplicated().sum():,} "
      f"({dup_frac * 100:.1f}% of customer tweets)")

short = customer[customer["text"].str.len() <= 25]
print(f"Customer tweets of <=25 chars: {len(short):,} "
      f"({len(short) / len(customer) * 100:.1f}%)")

nonsample = ["indonesian:", "malay:", "spanish:", "french:", "turkish:"]
lang_hint = customer["text"].str.contains("pake pula|ga mau|gak bisa|aisé|¿|ñ|pero|donc|échange|türkiye",
                                          case=False, na=False)
print(f"Non-English hints in customer tweets (rough): {lang_hint.sum():,}")

print(SEP)
print("4. INTENT/PROBLEM KEYWORD DISTRIBUTION (customer tweets)")
print(SEP)
texts = customer["text"].str.lower().fillna("")
patterns = {
    "playback (pause/skip/stop/crash)": r"(play|pause|skip|stop|stuck|freeze|crash|won.t play|keeps)",
    "account/login (hack/password/email)": r"(hack|login|password|email|account|sign in|locked)",
    "billing/subscription": r"(bill|charge|subscription|premium|plan|cancel|refund|money|pay)",
    "downloads/offline": r"(download|offline|storage)",
    "song/album unavailable": r"(unavailable|missing|gone|removed|not available|can.t find)",
    "app/UI/feature": r"(bug|glitch|update|interface|ui|design|feature|app|shuffle)",
    "playlist": r"(playlist)",
    "device/platform": r"\b(?:device|phone|computer|tv|car|speaker|web|android|ios|iphone)\b",
    "quality/sound": r"(quality|sound|audio|volume|loud|quiet)",
}
for name, pat in patterns.items():
    n = texts.str.contains(pat, regex=True).sum()
    print(f"  {name:<40} {n:>6,}  ({n / len(customer) * 100:.1f}%)")

print(SEP)
print("5. CORPUS SIZING FEEDSTOCK (for the RAG decision)")
print(SEP)
for n in (1000, 2000, 3000, 5000):
    sub = usable.nlargest(n, "reply_len")
    median_txt = sub["text"].str.len().median()
    print(f"  Top {n:>5,} pairs by reply length -> median reply len {median_txt:.0f} chars "
          f"| covers {sub['parent_id'].nunique():,} unique customers")

print("\nAnalysis complete. No files were modified.")