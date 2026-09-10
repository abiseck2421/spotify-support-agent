import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW = BASE_DIR / "data" / "raw" / "twcs.csv"
OUT = BASE_DIR / "data" / "processed" / "spotify.csv"

print("Loading full dataset (this takes ~1-2 min)...")
df = pd.read_csv(RAW)

print(f"Total tweets in dataset: {len(df):,}")

# Keep tweets that are part of SpotifyCares conversations.
# SpotifyCares is the account that wrote brand replies (inbound=False).
brand_replies = df[df["author_id"] == "SpotifyCares"]
reply_ids = set(brand_replies["tweet_id"])

# A customer tweet that mentions @SpotifyCares OR is part of a Spotify thread.
customer_tweets = df[
    (df["inbound"] == True)
    & (df["text"].str.contains("@SpotifyCares", na=False))
]

# Collect all tweet ids that belong to Spotify conversations.
# We rely on in_response_to_tweet_id (single parent id) to follow threads.
conversation_ids = set(customer_tweets["tweet_id"]) | reply_ids

# Expand to include the parent tweets of any conversation tweet (completes threads)
parent_ids = set(df[df["tweet_id"].isin(conversation_ids)]["in_response_to_tweet_id"].dropna().astype(int))
conversation_ids |= parent_ids

# Also include tweets inside those threads
thread_tweets = df[df["in_response_to_tweet_id"].isin(conversation_ids)]

# Combine: all Spotify brand replies + all customer tweets mentioning Spotify + thread tweets
spotify = pd.concat([brand_replies, customer_tweets, thread_tweets]).drop_duplicates(
    subset="tweet_id"
)

spotify = spotify.sort_values("created_at")

print(f"Spotify tweets extracted: {len(spotify):,}")
print(f"Brand replies: {(spotify['author_id']=='SpotifyCares').sum():,}")
print(f"Customer tweets: {(spotify['inbound']==True).sum():,}")

spotify.to_csv(OUT, index=False)
print(f"Saved to {OUT}")