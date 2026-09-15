"""Build a FAISS retrieval index for ESCALATION evidence.

Pool = escalation-type pairs (dm_push + contact_us + generic_thanks + other)
from data/processed/pairs.csv — i.e. replies where the real SpotifyCares team
DEFLECTED to a human/DM rather than auto-troubleshooting. The escalation policy
uses these as evidence that a similar query needs human hands.

Stores reply_type in the metadata slice so the policy can count how many of its
top-k hits are contact_us / dm_push (the only types that count as escalation).

Same leakage protection as the RAG index: golden + train-labeled tweets are
EXCLUDED, so retrieval can never "memorize" the query back to itself.

Outputs (under data/retrieval/escalation/, gitignored):
  index.faiss   FAISS IndexFlatIP (exact cosine on L2-normalized embeddings)
  slice.json    metadata: {tweet_id, text, reply, reply_len, reply_type}

Usage:
  python src/agent/build_escalation_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.reply_types import ESCALATION_TYPES  # noqa: E402
from src.rag.build_index import (  # noqa: E402
    EMBED_CACHE,
    EMBED_MODEL,
    OUT,
    load_eval_labels,
)

PAIRS = BASE_DIR / "data" / "processed" / "pairs.csv"
ESCDIR = OUT / "escalation"
EVAL_MIN = 15


def build() -> None:
    pairs = pd.read_csv(PAIRS)
    pairs = pairs[pairs["reply_type"].isin(ESCALATION_TYPES)].copy()
    pairs = pairs.dropna(subset=["customer_text_clean", "reply_text_clean"])
    pairs = pairs[pairs["customer_text_clean"].astype(str).str.len() >= EVAL_MIN]
    pairs = pairs[pairs["reply_text_clean"].astype(str).str.len() >= EVAL_MIN]
    print(f"Escalation-type pairs (len>={EVAL_MIN}): {len(pairs):,}")
    print(pairs["reply_type"].value_counts().to_string())

    labels = load_eval_labels()
    pool = pairs[~pairs["customer_tweet_id"].isin(labels["customer_tweet_id"])].copy()
    print(f"Eval holdout: {len(pairs) - len(pool):,} tweets excluded (no leak)")

    pool = pool.assign(reply_len=pool["reply_text_clean"].astype(str).str.len())
    pool = pool.sort_values("reply_len", ascending=False).reset_index(drop=True)

    meta = [
        {
            "tweet_id": int(r["customer_tweet_id"]),
            "text": str(r["customer_text_clean"]),
            "reply": str(r["reply_text_clean"]),
            "reply_len": int(r["reply_len"]),
            "reply_type": str(r["reply_type"]),
        }
        for _, r in pool.iterrows()
    ]

    ESCDIR.mkdir(parents=True, exist_ok=True)
    (ESCDIR / "slice.json").write_text(json.dumps(meta), encoding="utf-8")

    print(f"\nEmbedding {len(meta):,} texts with {EMBED_MODEL} ...")
    st = SentenceTransformer(EMBED_MODEL, cache_folder=str(EMBED_CACHE))
    embs = st.encode(
        pool["customer_text_clean"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, str(ESCDIR / "index.faiss"))
    print(f"index  -> {ESCDIR / 'index.faiss'} ({embs.shape[0]:,} vectors)")
    print(f"meta   -> {ESCDIR / 'slice.json'} ({len(meta):,} rows, reply_type stored)")
    print("Build complete.")


def main() -> None:
    build()


if __name__ == "__main__":
    main()