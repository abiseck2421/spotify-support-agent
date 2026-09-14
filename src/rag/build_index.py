"""Build FAISS retrieval indices at several corpus sizes.

Pool = RAG-grounding pairs (substantive + contact_us replies) from
data/processed/pairs.csv, sorted by reply length DESCENDING (detail-first).

Eval holdout = the subset of RAG-grounding pairs that ALSO carry an intent
label (data/intent/train_labels.csv + data/golden/golden.csv). Those queries
are EXCLUDED from every corpus size, so retrieval can never simply
"memorize" the query tweet back to itself (no leak).

Embeddings: sentence-transformers "all-MiniLM-L6-v2", L2-normalized, indexed
with FAISS IndexFlatIP (exact = cosine similarity on unit vectors).

Outputs (all under data/retrieval/, gitignored):
  eval_queries.csv , eval_query_embs.npy   the held-out queries + embeddings
  pool_meta.json   , pool_embs.npy           full sorted pool + embeddings
  {N}/index.faiss  , {N}/slice.json          index + metadata per corpus size

Usage:
  python src/rag/build_index.py                # default sizes 1k/2k/3k/5k/all
  python src/rag/build_index.py --sizes 1000,3000
"""

from __future__ import annotations

import argparse
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

PAIRS = BASE_DIR / "data" / "processed" / "pairs.csv"
TRAIN_LABELS = BASE_DIR / "data" / "intent" / "train_labels.csv"
GOLDEN = BASE_DIR / "data" / "golden" / "golden.csv"
OUT = BASE_DIR / "data" / "retrieval"
EMBED_CACHE = BASE_DIR / "data" / "embeddings"

EMBED_MODEL = "all-MiniLM-L6-v2"
EVAL_MIN = 15  # keep queries whose cleaned text is at least this long


def load_eval_labels() -> pd.DataFrame:
    """Union of tweet-level intent labels from train_labels + golden."""
    parts = []
    tr = pd.read_csv(TRAIN_LABELS)
    parts.append(tr[["tweet_id", "intent"]].rename(columns={"tweet_id": "customer_tweet_id"}))
    g = pd.read_csv(GOLDEN)[["tweet_id", "gold_intent"]]
    parts.append(g.rename(columns={"tweet_id": "customer_tweet_id", "gold_intent": "intent"}))
    lab = pd.concat(parts, ignore_index=True).dropna(subset=["intent"])
    lab["intent"] = lab["intent"].astype(str).str.strip().str.lower()
    lab = lab[lab["intent"].astype(bool)]
    lab = lab.drop_duplicates("customer_tweet_id").astype({"customer_tweet_id": int})
    return lab


def build(args) -> None:
    pairs = pd.read_csv(PAIRS)
    pairs = pairs[pairs["is_rag_grounding"] == True].copy()  # noqa: E712
    pairs = pairs[pairs["customer_text_clean"].astype(str).str.len() >= EVAL_MIN]

    labels = load_eval_labels()
    eval_set = pairs.merge(labels, on="customer_tweet_id", how="inner")
    pool = pairs[~pairs["customer_tweet_id"].isin(eval_set["customer_tweet_id"])].copy()

    pool = pool.assign(reply_len=pool["reply_text_clean"].astype(str).str.len())
    pool = pool.sort_values("reply_len", ascending=False).reset_index(drop=True)
    print(f"Eval holdout: {len(eval_set):,} queries (excluded from all corpuses)")
    print(f"Pool:         {len(pool):,} RAG-grounding pairs (detail-first sorted)\n")

    OUT.mkdir(parents=True, exist_ok=True)
    eval_out = eval_set[
        ["customer_tweet_id", "customer_text_clean", "reply_text_clean", "intent"]
    ].rename(
        columns={
            "customer_tweet_id": "tweet_id",
            "customer_text_clean": "text",
            "reply_text_clean": "reply",
            "intent": "label",
        }
    )
    eval_out.to_csv(OUT / "eval_queries.csv", index=False)
    print(f"Wrote eval queries -> {OUT / 'eval_queries.csv'}")

    pool_meta = [
        {
            "tweet_id": int(r["customer_tweet_id"]),
            "text": str(r["customer_text_clean"]),
            "reply": str(r["reply_text_clean"]),
            "reply_len": int(r["reply_len"]),
        }
        for _, r in pool.iterrows()
    ]
    (OUT / "pool_meta.json").write_text(json.dumps(pool_meta), encoding="utf-8")

    print(f"Embedding with {EMBED_MODEL} ...")
    st = SentenceTransformer(EMBED_MODEL, cache_folder=str(EMBED_CACHE))
    pool_embs = st.encode(
        pool["customer_text_clean"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    eval_embs = st.encode(
        eval_out["text"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(OUT / "pool_embs.npy", pool_embs)
    np.save(OUT / "eval_query_embs.npy", eval_embs)
    print(f"Saved pool embs ({pool_embs.shape}) + eval embs ({eval_embs.shape})\n")

    sizes = [int(s) for s in args.sizes.strip().split(",")]
    for n in sizes:
        n = min(n, len(pool_embs))
        idx = faiss.IndexFlatIP(pool_embs.shape[1])
        idx.add(pool_embs[:n])
        d = OUT / str(n)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(d / "index.faiss"))
        (d / "slice.json").write_text(json.dumps(pool_meta[:n]), encoding="utf-8")
        print(f"corpus {n:>5,}: index -> {d / 'index.faiss'}")

    print("\nBuild complete.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        default="1000,2000,3000,5000,9000",
        help="comma-separated corpus sizes (capped at pool size)",
    )
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()