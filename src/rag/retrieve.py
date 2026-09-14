"""Retrieval query interface over a built FAISS corpus.

Shared by the reply generator and the full agent. Loads one corpus size
(default FULL = the size chosen by the recall experiment) once, then answers
queries with FAISS top-k. Returns the matched historical
issue + reply + similarity, which downstream code feeds to Gemini as evidence.

CLI:
  python src/rag/retrieve.py --query "my playlist disappeared"
  python src/rag/retrieve.py --query "can't log in" --k 3 --size 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OUT = BASE_DIR / "data" / "retrieval"
EMBED_CACHE = BASE_DIR / "data" / "embeddings"
EMBED_MODEL = "all-MiniLM-L6-v2"

DEFAULT_SIZE = 8591  # FULL corpus chosen by the Day 4 recall experiment


class Retriever:
    def __init__(self, size: int = DEFAULT_SIZE) -> None:
        self.size = int(size)
        d = OUT / str(self.size)
        if not (d / "index.faiss").exists() or not (d / "slice.json").exists():
            raise FileNotFoundError(
                f"No corpus built for size {self.size}. Run build_index.py first."
            )
        self.index = faiss.read_index(str(d / "index.faiss"))
        self.meta = json.loads((d / "slice.json").read_text(encoding="utf-8"))
        self.st = SentenceTransformer(EMBED_MODEL, cache_folder=str(EMBED_CACHE))

    def retrieve(self, text: str, k: int = 5) -> list[dict]:
        """Top-k evidence: {model_index, tweet_id, text, reply, reply_len, score}."""
        emb = self.st.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        scores, idx = self.index.search(emb, k=k)
        out = []
        for pos, score in zip(idx[0].tolist(), scores[0].tolist()):
            if pos < 0 or pos >= len(self.meta):
                continue
            hit = dict(self.meta[pos])
            hit["model_index"] = int(pos)
            hit["score"] = round(float(score), 4)
            out.append(hit)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    args = ap.parse_args()

    r = Retriever(args.size)
    for ev in r.retrieve(args.query, k=args.k):
        print(f"[{ev['score']:.3f}] {ev['text'][:90]}\n      -> {ev['reply'][:140]}\n")


if __name__ == "__main__":
    main()