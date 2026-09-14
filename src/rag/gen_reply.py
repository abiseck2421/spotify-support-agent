"""Grounded reply generation (RAG evidence + intent) vs a no-RAG control.

Pipeline for one customer tweet:
  1. intent = intent-classifier prediction (the agent's own estimate, not the label)
  2. evidence = top-k retrieved historical issues+replies (full corpus)
  3. Gemini writes a reply USING the evidence (grounded) vs WITHOUT it (no-RAG)

Outputs a side-by-side CSV for a quick human/LLM sanity read:
  results/rag/generated_replies.csv   (gitignored)

Usage:
  python src/rag/gen_reply.py --n 12 --seed 9
  python src/rag/gen_reply.py --n 12 --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm import generate  # noqa: E402
from src.rag.retrieve import DEFAULT_SIZE, Retriever  # noqa: E402

OUT = BASE_DIR / "data" / "retrieval"
RESULTS = BASE_DIR / "results" / "rag"
MODEL_FILE = BASE_DIR / "results" / "models" / "intent_tfidf.pkl"
DEFAULT_GEN_MODEL = "gemini-flash-lite-latest"

GROUNDED_PROMPT = (
    "You are the SpotifyCares support agent replying on Twitter. A customer "
    "wrote the message below ({intent} problem area). You have HISTORICAL "
    "EXAMPLES of similar issues being resolved - use them to give an "
    "accurate, specific reply, restating the concrete fix the examples "
    "support, in your own words, 1-3 sentences.\n\n"
    "HISTORICAL ISSUE 1:\n{ev1}\n\nHISTORICAL ISSUE 2:\n{ev2}\n\n"
    "HISTORICAL ISSUE 3:\n{ev3}\n\n"
    "CUSTOMER: {query}\n\nREPLY:"
)

NO_RAG_PROMPT = (
    "You are the SpotifyCares support agent replying on Twitter. Write ONE "
    "helpful, specific reply (1-3 sentences) that resolves this customer's "
    "problem. Do not just thank them or push them to DM us.\n\n"
    "CUSTOMER: {query}\n\nREPLY:"
)


def load_pipeline():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def format_evidence(hit: dict) -> str:
    return f"Issue: {hit['text']}\nReply that fixed it: {hit['reply']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="queries to generate for")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--k", type=int, default=3, help="evidence count per query")
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    eval_df = pd.read_csv(OUT / "eval_queries.csv")
    rng = random.Random(args.seed)
    sample = eval_df.sample(min(args.n, len(eval_df)), random_state=args.seed)

    retriever = Retriever(DEFAULT_SIZE)
    pipe = load_pipeline()

    rows = []
    for _, row in sample.iterrows():
        query = str(row["text"])
        intent = str(pipe.predict([query])[0])
        evs = retriever.retrieve(query, k=args.k)
        ev_block = "\n\n".join(f"{i+1}.\n{format_evidence(e)}"
                               for i, e in enumerate(evs))
        grounded = generate(
            GROUNDED_PROMPT.format(
                intent=intent, query=query,
                ev1=format_evidence(evs[0]) if len(evs) > 0 else "(none)",
                ev2=format_evidence(evs[1]) if len(evs) > 1 else "(none)",
                ev3=format_evidence(evs[2]) if len(evs) > 2 else "(none)",
            ),
            model=args.model, max_tokens=512,
        ).strip()
        no_rag = generate(
            NO_RAG_PROMPT.format(query=query), model=args.model, max_tokens=512
        ).strip()
        rows.append({
            "tweet_id": int(row["tweet_id"]),
            "query": query,
            "known_label": row["label"],
            "intent_pred": intent,
            "evidence": json.dumps(
                [{"text": e["text"], "reply": e["reply"], "score": e["score"]} for e in evs]
            ),
            "grounded_reply": grounded,
            "no_rag_reply": no_rag,
            "model": args.model,
        })
        print(f"  [{len(rows)}/{len(sample)}] t{row['tweet_id']} "
              f"intent={intent} -> {grounded[:70]!r}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "generated_replies.csv", index=False)
    print(f"\nSaved {len(df)} side-by-side replies -> "
          f"{RESULTS / 'generated_replies.csv'}")


if __name__ == "__main__":
    main()