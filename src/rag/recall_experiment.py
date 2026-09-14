"""Retrieval-recall vs corpus size (evidence for the RAG cutoff).

Metric A (cheap proxy) — intent-agreement recall. For each of the 282
held-out eval queries (excluded from every corpus; known intent label), run
FAISS top-k retrieval over a corpus, predict the intent of each retrieved
example with the intent classifier, and count a hit when a retrieved example's
predicted intent equals the query's label.

  recall@k    = % of queries where >=1 of the top-k retrieved share intent
  precision@5 = mean fraction of the top-5 sharing intent

This is labelled a PROXY on purpose: it is bounded by classifier agreement,
so Metric B sanity-checks it with direct relevance judgments.

Metric B (LLM spot-check) — relevance@1 of the top-1 retrieved historical
reply, judged by Gemini (flash-lite) on a stratified sample of queries at the
sizes {1000, 3000, FULL}. A reply is RELEVANT only when it addresses the
same root cause (not merely the same topic).

Outputs (gitignored):
  results/rag/recall.json               Metric A table + B summary
  results/rag/relevance_spotcheck.csv   per-judgement rows for B

Usage:
  python src/rag/recall_experiment.py                 # A (no Gemini)
  python src/rag/recall_experiment.py --judge 30 --judge-sizes 1000,3000,FULL
  python src/rag/recall_experiment.py --judge 30 --judge-sizes FULL
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm import batch_generate, jsonify  # noqa: E402
from src.intent.taxonomy import normalize_label  # noqa: E402

OUT = BASE_DIR / "data" / "retrieval"
RESULTS = BASE_DIR / "results" / "rag"
MODEL_FILE = BASE_DIR / "results" / "models" / "intent_tfidf.pkl"
JUDGE_MODEL = "gemini-flash-lite-latest"

_TOP_K = (1, 3, 5)


def load_pipeline():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def table_A_sizes() -> list[str]:
    return sorted(
        (p.name for p in OUT.iterdir() if (p / "index.faiss").exists()),
        key=int,
    )


def metric_A() -> list[dict]:
    eval_df = pd.read_csv(OUT / "eval_queries.csv")
    eval_embs = np.load(OUT / "eval_query_embs.npy").astype(np.float32)
    pipe = load_pipeline()

    # classify every distinct retrieved text once across all sizes
    text_examples: dict[str, dict] = {}
    for size in table_A_sizes():
        meta = json.loads((OUT / size / "slice.json").read_text(encoding="utf-8"))
        for m in meta:
            text_examples.setdefault(m["text"], m)
    texts = list(text_examples)
    preds = pipe.predict(texts)
    pred_by_text = dict(zip(texts, preds))

    qt = eval_df["text"].astype(str).tolist()
    qlab = eval_df["label"].astype(str).str.strip().str.lower().tolist()
    full = table_A_sizes()[-1]

    rows = []
    for size in table_A_sizes():
        meta = json.loads((OUT / size / "slice.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(OUT / size / "index.faiss"))
        D, I = index.search(eval_embs, k=_TOP_K[-1])
        hits = {k: 0 for k in _TOP_K}
        prec_sum = 0
        for qi in range(len(qt)):
            toks = []
            for pos in I[qi]:
                if pos < 0 or pos >= len(meta):
                    continue
                toks.append(pred_by_text.get(meta[pos]["text"], ""))
            for k in _TOP_K:
                if any(t == qlab[qi] for t in toks[:k]):
                    hits[k] += 1
            prec_sum += (sum(1 for t in toks if t == qlab[qi]) / len(toks)) if toks else 0
        n = len(qt)
        rows.append(
            {
                "size": int(size),
                "n_queries": n,
                "recall@1": round(hits[1] / n, 4),
                "recall@3": round(hits[3] / n, 4),
                "recall@5": round(hits[5] / n, 4),
                "precision@5": round(prec_sum / n, 4),
            }
        )
    print("\nMetric A — intent-agreement recall (proxy)\n")
    print(f"{'size':>6} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'precision@5':>11}")
    for r in rows:
        print(f"{r['size']:>6,} {r['recall@1']:>9.3f} {r['recall@3']:>9.3f} "
              f"{r['recall@5']:>9.3f} {r['precision@5']:>11.3f}")
    print(f"\n(full corpus = {int(full):,} items; queries excluded from all corpuses)")
    return rows


_JUDGE_PROMPT = (
    "You are evaluating a support-knowledge RETRIEVAL system. For each case, "
    "decide whether the RETRIEVED REPLY is RELEVANT to the CUSTOMER PROBLEM: "
    "it addresses the SAME ROOT CAUSE (so it would plausibly help resolve the "
    "issue), not just the same general topic. Answer strictly with a JSON "
    "array of objects in the SAME order as the cases:\n"
    '[{"id": <int>, "relevant": "yes|no", "reason": "<12 words>"}]'
)


def judge_cases(cases: list[tuple[int, str, str]]) -> list[dict]:
    out: dict[int, dict] = {}
    for start in range(0, len(cases), 10):
        chunk = cases[start : start + 10]
        body = "\n\n".join(
            f"case {cid}:\ncustomer problem: {q}\nretrieved reply: {r}"
            for cid, q, r in chunk
        )
        for _ in range(3):
            try:
                raw = batch_generate([_JUDGE_PROMPT + "\n\n" + body], retries=3,
                                     model=JUDGE_MODEL)[0]
                data = json.loads(jsonify(raw))
                for item in data if isinstance(data, list) else []:
                    if isinstance(item, dict) and item.get("id") is not None:
                        out[int(item["id"])] = item
                break
            except Exception as e:  # noqa: BLE001
                print(f"  judge chunk retry ({e})")
    return out


def metric_B(n: int, sizes: list[str]) -> list[dict]:
    eval_df = pd.read_csv(OUT / "eval_queries.csv")
    eval_embs = np.load(OUT / "eval_query_embs.npy").astype(np.float32)

    strat = eval_df["label"].astype(str)
    sample = eval_df.groupby(strat, group_keys=False).apply(
        lambda g: g.sample(min(len(g), 3), random_state=9)
    )
    if len(sample) < n:
        need = n - len(sample)
        rest = eval_df.drop(sample.index).sample(need, random_state=9)
        sample = pd.concat([sample, rest])
    sample = sample.head(n).reset_index(drop=True)
    sidx = eval_df.index.isin(sample.index)

    qv = (eval_df["text"].astype(str).tolist(),)
    full = table_A_sizes()[-1]
    resolved = [s if s not in ("FULL", "full") else full for s in sizes]
    size_res_names = dict(zip(resolved, sizes))

    rows = []
    for size in resolved:
        meta = json.loads((OUT / size / "slice.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(OUT / size / "index.faiss"))
        D, I = index.search(eval_embs[sidx], k=1)
        cases = []
        for qi in range(len(sample)):
            pos = int(I[qi][0])
            cid = int(sample.iloc[qi]["tweet_id"])
            cases.append((cid, qv[0][qi], meta[pos]["reply"]))
        judged = judge_cases(cases)
        yes = sum(
            1 for cid, *_ in cases
            if str(judged.get(cid, {}).get("relevant", "")).lower().startswith("y")
        )
        rows.append(
            {
                "size_label": size_res_names[size],
                "size": int(size),
                "n_judged": len(cases),
                "relevance@1": round(yes / len(cases), 3),
            }
        )
        debug_df = pd.DataFrame(
            [
                {
                    "tweet_id": cid,
                    "query": q,
                    "reply": r,
                    "relevant": judged.get(cid, {}).get("relevant", ""),
                    "reason": judged.get(cid, {}).get("reason", ""),
                }
                for cid, q, r in cases
            ]
        )
        debug_df.to_csv(RESULTS / f"relevance_spotcheck_{size}.csv", index=False)
        print(f"\nMetric B — relevance@1 judged by {JUDGE_MODEL} (n={len(cases)}):")
        for r in rows:
            print(f"  size {r['size_label']:>6}  relevance@1 = {r['relevance@1']:.3f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=int, default=0,
                    help="run Metric B on this many sampled queries (0 = skip)")
    ap.add_argument("--judge-sizes", default="1000,3000,FULL",
                    help="comma-separated sizes for the spot-check (FULL = largest)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    table_a = metric_A()
    summary = {"metric_A": table_a}

    if args.judge > 0:
        sizes = [s.strip() for s in args.judge_sizes.split(",") if s.strip()]
        table_b = metric_B(args.judge, sizes)
        summary["metric_B"] = table_b
        summary["judge_model"] = JUDGE_MODEL

    (RESULTS / "recall.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS / 'recall.json'}")


if __name__ == "__main__":
    main()