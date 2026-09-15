"""End-to-end evaluation harness over the golden set (Day 5/6, Part B).

Runs the FULL agent stack per golden row and scores each stage:

  1. intent      <- EscalationPolicy (src/agent/escalation.py — classifies too)
  2. escalation  <- same call; HARD routing decision
  3. RAG evidence<- Retriever(DEFAULT_SIZE) top-k (src/rag/retrieve.py)
  4. reply       <- Gemini via GROUNDED_PROMPT (src/rag/gen_reply.py)
  5. control     <- Gemini via NO_RAG_PROMPT (no evidence)
  6. LLM judge   <- Gemini flash-lite scores grounding/helpfulness/tone 1-5
                    for BOTH replies against the reference reply

Routing rule (Decision 12 + Part A): escalate=yes means the generated reply is
NOT an auto-resolution — it must go to a human. Those rows are recorded and
analyzed separately (secondary), never counted in the headline reply stats.
escalate=no rows are the production auto-answers and drive the headline metrics.

API-failure tolerance: every Gemini call is wrapped; a failed row is recorded
with the error and processing continues. per_row.csv doubles as a checkpoint —
re-running skips tweet_ids already evaluated (resume-safe).

Outputs (gitignored via results/):
  results/evaluation/per_row.csv
  results/evaluation/harness_report.json

Usage:
  python src/evaluation/harness.py                # all 59 rows
  python src/evaluation/harness.py --limit 5      # smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.escalation import EscalationPolicy  # noqa: E402
from src.intent.taxonomy import LABELS  # noqa: E402
from src.llm import generate, jsonify  # noqa: E402
from src.rag.gen_reply import (  # noqa: E402
    GROUNDED_PROMPT,
    NO_RAG_PROMPT,
    format_evidence,
)
from src.rag.retrieve import DEFAULT_SIZE, Retriever  # noqa: E402

GOLDEN_FILE = BASE_DIR / "data" / "golden" / "golden.csv"
OUT_DIR = BASE_DIR / "results" / "evaluation"
ROW_CSV = OUT_DIR / "per_row.csv"
REPORT_JSON = OUT_DIR / "harness_report.json"

DEFAULT_GEN_MODEL = "gemini-flash-lite-latest"
AXES = ("grounding", "helpfulness", "tone")

JUDGE_PROMPT = (
    "You are an expert judge of customer-support AI replies. A customer wrote "
    "a problem tweet. You are given the REFERENCE reply (the fix the real "
    "SpotifyCares team gave) and TWO ASSISTANT replies for the same tweet. "
    "Score each assistant reply on three axes, 1 = poor, 5 = excellent:\n\n"
    "- grounding: sticks to the verified fix in the reference instead of "
    "inventing steps (making up fixes => low). If the reference has no concrete "
    "fix (e.g. it just asks the customer to DM us), grounding means the "
    "assistant admits it cannot self-serve rather than fabricating a fix.\n"
    "- helpfulness: how likely the reply moves the customer toward resolution.\n"
    "- tone: warm, professional, human.\n\n"
    "CUSTOMER: {query}\n\n"
    "REFERENCE: {reference}\n\n"
    "ASSISTANT A (grounded): {grounded}\n\n"
    "ASSISTANT B (no RAG): {no_rag}\n\n"
    'Respond ONLY with JSON:\n{{"A":{{"grounding":1,"helpfulness":1,"tone":1}},'
    '"B":{{"grounding":1,"helpfulness":1,"tone":1}}}}'
)


# ─── helpers ───────────────────────────────────────────────────────────────────

def _safe(call, *args, **kwargs):
    """Run call; return (value, None) or (None, error-string)."""
    try:
        return call(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001 - one row failing must not kill the run
        return None, f"{type(e).__name__}: {e}"


def _grounded_prompt(query: str, intent: str, evs: list[dict]) -> str:
    return GROUNDED_PROMPT.format(
        intent=intent,
        query=query,
        ev1=format_evidence(evs[0]) if len(evs) > 0 else "(none)",
        ev2=format_evidence(evs[1]) if len(evs) > 1 else "(none)",
        ev3=format_evidence(evs[2]) if len(evs) > 2 else "(none)",
    )


def _clamp_score(v) -> int | None:
    """Coerce a judge value (int/float/str like '4' or '4.0') into 1..5."""
    if isinstance(v, bool):
        return None
    try:
        return max(1, min(5, int(round(float(str(v).strip())))))
    except (TypeError, ValueError):
        return None


def _parse_judge(raw: str) -> dict | None:
    try:
        data = json.loads(jsonify(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, dict[str, int | None]] = {}
    for key, alias in (("A", "grounded"), ("B", "no_rag")):
        block = data.get(key)
        if not isinstance(block, dict):
            block = data.get(alias, {})
        if not isinstance(block, dict):
            out[key] = {a: None for a in AXES}
            continue
        out[key] = {a: _clamp_score(block.get(a)) for a in AXES}
    return out


def judge_replies(query: str, reference: str, grounded: str, no_rag: str, model: str) -> str:
    prompt = JUDGE_PROMPT.format(
        query=query[:500],
        reference=(reference or "(no reference reply)")[:500],
        grounded=grounded[:500],
        no_rag=no_rag[:500],
    )
    return generate(prompt, model=model, max_tokens=256)


# ─── per-row pipeline ──────────────────────────────────────────────────────────

def process_row(row, policy: EscalationPolicy, retriever: Retriever,
                gen_model: str, k: int) -> dict:
    query = str(row["text"])
    esc = policy.evaluate(query)  # intent + escalation in one call

    evs = retriever.retrieve(query, k=k)
    ev_scores = [e["score"] for e in evs]

    grounded, ge = _safe(generate, _grounded_prompt(query, esc["intent"], evs),
                         model=gen_model, max_tokens=512)
    no_rag, ne = _safe(generate, NO_RAG_PROMPT.format(query=query),
                       model=gen_model, max_tokens=512)

    judge: dict | None = None
    if grounded and no_rag:
        raw, je = _safe(judge_replies, query, str(row["reference_reply_clean"]),
                        grounded, no_rag, gen_model)
        judge = _parse_judge(raw) if (raw and not je) else None
        if judge is None:
            je = je or "judge output unparseable"
    else:
        je = "skipped (missing generated reply)"

    out = {
        "tweet_id": int(row["tweet_id"]),
        "text": query,
        "gold_intent": str(row.get("gold_intent", "")),
        "pred_intent": esc["intent"],
        "gold_escalate": str(row.get("gold_escalate", "")),
        "escalate": esc["escalate"],
        "reason": esc["reason"],
        "signals": json.dumps(esc["signals"]),
        "evidence_count": esc["evidence_count"],
        "reference_reply": str(row.get("reference_reply_clean", "")),
        "ev_scores": json.dumps(ev_scores),
        "routed": "escalate" if esc["escalate"] == "yes" else "auto",
        "grounded_reply": grounded or "",
        "no_rag_reply": no_rag or "",
        "gen_error": json.dumps({"grounded": ge, "no_rag": ne}),
        "judge_error": je or "",
    }
    g = (judge or {}).get("A", {}) or {}
    b = (judge or {}).get("B", {}) or {}
    for axis in AXES:
        out[f"rag_{axis}"] = g.get(axis)
        out[f"no_rag_{axis}"] = b.get(axis)
    return out


# ─── aggregation ───────────────────────────────────────────────────────────────

def _num_mean_std(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 3) if len(s) else None,
        "std": round(float(s.std()), 3) if len(s) > 1 else None,
        "median": round(float(s.median()), 3) if len(s) else None,
    }


def _axis_comparison(df: pd.DataFrame, col_rag: str, col_norag: str) -> dict:
    row = df[[col_rag, col_norag]].apply(pd.to_numeric, errors="coerce").dropna()
    rag_win = int((row[col_rag] > row[col_norag]).sum())
    tie = int((row[col_rag] == row[col_norag]).sum())
    norag_win = int((row[col_rag] < row[col_norag]).sum())
    return {
        "n": int(len(row)),
        "rag_wins": rag_win,
        "tie": tie,
        "no_rag_wins": norag_win,
        "mean_delta_rag_minus_norag": round(float((row[col_rag] - row[col_norag]).mean()), 3)
        if len(row) else None,
    }


def _reply_block(rows: pd.DataFrame) -> dict:
    clean = rows[rows["judge_error"].fillna("").astype(str).str.strip().str.len() == 0]
    block: dict[str, dict] = {}
    for tag, cols in (("rag", "rag_"), ("no_rag", "no_rag_")):
        block[tag] = {a: _num_mean_std(clean[f"{cols}{a}"]) for a in AXES}
    block["comparisons"] = {
        a: _axis_comparison(clean, f"rag_{a}", f"no_rag_{a}") for a in AXES
    }
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--k", type=int, default=3, help="RAG evidence per query")
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between rows (free-tier friendly)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    golden = pd.read_csv(GOLDEN_FILE).dropna(subset=["text", "gold_escalate"])
    if args.limit > 0:
        golden = golden.head(args.limit)
    print(f"Golden rows to evaluate: {len(golden)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if ROW_CSV.exists():
        done = set(pd.read_csv(ROW_CSV)["tweet_id"].astype(int))
        print(f"Resume: {len(done)} rows already evaluated; skipping them")

    policy = EscalationPolicy()
    retriever = Retriever(DEFAULT_SIZE)

    pending = golden[~golden["tweet_id"].astype(int).isin(done)]
    print(f"Pending: {len(pending)} rows\n")

    new_rows: list[dict] = []
    for i, (_, row) in enumerate(pending.iterrows(), 1):
        rec = process_row(row, policy, retriever, args.model, args.k)
        new_rows.append(rec)
        print(f"  [{i}/{len(pending)}] t{rec['tweet_id']} "
              f"intent={rec['pred_intent']} escalate={rec['escalate']} "
              f"route={rec['routed']} | {str(row['text'])[:55]!r}")
        if rec["judge_error"]:
            print(f"      judge problem: {rec['judge_error']}")

        if new_rows and (i % 5 == 0 or i == len(pending)):
            all_rows = pd.concat(
                [pd.read_csv(ROW_CSV), pd.DataFrame(new_rows)], ignore_index=True
            ) if ROW_CSV.exists() else pd.DataFrame(new_rows)
            all_rows = all_rows.drop_duplicates("tweet_id", keep="last")
            all_rows.to_csv(ROW_CSV, index=False)
            print(f"      checkpoint -> per_row.csv ({len(all_rows)} rows)")

        if args.sleep:
            time.sleep(args.sleep)

    rows = pd.read_csv(ROW_CSV) if ROW_CSV.exists() else pd.DataFrame()
    rows = rows[rows["tweet_id"].astype(int).isin(golden["tweet_id"].astype(int))]
    if len(rows) == 0:
        print("No rows to report.")
        return

    # 1. intent
    y_i_true = rows["gold_intent"].astype(str).str.strip()
    y_i_pred = rows["pred_intent"].astype(str).str.strip()
    report_i = classification_report(
        y_i_true, y_i_pred, output_dict=True, zero_division=0, labels=LABELS
    )
    intent_metrics = {
        "n": int(len(rows)),
        "accuracy": round(accuracy_score(y_i_true, y_i_pred), 4),
        "macro_f1": round(report_i["macro avg"]["f1-score"], 4),
        "per_class_f1": {
            lbl: round(report_i[lbl]["f1-score"], 4) for lbl in LABELS
        },
    }

    # 2. escalation
    y_e_true = rows["gold_escalate"].astype(str).str.strip().str.lower() == "yes"
    y_e_pred = rows["escalate"].astype(str).str.strip().str.lower() == "yes"
    tp = int((y_e_true & y_e_pred).sum())
    fp = int((~y_e_true & y_e_pred).sum())
    fn = int((y_e_true & ~y_e_pred).sum())
    tn = int((~y_e_true & ~y_e_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    escalation_metrics = {
        "n": int(len(rows)),
        "accuracy": round((tp + tn) / len(rows), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    # 3. reply quality — auto rows drive the headline; escalated rows are secondary
    terms = {
        "all": rows,
        "auto": rows[rows["routed"] == "auto"],
        "escalated_secondary": rows[rows["routed"] == "escalate"],
    }
    reply_metrics = {name: _reply_block(sub) for name, sub in terms.items()}

    report = {
        "model": args.model,
        "golden_n": int(len(golden)),
        "evaluated_rows": int(len(rows)),
        "intent": intent_metrics,
        "escalation": escalation_metrics,
        "reply": reply_metrics,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {ROW_CSV}")
    print(f"Saved -> {REPORT_JSON}")

    print(f"\n=== INTENT (n={intent_metrics['n']}) ===")
    print(f"  accuracy = {intent_metrics['accuracy']:.3f}  macro-F1 = {intent_metrics['macro_f1']:.3f}")
    print(f"\n=== ESCALATION (n={escalation_metrics['n']}) ===")
    print(f"  acc={escalation_metrics['accuracy']:.3f}  prec={escalation_metrics['precision']:.3f}  "
          f"rec={escalation_metrics['recall']:.3f}  F1={escalation_metrics['f1']:.3f}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    for name, blk in reply_metrics.items():
        print(f"\n=== REPLY quality — {name} (evaluated rows with judge) ===")
        for tag in ("rag", "no_rag"):
            vals = "  ".join(
                f"{a}={blk[tag][a]['mean']}" for a in AXES if blk[tag][a]["n"]
            )
            print(f"  {tag:<7} {vals}")
        for a in AXES:
            cm = blk["comparisons"][a]
            if cm["n"]:
                print(f"  {a:<11} delta(rag-no_rag)={cm['mean_delta_rag_minus_norag']:+.3f}  "
                      f"rag_wins={cm['rag_wins']} tie={cm['tie']} no_rag_wins={cm['no_rag_wins']}")


if __name__ == "__main__":
    main()