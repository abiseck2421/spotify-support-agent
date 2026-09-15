"""Rule-based escalation policy for the support agent (auto vs escalate).

NOT ML — only ~11 golden escalate=yes examples exist; a learned model would
overfit. Instead we combine three signal families:

  1. explicit security / billing-error keywords  (STRONGEST)
  2. retrieval evidence  (top-5 escalation-index hits; contact_us/dm_push ≥ 2)
  3. intent label  (never escalates alone)

Priority order (Decision 12):
  1. strong security keyword    → yes / account_security
  2. billing-error keyword      → yes / billing_error
  3. benign self-service query  → no / auto_resolvable (payment/plan/price/password
                                    login-recovery — gates the retrieval path only)
  4. retrieval ≥ 2 AND intent in {billing_subscription, account_login} → yes
  5. otherwise                  → no / auto_resolvable

"can't sign in" / "logged out" alone is treated as BENIGN self-service login
recovery (the golden set marks signed-out+can't-sign-in as auto); it still
escalates whenever paired with a strong security keyword (e.g. "account
locked"). Escalation MUST be checked BEFORE reply generation — an escalate=yes
case goes to a human, never gets an auto-reply.

CLI:  python src/agent/escalation.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import src.intent.train as _train  # noqa: F401,E402 — registers WordCharTfidf for pickle
from src.rag.retrieve import Retriever  # noqa: E402

MODEL_FILE = BASE_DIR / "results" / "models" / "intent_tfidf.pkl"
GOLDEN_FILE = BASE_DIR / "data" / "golden" / "golden.csv"
OUT_DIR = BASE_DIR / "results" / "agent"
OUT_CSV = OUT_DIR / "escalation_golden.csv"

EVIDENCE_K = 5
EVIDENCE_THRESHOLD = 2
CANDIDATE_INTENTS = ("billing_subscription", "account_login")

# ─── Keyword patterns ──────────────────────────────────────────────────────────

# Strong security: escalate alone
STRONG_SECURITY = {
    "hacked":             re.compile(r"\bhack(?:ed|ing)?\b", re.I),
    "unauthorized":       re.compile(r"\bunauthori[sz]ed\b", re.I),
    "account_locked":     re.compile(r"\baccount\s+(?:is\s+)?lock(?:ed|out)\b", re.I),
    "locked_out":         re.compile(r"\block(?:ed)?\s+out\b", re.I),
    "stolen":             re.compile(r"\bstolen\b", re.I),
    "compromised":        re.compile(r"\bcompromised\b", re.I),
    "account_takenover":  re.compile(
        r"\b(?:someone|somebody|they)\b.{0,40}\b(?:on|into)\s+my\s+account\b", re.I
    ),
}

# Weak security: never escalates alone; only surfaced for debugging
WEAK_SECURITY = {
    "password_reset":  re.compile(
        r"\b(?:reset(?:ting)?\s+(?:my\s+)?password|forgot(?:ten)?\s+(?:my\s+)?password|"
        r"password\s+reset|can'?t\s+reset\s+(?:my\s+)?password)\b", re.I
    ),
}

# Billing errors: escalate alone
BILLING_ERROR = {
    "double_charged":     re.compile(r"\bdouble[\s-]*charg(?:e|ed|ing)\b", re.I),
    "charged_twice":      re.compile(r"\bcharg(?:e|ed)\s+twice\b", re.I),
    "charged_again":      re.compile(r"\bcharg(?:e|ed)\s+again\b", re.I),
    "wrong_charge":       re.compile(r"\b(?:wrong|incorrect)\s+charg(?:e|ed|ing)\b", re.I),
    "unauthorized_charge": re.compile(r"\bunauthori[sz]ed\s+charg(?:e|ed|ing)\b", re.I),
    "overcharged":        re.compile(r"\bovercharg(?:e|ed|ing)\b", re.I),
    "refund":             re.compile(r"\brefund(?:s|ed|ing)?\b", re.I),
    "unexpected_charge":  re.compile(r"\bunexpected\s+charg(?:e|ed|ing)\b", re.I),
    "still_charged":      re.compile(r"\b(?:still|keeps?|keep\s+on)\s+(?:being\s+)?charg(?:e|ed|ing)\b", re.I),
    "cancel_still_charged": re.compile(r"\bcancell?ed?\b.{0,40}\bcharg(?:e|ed|ing)\b", re.I | re.S),
    "charged_then_cancel":  re.compile(r"\bcharg(?:e|ed)\b.{0,40}\bcancell?ed?\b", re.I | re.S),
}

# Benign self-service INQUIRIES. Decision 12: billing/plan queries must NOT
# escalate even when retrieval evidence is loud. These gate ONLY the
# retrieval-evidence path — strong security / billing-error keywords (checked
# first) always outrank them.
BENIGN_INQUIRY = {
    "change_payment":     re.compile(r"\b(?:change|update|switch|add|remove|edit|set\s+up)\b.{0,35}\bpayment\b", re.I),
    "plan_request":       re.compile(r"\b(?:would\s+like\s+to|want\s+to|how\s+do\s+i|can\s+i|need\s+to|i\s+want)\b.{0,40}\b(?:plan|membership|subscription)\b", re.I),
    "price_query":        re.compile(r"\bhow\s+much\b", re.I),
    "password_selfhelp":  re.compile(r"\b(?:reset|forgot(?:ten)?)\b.{0,30}\bpassword\b", re.I),
    "logged_out":         re.compile(r"\b(?:logged|signed|log|sign)\s+out\b", re.I),
    "cant_sign_in":       re.compile(r"\bcan'?t\s+(?:log|sign)\s*(?:in|on)\b", re.I),
    "cant_select":        re.compile(r"\bcan'?t\s+select\b", re.I),
}


def _match(text: str, pattern_dict: dict[str, re.Pattern]) -> list[str]:
    """Return list of matched pattern names."""
    return [k for k, p in pattern_dict.items() if p.search(text)]


def decide(
    intent: str,
    strong: list[str],
    weak: list[str],
    billing: list[str],
    benign: list[str],
    evidence_count: int,
) -> tuple[str, str]:
    """Priority-ordered decision: (escalate, reason).

    1. strong security keyword    → yes / account_security
    2. billing-error keyword      → yes / billing_error
    3. benign self-service query  → no  (overrides weak + retrieval evidence)
    4. retrieval≥2 AND intent candidate → yes
    5. otherwise                  → no
    """
    if strong:
        return ("yes", "account_security")
    if billing:
        return ("yes", "billing_error")
    if benign:
        return ("no", "auto_resolvable")
    if evidence_count >= EVIDENCE_THRESHOLD and intent in CANDIDATE_INTENTS:
        reason = "account_security" if intent == "account_login" else "billing_error"
        return ("yes", reason)
    return ("no", "auto_resolvable")


class EscalationPolicy:
    """Load intent classifier + escalation retriever; score one tweet."""

    def __init__(self) -> None:
        import src.intent.features as _feat  # noqa: F401 — registers for pickle

        with open(MODEL_FILE, "rb") as f:
            self._intent_model = pickle.load(f)
        self._retriever = Retriever(name="escalation")

    def evaluate(self, text: str) -> dict:
        """Classify intent, detect keywords, check retrieval evidence, decide."""
        intent = str(self._intent_model.predict([text])[0])
        lower = (text or "").lower()

        strong  = _match(lower, STRONG_SECURITY)
        weak    = _match(lower, WEAK_SECURITY)
        billing = _match(lower, BILLING_ERROR)
        benign  = _match(lower, BENIGN_INQUIRY)

        hits = self._retriever.retrieve(text, k=EVIDENCE_K)
        escalation_hits = [
            h for h in hits
            if h.get("reply_type") in ("contact_us", "dm_push")
        ]
        evidence_count = len(escalation_hits)

        escalate, reason = decide(intent, strong, weak, billing, benign, evidence_count)

        signals: list[str] = [intent] + strong + billing + weak + benign
        if evidence_count >= EVIDENCE_THRESHOLD:
            signals.append(f"retrieval_{evidence_count}_{EVIDENCE_K}")

        return {
            "escalate": escalate,
            "reason": reason,
            "signals": signals,
            "intent": intent,
            "evidence_count": evidence_count,
        }


# ─── Golden evaluation ─────────────────────────────────────────────────────────

def _y(v) -> bool:
    return str(v).strip().lower() == "yes"


def report(out: pd.DataFrame) -> None:
    y_true = out["gold_escalate"].apply(_y)
    y_pred = out["escalate"].apply(_y)

    tp = int(( y_true &  y_pred).sum())
    fp = int((~y_true &  y_pred).sum())
    fn = int(( y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    acc  = (tp + tn) / len(out)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\n{'='*60}")
    print(f"Escalation vs gold_escalate  (n={len(out)})")
    print(f"{'='*60}")
    print(f"  accuracy   = {acc:.3f}  ({tp+tn}/{len(out)})")
    print(f"  precision  = {prec:.3f}  (tp={tp}, fp={fp})")
    print(f"  recall     = {rec:.3f}  (tp={tp}, fn={fn})")
    print(f"  F1 (yes)   = {f1:.3f}")
    print(f"  confusion  = TP={tp} FP={fp} | FN={fn} TN={tn}")

    def show(label: str, mask: pd.Series, n: int = 5) -> None:
        sub = out[mask]
        if len(sub) == 0:
            return
        print(f"\n  {label}:")
        for _, r in sub.head(n).iterrows():
            text = str(r["text"])[:60]
            sigs = str(r["signals"])[:70]
            print(f"    gold={str(r['gold_escalate']):<3} pred={str(r['escalate']):<3} "
                  f"reason={str(r['reason']):<18} {text!r}")
            print(f"      signals={sigs}")

    show("TRUE POSITIVES", y_true & y_pred)
    show("FALSE POSITIVES (over-escalate)", ~y_true & y_pred)
    show("FALSE NEGATIVES (missed escalation!)", y_true & ~y_pred)
    show("TRUE NEGATIVES (sample)", ~y_true & ~y_pred, n=4)


def main() -> None:
    golden = pd.read_csv(GOLDEN_FILE)
    golden = golden.dropna(subset=["text", "gold_escalate"])
    print(f"Golden rows: {len(golden)}")

    policy = EscalationPolicy()
    rows = []
    for _, r in golden.iterrows():
        res = policy.evaluate(str(r["text"]))
        rows.append({
            "tweet_id":        r["tweet_id"],
            "text":            r["text"],
            "gold_intent":     r.get("gold_intent", ""),
            "pred_intent":     res["intent"],
            "gold_escalate":   r["gold_escalate"],
            "escalate":        res["escalate"],
            "reason":          res["reason"],
            "signals":         json.dumps(res["signals"]),
            "evidence_count":  res["evidence_count"],
        })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")

    report(out)


if __name__ == "__main__":
    main()
