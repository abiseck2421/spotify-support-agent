"""Compare the 3 intent classifiers honestly.

Trivial (majority), keyword/regex baseline 2, and our TF-IDF+LogReg model.
Two eval surfaces:
  1. held-out slice of the Gemini train labels (stratified 80/20)
  2. the human-confirmed golden set (test-only) — the number the report quotes
Saves JSON to results/intents/eval_*.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.intent.baselines import TrivialBaseline, keyword_intent
from src.intent.taxonomy import LABELS
import src.intent.train as _train  # noqa: F401 - registers WordCharTfidf for pickle

TRAIN_FILE = BASE_DIR / "data" / "intent" / "train_labels.csv"
GOLDEN_FILE = BASE_DIR / "data" / "golden" / "golden.csv"
MODEL_FILE = BASE_DIR / "results" / "models" / "intent_tfidf.pkl"
OUT_DIR = BASE_DIR / "results" / "intents"

SEED = 3


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_FILE)
    df = df[df["intent"].notna() & df["text"].notna()]
    return df


def load_golden() -> pd.DataFrame:
    if not GOLDEN_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(GOLDEN_FILE)
    df = df[df["gold_intent"].notna() & df["text"].notna()]
    return df


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "per_class": classification_report(y_true, y_pred, output_dict=True,
                                           zero_division=0, labels=LABELS),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": list(LABELS),
    }


def main() -> None:
    train = load_train()
    golden = load_golden()
    print(f"Train labels: {len(train):,} | Golden (gold_intent): {len(golden):,}")

    tr, ho = train_test_split(
        train, test_size=0.2, stratify=train["intent"], random_state=SEED
    )

    # 1. trivial
    trivial = TrivialBaseline()
    trivial.fit(tr["intent"].tolist())

    # 2. keyword
    # 3. our model
    #    - held-out uses a model fit on the 80% train slice only (honest CV-ish)
    #    - golden (truly unseen) uses the saved full-data model
    heldout_model = make_pipeline_from_train(tr)
    golden_model = None
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            golden_model = pickle.load(f)
        print("Loaded saved full-data model for the golden evaluation.")

    sets = {"heldout": ho}
    if len(golden):
        sets["golden"] = golden

    results = {}
    for sname, sdf in sets.items():
        X = sdf["text"].astype(str).tolist()
        y = list(sdf["intent"] if sname == "heldout" else sdf["gold_intent"])
        model = heldout_model if sname == "heldout" else golden_model
        row = {
            "trivial": evaluate(y, trivial.predict(X)),
            "keyword": evaluate(y, [keyword_intent(t) for t in X]),
            "our_tfidf_logreg": evaluate(y, model.predict(X)),
        }
        results[sname] = row
        print(f"\n=== {sname} (n={len(y)}) ===")
        for name, met in row.items():
            print(f"{name:<18} acc={met['accuracy']:.3f}  "
                  f"macro-F1={met['per_class']['macro avg']['f1-score']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sname, row in results.items():
        with open(OUT_DIR / f"eval_{sname}.json", "w") as f:
            json.dump(row, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'eval_*.json'}")


def make_pipeline_from_train(tr: pd.DataFrame):
    from src.intent.train import make_pipeline

    pipe = make_pipeline()
    pipe.fit(tr["text"].astype(str).tolist(), tr["intent"].tolist())
    return pipe


if __name__ == "__main__":
    main()