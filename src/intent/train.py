"""Train the intent classifier (OUR model, not a baseline).

Model: TF-IDF word n-grams -> LogisticRegression, straight scikit-learn, no
tuning games. Trained on Gemini-drafted labels (data/intent/train_labels.csv)
with a stratified 5-fold CV hold-out. Saves:
  - fitted model:  results/models/intent_tfidf.pkl
  - hold-out CV:   results/intents/train_holdout.json
Never touches the golden set (test-only).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.intent.features import WordCharTfidf, make_word_char_tfidf  # noqa: E402

LABELS_FILE = BASE_DIR / "data" / "intent" / "train_labels.csv"
MODEL_OUT = BASE_DIR / "results" / "models" / "intent_tfidf.pkl"
METRICS_OUT = BASE_DIR / "results" / "intents" / "train_holdout.json"

SEED = 3


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_FILE)
    df = df[df["intent"].notna() & df["text"].notna()]
    df = df[df["intent"].astype(str).str.strip().astype(bool)]
    return df


def make_pipeline() -> Pipeline:
    """Word-TFIDF + character-TFIDF features -> balanced LogisticRegression.

    Char n-grams matter on short, typo-heavy customer tweets; balanced
    weights stop the rare classes (quality_sound, playlist) from being
    crushed by the frequent ones.
    """
    return Pipeline(
        [
            ("feat", make_word_char_tfidf()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]
    )


def main() -> None:
    df = load_labels()
    print(f"Training on {len(df):,} labels. Classes: {df['intent'].nunique()}")
    print(df["intent"].value_counts().to_string())

    X = df["text"].astype(str).tolist()
    y = df["intent"].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_accs = []
    for tr_idx, va_idx in skf.split(X, y):
        pipe = make_pipeline()
        pipe.fit([X[i] for i in tr_idx], [y[i] for i in tr_idx])
        preds = pipe.predict([X[i] for i in va_idx])
        fold_accs.append(accuracy_score([y[i] for i in va_idx], preds))

    # Final model on ALL labels (downstream Day 5 replay uses everything).
    pipe = make_pipeline()
    pipe.fit(X, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipe, f)

    report = classification_report(
        y, pipe.predict(X), output_dict=True, zero_division=0
    )
    metrics = {
        "crossval_mean_acc": round(sum(fold_accs) / len(fold_accs), 4),
        "crossval_acc_per_fold": fold_accs,
        "train_set_accuracy": round(accuracy_score(y, pipe.predict(X)), 4),
        "classes": {k: v for k, v in report.items() if isinstance(v, dict)},
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nCross-val mean acc (5-fold): {metrics['crossval_mean_acc']:.3f}")
    print(f"Saved model  -> {MODEL_OUT}")
    print(f"Saved CV     -> {METRICS_OUT}")


if __name__ == "__main__":
    main()