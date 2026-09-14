"""Feature builders shared by train.py and evaluate.py.

Exists so the pickled intent pipeline references a stable module path
(src.intent.features) instead of whatever __main__ ran the training script.
"""

from __future__ import annotations

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class WordCharTfidf:
    """Sparse hstack of word and char TF-IDF matrices (sklearn-compatible)."""

    def __init__(self, word_vec: TfidfVectorizer, char_vec: TfidfVectorizer) -> None:
        self.word_vec = word_vec
        self.char_vec = char_vec

    def fit(self, X, y=None):
        self.word_vec.fit(X)
        self.char_vec.fit(X)
        return self

    def transform(self, X):
        return sparse.hstack(
            [self.word_vec.transform(X), self.char_vec.transform(X)], "csr"
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def make_word_char_tfidf() -> WordCharTfidf:
    word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=120_000)
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
        max_features=120_000,
        min_df=1,
    )
    return WordCharTfidf(word_vec, char_vec)