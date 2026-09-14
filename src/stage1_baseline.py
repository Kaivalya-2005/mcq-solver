"""
Stage 1 Baseline: Smart MCQ Solver
------------------------------------
Approach: TF-IDF cosine similarity between the question prompt and each
answer option. Rank the 5 options by similarity, output the top 3.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.data_utils import load_train, load_test, make_submission, OPTION_COLS
from src.map3 import map_at_3


def rank_options_tfidf(df, vectorizer):
    """Rank options by TF-IDF cosine similarity to prompt."""
    prompt_vecs = vectorizer.transform(df["prompt"].tolist())

    rankings = []
    for i in range(len(df)):
        option_texts = [str(df.iloc[i][c]) for c in OPTION_COLS]
        option_vecs = vectorizer.transform(option_texts)
        sims = cosine_similarity(prompt_vecs[i], option_vecs)[0]
        order = np.argsort(-sims)  # descending similarity
        ranked_labels = [OPTION_COLS[j] for j in order]
        rankings.append(ranked_labels)
    return rankings


def build_vectorizer(df):
    """Fit TF-IDF on ALL text (prompts + all options)."""
    corpus = df["prompt"].tolist()
    for c in OPTION_COLS:
        corpus += df[c].astype(str).tolist()
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )
    vec.fit(corpus)
    return vec


def run_baseline():
    """Main baseline pipeline."""
    train = load_train()
    test = load_test()

    # Validate on held-out split
    tr, val = train_test_split(train, test_size=0.2, random_state=42)

    vectorizer = build_vectorizer(tr)
    val_rankings = rank_options_tfidf(val, vectorizer)
    val_top3 = [r[:3] for r in val_rankings]

    score = map_at_3(val["answer"].tolist(), val_top3)
    print(f"Baseline TF-IDF similarity — Validation MAP@3: {score:.4f}")

    random_map3 = (1 / 5) * (1 + 1 / 2 + 1 / 3)
    print(f"Random baseline MAP@3 (for reference): {random_map3:.4f}")

    # Refit on full train, predict on test
    full_vectorizer = build_vectorizer(train)
    test_rankings = rank_options_tfidf(test, full_vectorizer)
    test_top3 = [r[:3] for r in test_rankings]

    sub = make_submission(test, test_top3, "./outputs/submission_baseline_tfidf.csv")
    print("\nSaved submission to ./outputs/submission_baseline_tfidf.csv")
    print(sub.head())


if __name__ == "__main__":
    run_baseline()