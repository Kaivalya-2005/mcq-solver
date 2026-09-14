"""
Stage 1 Baseline: Smart MCQ Solver
------------------------------------
Approach: TF-IDF cosine similarity between the question prompt and each
answer option. Rank the 5 options by similarity, output the top 3.

This has NO deep learning and NO retrieval yet -- it's the sanity-check
baseline every later model (DeBERTa fine-tune, RAG+LLM) must beat.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

OPTION_COLS = ["A", "B", "C", "D", "E"]


def map_at_3(true_labels, pred_top3_lists):
    """
    Mean Average Precision @ 3, as used by the competition.
    true_labels: list of single correct labels, e.g. ['B', 'A', ...]
    pred_top3_lists: list of lists of 3 predicted labels in ranked order,
                      e.g. [['B','C','A'], ...]
    """
    scores = []
    for true, preds in zip(true_labels, pred_top3_lists):
        score = 0.0
        for i, p in enumerate(preds[:3]):
            if p == true:
                score = 1.0 / (i + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def rank_options_tfidf(df, vectorizer):
    """
    For every row, vectorize the prompt and the 5 options with the SAME
    fitted vectorizer, then rank options by cosine similarity to the prompt.
    Returns a list of ranked-label-lists, e.g. [['C','A','E'], ...]
    """
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
    """Fit TF-IDF on ALL text (prompts + all options) so prompt/option
    vectors live in the same vocabulary space."""
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


if __name__ == "__main__":
    train = pd.read_csv("/mnt/user-data/uploads/train_2_.csv")
    test = pd.read_csv("/mnt/user-data/uploads/test_2_.csv")

    # ---- 1. Validate the baseline on a held-out split of TRAIN data ----
    tr, val = train_test_split(train, test_size=0.2, random_state=42)

    vectorizer = build_vectorizer(tr)  # fit only on training split
    val_rankings = rank_options_tfidf(val, vectorizer)
    val_top3 = [r[:3] for r in val_rankings]

    score = map_at_3(val["answer"].tolist(), val_top3)
    print(f"Baseline TF-IDF similarity — Validation MAP@3: {score:.4f}")

    # Reference: a random ranker gets MAP@3 ~= (1/5 + 1/5*1/2 + 1/5*1/3) = 0.3667
    random_map3 = (1 / 5) * (1 + 1 / 2 + 1 / 3)
    print(f"Random baseline MAP@3 (for reference): {random_map3:.4f}")

    # ---- 2. Refit on FULL train data, predict on real TEST set ----
    full_vectorizer = build_vectorizer(train)
    test_rankings = rank_options_tfidf(test, full_vectorizer)
    test_top3 = [r[:3] for r in test_rankings]

    submission = pd.DataFrame({
        "ID": test["id"],
        "Prediction": [" ".join(r) for r in test_top3],
    })
    submission.to_csv("/mnt/user-data/outputs/submission_baseline_tfidf.csv", index=False)
    print("\nSaved submission to submission_baseline_tfidf.csv")
    print(submission.head())
