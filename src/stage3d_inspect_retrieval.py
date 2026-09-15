"""
Stage 3d: Evaluate retrieval quality BEFORE building on top of it.

We don't have human-labeled "relevant passage" annotations, so true
Recall@k/MRR isn't directly computable. Two things instead:

  1. Print ~20 (prompt, correct answer, retrieved_context) triples for
     manual eyeballing -- this is what the plan calls "manually inspect
     ~50 questions."
  2. A heuristic proxy: does the retrieved context share more vocabulary
     with the CORRECT option than with the average incorrect option?
     If retrieval is doing its job, yes. This isn't a substitute for
     manual review, just a fast automatic sanity check across all rows.

Usage:
    uv run python -m src.stage3d_inspect_retrieval
"""
import os
import re
import pandas as pd

CORPUS_DIR = "./data_enriched"
OPTION_COLS = ["A", "B", "C", "D", "E"]
N_MANUAL_SAMPLES = 20


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "that", "this", "these", "those", "it", "its", "as", "by", "from",
    "which", "what", "who", "when", "where", "how", "than", "then",
    "also", "such", "can", "may", "not", "no", "if", "into", "over",
    "between", "their", "his", "her", "they", "he", "she", "we", "you",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "one", "two", "each", "other", "some", "all", "more",
    "most", "so", "there", "any", "used", "use", "based",
}


def tokenize(text):
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return set(t for t in tokens if t not in STOPWORDS and len(t) > 2)


def overlap_score(context_tokens, option_text):
    opt_tokens = tokenize(option_text)
    if not opt_tokens:
        return 0.0
    return len(context_tokens & opt_tokens) / len(opt_tokens)


def main():
    train = pd.read_csv(f"{CORPUS_DIR}/train_with_context.csv")

    # ---- 1. Manual inspection printout ----
    print("=" * 80)
    print(f"MANUAL INSPECTION SAMPLE ({N_MANUAL_SAMPLES} rows)")
    print("=" * 80)
    sample = train.sample(n=min(N_MANUAL_SAMPLES, len(train)), random_state=1)
    for _, row in sample.iterrows():
        print(f"\nQ: {row['prompt']}")
        print(f"Correct answer ({row['answer']}): {row[row['answer']]}")
        print(f"Sources: {row.get('retrieved_sources', 'n/a')}")
        ctx = str(row["retrieved_context"])
        print(f"Context (first 300 chars): {ctx[:300]}...")

    # ---- 2. Automatic proxy metric across the whole set ----
    print("\n" + "=" * 80)
    print("AUTOMATIC PROXY METRIC (not a substitute for manual review)")
    print("=" * 80)
    correct_overlaps, incorrect_overlaps = [], []
    for _, row in train.iterrows():
        ctx_tokens = tokenize(row["retrieved_context"])
        if not ctx_tokens:
            continue
        correct_overlaps.append(overlap_score(ctx_tokens, row[row["answer"]]))
        for c in OPTION_COLS:
            if c != row["answer"]:
                incorrect_overlaps.append(overlap_score(ctx_tokens, row[c]))

    print(f"Mean vocab overlap: context vs CORRECT option   = {sum(correct_overlaps)/len(correct_overlaps):.4f}")
    print(f"Mean vocab overlap: context vs INCORRECT options = {sum(incorrect_overlaps)/len(incorrect_overlaps):.4f}")
    print("\nIf the first number is meaningfully higher than the second, retrieval")
    print("is (weakly) favoring the correct answer -- good sign to proceed to Stage 4.")
    print("If they're close/equal, the corpus or query cleaning needs work before")
    print("building the LLM reasoning stage on top of it.")


if __name__ == "__main__":
    main()