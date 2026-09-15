import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from src.map3 import map_at_3
from src.splits import clean_query,group_aware_split

OPTION_COLS = ["A", "B", "C", "D", "E"]
OPTION_TO_INDEX = {c: i for i, c in enumerate(OPTION_COLS)}

DATA_DIR = os.environ.get("MCQ_DATA_DIR", "./data")
CORPUS_DIR = "./data_enriched"
OUTPUT_DIR = "./outputs"

STAGE2_BASE_DIR = "./stage2_out"
LLM_CACHE_PATH = f"{CORPUS_DIR}/llm_cache.json"

DEFAULT_ALPHAS = [round(x / 10.0, 1) for x in range(11)]


class MultipleChoiceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        flattened = []
        for feature in features:
            for i in range(5):
                flattened.append({key: value[i] for key, value in feature.items()})

        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        batch_size = len(features)
        return {key: value.view(batch_size, 5, -1) for key, value in batch.items()}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def rank_to_prob_dist(ranking: List[str]) -> np.ndarray:
    cleaned = []
    for letter in ranking:
        letter = str(letter).strip().upper()
        if letter in OPTION_TO_INDEX and letter not in cleaned:
            cleaned.append(letter)
    for option in OPTION_COLS:
        if option not in cleaned:
            cleaned.append(option)
    cleaned = cleaned[:5]

    weights_by_rank = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    probs = np.zeros(5, dtype=np.float32)
    for rank_idx, letter in enumerate(cleaned):
        probs[OPTION_TO_INDEX[letter]] = weights_by_rank[rank_idx]
    probs = probs / probs.sum()
    return probs


def find_stage2_model_dir() -> str:
    env_path = os.environ.get("MCQ_STAGE2_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"MCQ_STAGE2_MODEL_PATH does not exist: {env_path}")

    final_dir = Path(STAGE2_BASE_DIR) / "final"
    if final_dir.exists():
        return str(final_dir)

    ckpts = sorted(Path(STAGE2_BASE_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        return str(ckpts[-1])

    raise FileNotFoundError(
        "No Stage 2 checkpoint found. Expected one of: "
        "./stage2_out/final, ./stage2_out/checkpoint-*, or MCQ_STAGE2_MODEL_PATH."
    )


def build_features(df: pd.DataFrame, tokenizer, max_len: int = 256) -> List[Dict[str, List[int]]]:
    questions = []
    options = []
    for _, row in df.iterrows():
        for option in OPTION_COLS:
            questions.append(str(row["prompt"]))
            options.append(str(row[option]))

    tokens = tokenizer(questions, options, truncation=True, max_length=max_len, padding=False)

    features = []
    n = len(df)
    for i in range(n):
        start = i * 5
        end = start + 5
        feat = {k: v[start:end] for k, v in tokens.items()}
        features.append(feat)
    return features


def predict_deberta_probs(df: pd.DataFrame, model_dir: str, batch_size: int = 8) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    features = build_features(df, tokenizer)
    loader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=MultipleChoiceCollator(tokenizer))

    logits_list = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits_list.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    return softmax_np(logits)


def load_llm_cache(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM cache not found: {path}")
    with open(path) as f:
        return json.load(f)


def read_cached_ranking(cache: Dict[str, List[str]], split_name: str, row_id: int) -> Tuple[List[str], str]:
    namespaced = [f"{split_name}:{row_id}", f"train:{row_id}" if split_name == "val" else f"test:{row_id}"]
    for key in namespaced:
        if key in cache:
            return cache[key], key

    key = str(row_id)
    if key in cache:
        return cache[key], key

    return [], ""


def extract_llm_probs(df: pd.DataFrame, cache: Dict[str, List[str]], split_name: str, strict_val_cache: bool = True) -> np.ndarray:
    probs = []
    ambiguous_plain_id_rows = []
    missing_rows = []

    for _, row in df.iterrows():
        row_id = int(row["id"])
        ranking, source_key = read_cached_ranking(cache, split_name=split_name, row_id=row_id)

        if not ranking:
            missing_rows.append(row_id)
            continue

        if split_name == "val" and source_key == str(row_id) and 1 <= row_id <= 500:
            ambiguous_plain_id_rows.append(row_id)

        probs.append(rank_to_prob_dist(ranking))

    if missing_rows:
        raise ValueError(
            f"Missing {len(missing_rows)} {split_name} rows in LLM cache. "
            f"Examples: {missing_rows[:10]}"
        )

    if split_name == "val" and strict_val_cache and ambiguous_plain_id_rows:
        raise ValueError(
            "Validation LLM cache keys are ambiguous for overlapping IDs 1..500 "
            f"({len(ambiguous_plain_id_rows)} rows). "
            "Current cache stores plain 'id' keys that can be overwritten by test runs. "
            "To proceed correctly, provide split-aware validation cache entries "
            "(e.g., val:<id>/train:<id>) or run with --allow-ambiguous-val-cache "
            "to force usage of plain-id cache (not recommended)."
        )

    return np.array(probs, dtype=np.float32)


def probs_to_top3_lists(probs: np.ndarray) -> List[List[str]]:
    idx = np.argsort(-probs, axis=1)[:, :3]
    return [[OPTION_COLS[j] for j in row] for row in idx]


def evaluate_alphas(
    y_true: List[str],
    deberta_probs: np.ndarray,
    llm_probs: np.ndarray,
    alphas: List[float],
) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        fused = alpha * deberta_probs + (1.0 - alpha) * llm_probs
        top3 = probs_to_top3_lists(fused)
        score = map_at_3(y_true, top3)
        rows.append({"alpha": alpha, "map3": score})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Stage 5 ensemble: DeBERTa + LLM-RAG")
    parser.add_argument("--allow-ambiguous-val-cache", action="store_true", help="Allow plain-id LLM cache usage for val IDs 1..500 (not recommended).")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading context-enriched datasets...")
    train_ctx = pd.read_csv(f"{CORPUS_DIR}/train_with_context.csv")
    test_ctx = pd.read_csv(f"{CORPUS_DIR}/test_with_context.csv")

    print("Applying leakage-free group-aware split (same as Stage 2/4)...")
    train_split, val_split = group_aware_split(train_ctx, test_size=0.2, seed=42)
    print(f"Train rows: {len(train_split)} | Val rows: {len(val_split)} | Test rows: {len(test_ctx)}")

    val_core = set(val_split["prompt"].map(clean_query))
    train_core = set(train_split["prompt"].map(clean_query))
    overlap = len(val_core & train_core)
    print(f"Prompt-string overlap train/val: {overlap} (grouping is enforced by src.splits.clean_query)")

    print("Locating Stage 2 best checkpoint/model...")
    model_dir = find_stage2_model_dir()
    print(f"Using Stage 2 model: {model_dir}")

    print("Running DeBERTa inference to get 5-way probabilities...")
    deberta_val_probs = predict_deberta_probs(val_split, model_dir=model_dir, batch_size=args.batch_size)
    deberta_test_probs = predict_deberta_probs(test_ctx, model_dir=model_dir, batch_size=args.batch_size)

    print("Loading Stage 4 cache and converting rankings to normalized score distributions...")
    llm_cache = load_llm_cache(LLM_CACHE_PATH)
    llm_val_probs = extract_llm_probs(
        val_split,
        llm_cache,
        split_name="val",
        strict_val_cache=not args.allow_ambiguous_val_cache,
    )
    llm_test_probs = extract_llm_probs(test_ctx, llm_cache, split_name="test", strict_val_cache=False)

    y_val = val_split["answer"].tolist()

    print("Evaluating alpha sweep...")
    results = evaluate_alphas(y_val, deberta_val_probs, llm_val_probs, DEFAULT_ALPHAS)
    results_path = f"{OUTPUT_DIR}/stage5_ensemble_results.csv"
    results.to_csv(results_path, index=False)

    print("\nAlpha    MAP@3")
    for _, row in results.iterrows():
        print(f"{row['alpha']:.1f}      {row['map3']:.4f}")

    best_idx = results["map3"].idxmax()
    best_alpha = float(results.loc[best_idx, "alpha"])
    best_map3 = float(results.loc[best_idx, "map3"])

    stage2_map3 = float(results.loc[np.isclose(results["alpha"], 1.0), "map3"].iloc[0])
    stage4_map3 = float(results.loc[np.isclose(results["alpha"], 0.0), "map3"].iloc[0])

    print("\nSummary")
    print(f"Best alpha: {best_alpha:.1f}")
    print(f"Best ensemble MAP@3: {best_map3:.4f}")
    print(f"Stage 2 MAP@3 (alpha=1.0): {stage2_map3:.4f}")
    print(f"Stage 4 MAP@3 (alpha=0.0): {stage4_map3:.4f}")
    print(f"Improvement over Stage 2: {best_map3 - stage2_map3:+.4f}")
    print(f"Improvement over Stage 4: {best_map3 - stage4_map3:+.4f}")

    if best_map3 <= stage4_map3:
        print("Ensemble does not beat Stage 4 on this leakage-free validation split.")

    fused_test_probs = best_alpha * deberta_test_probs + (1.0 - best_alpha) * llm_test_probs
    test_top3 = probs_to_top3_lists(fused_test_probs)

    submission = pd.DataFrame({
        "ID": test_ctx["id"],
        "Prediction": [" ".join(row) for row in test_top3],
    })
    submission_path = f"{OUTPUT_DIR}/submission_stage5_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {submission_path}")
    print("Done.")


if __name__ == "__main__":
    main()
