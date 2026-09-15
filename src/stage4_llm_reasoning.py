"""
Stage 4: Feed (question + retrieved context + 5 options) to an LLM and ask
it to rank all 5 options. This is the "Gen AI reasoning" stage -- retrieval
gives facts, the LLM does the judgment call.

Uses Groq's free API tier -- no local compute burden, fast inference,
no cost.

One-time setup:
    uv add groq
    export GROQ_API_KEY=gsk_...      # free key at console.groq.com

Then:
    uv run python -m src.stage4_llm_reasoning

Runs ONLY on the validation split (for scoring) + test split (for the
final submission) -- no need to burn API calls on the full train set,
since this stage doesn't train anything.
"""
import os
import re
import json
import time
import signal
import sys
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.splits import group_aware_split
from src.map3 import map_at_3

CORPUS_DIR = "./data_enriched"
CACHE_PATH = f"{CORPUS_DIR}/llm_cache.json"
OPTION_COLS = ["A", "B", "C", "D", "E"]
SAVE_EVERY = 20
MAX_RETRIES = 5


class DailyTokenLimitReached(RuntimeError):
    pass


class ModelNotAvailable(RuntimeError):
    pass

DEFAULT_MODEL = os.environ.get("MCQ_GROQ_MODEL", "openai/gpt-oss-20b")
MODEL_SEQUENCE = [
    name.strip()
    for name in os.environ.get(
        "MCQ_GROQ_MODEL_SEQUENCE",
        "qwen/qwen3.8-27b,groq/compound-mini,openai/gpt-oss-20b",
    ).split(",")
    if name.strip()
]
MODEL_FALLBACKS = [DEFAULT_MODEL, *MODEL_SEQUENCE, "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MAX_CONTEXT_CHARS = int(os.environ.get("MCQ_MAX_CONTEXT_CHARS", "300"))
MAX_NEW_TOKENS = int(os.environ.get("MCQ_MAX_NEW_TOKENS", "20"))
MIN_SECONDS_BETWEEN_CALLS = float(os.environ.get("MCQ_MIN_CALL_GAP", "2.0"))

SYSTEM_PROMPT = (
    "Rank the 5 answer options from most to least likely correct. "
    "Use the context if helpful. Reply with JSON only: "
    '{"ranking":["A","B","C","D","E"]}'
)


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def parse_ranking(text):
    """Robust parse: try JSON first, fall back to regex-extracting A-E letters."""
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        obj = json.loads(cleaned)
        ranking = obj["ranking"]
        if set(ranking) == set(OPTION_COLS) and len(ranking) == 5:
            return ranking
    except Exception:
        pass
    letters = re.findall(r"\b([A-E])\b", text)
    seen = []
    for l in letters:
        if l not in seen:
            seen.append(l)
    for opt in OPTION_COLS:
        if opt not in seen:
            seen.append(opt)
    return seen[:5]


def build_user_message(row, context):
    def squash(text):
        return re.sub(r"\s+", " ", str(text)).strip()

    options_text = " | ".join(f"{c}: {squash(row[c])}" for c in OPTION_COLS)
    context = squash(context)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rstrip() + "..."
    question = squash(row["prompt"])
    return f"CTX: {context if context else '(none)'}\nQ: {question}\nOPTS: {options_text}"


def resolve_model(client):
    """Pick the first usable model from the preferred/fallback list.

    We prefer not to discover model access by burning chat tokens. If the SDK
    can list accessible models, use that to choose a valid one up front.
    Otherwise, fall back to the configured/default model.
    """
    preferred = []
    for name in MODEL_FALLBACKS:
        if name and name not in preferred:
            preferred.append(name)

    try:
        resp = client.models.list()
        available = set()
        for item in getattr(resp, "data", []) or []:
            model_id = getattr(item, "id", None)
            if model_id:
                available.add(model_id)
        for candidate in preferred:
            if candidate in available:
                return candidate
        if available:
            # If the API returns models but none of our hard-coded candidates
            # match, use a deterministic accessible model from the known list.
            return sorted(available)[0]
    except Exception:
        pass

    return preferred[0]


def resolve_model_sequence(client):
    """Return accessible model candidates in a deterministic order."""
    preferred = []
    for name in MODEL_FALLBACKS:
        if name and name not in preferred:
            preferred.append(name)

    try:
        resp = client.models.list()
        available = []
        for item in getattr(resp, "data", []) or []:
            model_id = getattr(item, "id", None)
            if model_id:
                available.append(model_id)

        ordered = [m for m in preferred if m in available]
        for model_id in sorted(available):
            if model_id not in ordered:
                ordered.append(model_id)
        return ordered or preferred
    except Exception:
        return preferred


def ask_llm(client, model, row, context):
    user_msg = build_user_message(row, context)
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS)  # proactive pacing, not just reactive backoff
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return parse_ranking(resp.choices[0].message.content)
        except Exception as e:
            err = str(e).lower()

            if "model_not_found" in err or "does not exist or you do not have access" in err:
                raise ModelNotAvailable(str(e)) from e

            # Groq daily token exhaustion cannot be fixed by retrying.
            # Save progress and stop immediately so the run can be resumed later.
            if "tokens per day" in err or "rate_limit_exceeded" in err:
                raise DailyTokenLimitReached(str(e)) from e

            # Groq returns a Retry-After header on 429s -- honor it exactly
            # instead of guessing with exponential backoff.
            retry_after = None
            resp_obj = getattr(e, "response", None)
            if resp_obj is not None:
                retry_after = resp_obj.headers.get("retry-after")
            wait = float(retry_after) if retry_after else 2 ** attempt
            print(f"  [LLM call failed, attempt {attempt+1}] {e} -- waiting {wait}s")
            time.sleep(wait)
    return OPTION_COLS.copy()  # fallback: arbitrary order if all retries fail


def get_rankings(df, client, models, cache, id_col="id", label=""):
    rankings = {}
    completed = True
    model_idx = 0
    for i, (_, row) in enumerate(df.iterrows()):
        key = str(row[id_col])
        if key in cache:
            rankings[key] = cache[key]
            continue
        context = row.get("retrieved_context", "")
        while True:
            if model_idx >= len(models):
                print("\n[STOP] No usable Groq models remain.")
                save_cache(cache)
                completed = False
                break

            model = models[model_idx]
            try:
                ranking = ask_llm(client, model, row, context)
                break
            except ModelNotAvailable as e:
                print(f"  [model skipped] {model}: {e}")
                model_idx += 1
                continue
            except DailyTokenLimitReached as e:
                print(f"  [model quota hit] {model}: {e}")
                model_idx += 1
                continue

        if not completed:
            break

        cache[key] = ranking
        rankings[key] = ranking
        if (i + 1) % SAVE_EVERY == 0:
            save_cache(cache)
            print(f"  [{label}] {i + 1}/{len(df)}")
    save_cache(cache)
    return rankings, completed


def main():
    if "GROQ_API_KEY" not in os.environ:
        print("ERROR: set GROQ_API_KEY first (free key at console.groq.com).")
        sys.exit(1)

    client = Groq()
    models = resolve_model_sequence(client)
    if not models:
        print("ERROR: no Groq models available for this key.")
        sys.exit(1)

    print(f"Provider: Groq ({models[0]}) -- free tier.")
    print(f"Model sequence: {', '.join(models[:5])}{' ...' if len(models) > 5 else ''}")
    if DEFAULT_MODEL not in models:
        print(f"Requested model {DEFAULT_MODEL!r} is not available; using the first available model instead.")

    cache = load_cache()
    print(f"Loaded cache with {len(cache)} entries.")

    def handle_sigint(sig, frame):
        print("\nInterrupted -- saving cache before exit...")
        save_cache(cache)
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    train_ctx = pd.read_csv(f"{CORPUS_DIR}/train_with_context.csv")
    test_ctx = pd.read_csv(f"{CORPUS_DIR}/test_with_context.csv")

    # Same group-aware split used in Stage 2 -- see src/splits.py.
    tr, val = group_aware_split(train_ctx, test_size=0.2, seed=42)
    print(f"Validation set: {len(val)} rows (group-aware, no near-duplicate leakage)")

    print("\nRunning LLM reasoning on validation set...")
    val_rankings, val_completed = get_rankings(val, client, models, cache, label="val")
    if not val_completed:
        print("\nValidation stopped early due to Groq daily token limits.")
        print("Re-run tomorrow (or with a cheaper model / shorter context); cache will resume from where it stopped.")
        sys.exit(0)

    val_top3 = [val_rankings[str(row["id"])][:3] for _, row in val.iterrows()]
    score = map_at_3(val["answer"].tolist(), val_top3)
    print(f"\nStage 4 LLM+RAG -- Validation MAP@3: {score:.4f}")

    print("\nRunning LLM reasoning on test set...")
    test_rankings, test_completed = get_rankings(test_ctx, client, models, cache, label="test")
    if not test_completed:
        print("\nTest inference stopped early due to Groq daily token limits.")
        print("Re-run later; cache will resume from where it stopped.")
        sys.exit(0)

    test_top3 = [test_rankings[str(row["id"])][:3] for _, row in test_ctx.iterrows()]

    os.makedirs("./outputs", exist_ok=True)
    submission = pd.DataFrame({
        "ID": test_ctx["id"],
        "Prediction": [" ".join(r) for r in test_top3],
    })
    submission.to_csv("./outputs/submission_stage4_llm_rag.csv", index=False)
    print("Saved ./outputs/submission_stage4_llm_rag.csv")
    print(submission.head())


if __name__ == "__main__":
    main()