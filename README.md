# Smart MCQ Solver

MCQ solving pipeline for knowledge-intensive multiple-choice questions.
The project combines a simple lexical baseline, a DeBERTa classifier,
local Wikipedia retrieval, and optional LLM reasoning.

## What this repository does

The goal is to predict the best answer choice among A–E and rank the
top 3 options for MAP@3 scoring.

High-level flow:

1. **Stage 1** builds a TF-IDF baseline.
2. **Stage 2** fine-tunes a DeBERTa multiple-choice model.
3. **Stage 3** builds a local Wikipedia knowledge base and retrieves evidence.
4. **Stage 4** sends question + retrieved evidence to Groq LLMs for ranking.

The code is organized so each stage can be run independently.

## Repository layout

- [src/data_utils.py](src/data_utils.py) — shared CSV loading and submission helpers.
- [src/map3.py](src/map3.py) — MAP@3 metric implementation.
- [src/stage1_baseline.py](src/stage1_baseline.py) — TF-IDF baseline.
- [src/stage2_deberta_train.py](src/stage2_deberta_train.py) — DeBERTa training.
- [src/stage3a_build_corpus.py](src/stage3a_build_corpus.py) — build Wikipedia corpus.
- [src/stage3b_build_index.py](src/stage3b_build_index.py) — chunk corpus and build FAISS index.
- [src/stage3c_retrieve.py](src/stage3c_retrieve.py) — retrieve context for each MCQ.
- [src/stage3d_inspect_retrieval.py](src/stage3d_inspect_retrieval.py) — inspect retrieval quality.
- [src/stage4_llm_reasoning.py](src/stage4_llm_reasoning.py) — Groq LLM reasoning stage.
- [src/rag/wiki_client.py](src/rag/wiki_client.py) — Wikipedia API/session helpers.

## Data files

The competition data is not committed to git.

Expected files:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

You can override the data location with:

```bash
export MCQ_DATA_DIR=/path/to/data
```

## Environment variables

Create a `.env` file in the project root for Stage 4:

```env
GROQ_API_KEY=gsk_...
MCQ_GROQ_MODEL=qwen/qwen3.8-27b
MCQ_GROQ_MODEL_SEQUENCE=qwen/qwen3.8-27b,groq/compound-mini,openai/gpt-oss-20b
MCQ_MAX_CONTEXT_CHARS=250
MCQ_MAX_NEW_TOKENS=16
MCQ_MIN_CALL_GAP=1.0
```

Other useful variables:

- `MCQ_DATA_DIR` — location of `train.csv` and `test.csv`
- `MCQ_SEARCH_LIMIT` — candidate titles per question in Stage 3a

## Setup

This project uses `uv`.

```bash
uv sync
```

If you want to install packages manually:

```bash
uv add pandas numpy scikit-learn transformers torch datasets accelerate \
  sentence-transformers huggingface_hub sentencepiece protobuf requests \
  rank-bm25 faiss-cpu groq python-dotenv
```

## Stage-by-stage documentation

### Stage 1 — Baseline TF-IDF ranking

[src/stage1_baseline.py](src/stage1_baseline.py)

What it does:

- Loads `train.csv` and `test.csv`.
- Splits train into train/validation.
- Fits a TF-IDF vectorizer on the prompt and option text.
- Scores each option against the question using cosine similarity.
- Ranks A–E and evaluates with MAP@3.
- Writes `outputs/submission_baseline_tfidf.csv`.

Why it exists:

- Gives a quick CPU-only baseline.
- Confirms the data pipeline and metric are working.

Run:

```bash
uv run python -m src.stage1_baseline
```

### Stage 2 — DeBERTa multiple-choice fine-tuning

[src/stage2_deberta_train.py](src/stage2_deberta_train.py)

What it does:

- Reads train/test CSVs.
- Tokenizes each question-option pair.
- Fine-tunes `microsoft/deberta-v3-base` as a multiple-choice classifier.
- Evaluates accuracy and MAP@3 on a validation split.
- Predicts test labels and writes a submission file.

Important notes:

- This stage is much heavier than Stage 1.
- It is intended for GPU environments like Kaggle or Colab.
- It uses a stratified train/validation split by answer label.

Run:

```bash
uv run python -m src.stage2_deberta_train
```

### Stage 3a — Build the local Wikipedia corpus

[src/stage3a_build_corpus.py](src/stage3a_build_corpus.py)

What it does:

- Cleans the question prompts.
- Searches Wikipedia once per unique cleaned query.
- Collects candidate page titles.
- Fetches article text once per unique title.
- Saves a corpus to `data_enriched/corpus.jsonl`.

Caches:

- `data_enriched/titles_cache.json`
- `data_enriched/articles_cache.json`

This stage is network-bound, but only runs once to build the corpus.

Run:

```bash
uv run python -m src.stage3a_build_corpus
```

### Stage 3b — Chunk the corpus and build indexes

[src/stage3b_build_index.py](src/stage3b_build_index.py)

What it does:

- Loads `corpus.jsonl`.
- Cleans article text.
- Splits each article into chunks of about 300 words with 50-word overlap.
- Saves chunk metadata to `data_enriched/chunks.jsonl`.
- Embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`.
- Builds a FAISS index at `data_enriched/chunks.faiss`.

Notes:

- This is fully local.
- BM25 is rebuilt from `chunks.jsonl` during retrieval, so no separate BM25 file is saved.

Run:

```bash
uv run python -m src.stage3b_build_index
```

### Stage 3c — Hybrid retrieval for each MCQ

[src/stage3c_retrieve.py](src/stage3c_retrieve.py)

What it does:

- Cleans each question prompt.
- Searches the local chunks with:
  - BM25 lexical matching
  - FAISS embedding similarity
- Fuses the rankings with Reciprocal Rank Fusion.
- Removes duplicate titles so the final context is not repetitive.
- Stores the result in:
  - `data_enriched/train_with_context.csv`
  - `data_enriched/test_with_context.csv`

Output columns include:

- original MCQ fields
- `retrieved_context`
- `retrieved_sources`

Run:

```bash
uv run python -m src.stage3c_retrieve
```

### Stage 3d — Inspect retrieval quality

[src/stage3d_inspect_retrieval.py](src/stage3d_inspect_retrieval.py)

What it is for:

- Manually inspect retrieved passages.
- Check whether evidence is relevant.
- Debug bad queries or poor chunking.

This is an analysis/debugging helper rather than a pipeline stage.

Run:

```bash
uv run python -m src.stage3d_inspect_retrieval
```

### Stage 4 — Groq LLM reasoning over retrieved evidence

[src/stage4_llm_reasoning.py](src/stage4_llm_reasoning.py)

What it does:

- Loads `train_with_context.csv` and `test_with_context.csv`.
- Builds a group-aware validation split from the training data.
- Sends the question, options, and retrieved context to Groq.
- Asks the LLM to rank all 5 options from best to worst.
- Evaluates validation MAP@3.
- Writes `outputs/submission_stage4_llm_rag.csv`.

Important behavior:

- Loads `.env` automatically from the project root.
- Saves cache to `data_enriched/llm_cache.json`.
- Resumes from cache after interruption.
- Falls back across multiple accessible Groq models.

Run:

```bash
uv run python -m src.stage4_llm_reasoning
```

## Metric

[src/map3.py](src/map3.py) implements MAP@3, which is used across the
project to compare ranking quality consistently.

Interpretation:

- If the correct answer is ranked 1st, score = 1.0.
- If ranked 2nd, score = 0.5.
- If ranked 3rd, score = 0.333...
- Otherwise, score = 0.0.

## Outputs

Typical generated files:

- `outputs/submission_baseline_tfidf.csv`
- `outputs/submission_stage4_llm_rag.csv`
- `data_enriched/corpus.jsonl`
- `data_enriched/chunks.jsonl`
- `data_enriched/chunks.faiss`
- `data_enriched/train_with_context.csv`
- `data_enriched/test_with_context.csv`
- `data_enriched/llm_cache.json`

## Suggested workflow

1. Run Stage 1 to verify the pipeline.
2. Run Stage 3a, 3b, 3c to build local retrieval.
3. Inspect retrieval with Stage 3d.
4. Run Stage 4 only after retrieval is working.
5. Train Stage 2 when GPU is available.

## Current status

- Stage 1: baseline available.
- Stage 2: DeBERTa training available.
- Stage 3: local Wikipedia RAG pipeline available.
- Stage 4: Groq reasoning available with caching and model fallback.

## Results

Verified validation scores:

| Stage | Method | Val metric |
|---|---|---|
| 1 | TF-IDF baseline | 0.3204 MAP@3 |
| 2 | DeBERTa-v3 fine-tuned | 0.2425 accuracy, 0.4104 MAP@3 |
| 4 | Groq LLM + retrieved context | 0.6850 MAP@3 |

Generated submissions:

- [outputs/submission_baseline_tfidf.csv](outputs/submission_baseline_tfidf.csv)
- [outputs/submission_stage4_llm_rag.csv](outputs/submission_stage4_llm_rag.csv)

## Notes

- The repository is designed to run locally for retrieval stages.
- Stage 4 depends on Groq API access and available quota.
- If a Groq model is unavailable, the script automatically tries another one.
