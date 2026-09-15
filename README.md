# Smart MCQ Solver

MCQ solving pipeline for knowledge-intensive multiple-choice questions.
The project combines a simple lexical baseline, a DeBERTa classifier,
local Wikipedia retrieval, LLM reasoning, and a final ensemble blend.

## What this repository does

The goal is to predict the best answer choice among A–E and rank the
top 3 options for MAP@3 scoring.

High-level flow:

1. **Stage 1** builds a TF-IDF baseline.
2. **Stage 2** fine-tunes a DeBERTa multiple-choice model.
3. **Stage 3** builds a local Wikipedia knowledge base and retrieves evidence.
4. **Stage 4** sends question + retrieved evidence to Groq LLMs for ranking.
5. **Stage 5** creates an optimized ensemble blending DeBERTa probabilities and LLM rankings.

The code is organized so each stage can be run independently.

## Repository layout

- [src/data_utils.py](src/data_utils.py) — shared CSV loading and submission helpers.
- [src/map3.py](src/map3.py) — MAP@3 metric implementation.
- [src/splits.py](src/splits.py) — prompt cleaning and group-aware leak-free train/val split.
- [src/stage1_baseline.py](src/stage1_baseline.py) — TF-IDF baseline.
- [src/stage2_deberta_train.py](src/stage2_deberta_train.py) — DeBERTa training.
- [src/stage3a_build_corpus.py](src/stage3a_build_corpus.py) — build Wikipedia corpus.
- [src/stage3b_build_index.py](src/stage3b_build_index.py) — chunk corpus and build FAISS index.
- [src/stage3c_retrieve.py](src/stage3c_retrieve.py) — retrieve context for each MCQ.
- [src/stage3d_inspect_retrieval.py](src/stage3d_inspect_retrieval.py) — inspect retrieval quality.
- [src/stage4_llm_reasoning.py](src/stage4_llm_reasoning.py) — Groq LLM reasoning stage.
- [src/stage5_ensemble.py](src/stage5_ensemble.py) — DeBERTa + LLM-RAG probability blending.
- [src/rag/wiki_client.py](src/rag/wiki_client.py) — Wikipedia API/session helpers.

## Data files

The competition data is not committed to git.

Expected files:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

You can override the data location with:

export MCQ_DATA_DIR=/path/to/data

## Environment variables

Create a `.env` file in the project root for Stage 4:

GROQ_API_KEY=gsk_...
MCQ_GROQ_MODEL=qwen/qwen3.8-27b
MCQ_GROQ_MODEL_SEQUENCE=qwen/qwen3.8-27b,groq/compound-mini,openai/gpt-oss-20b
MCQ_MAX_CONTEXT_CHARS=250
MCQ_MAX_NEW_TOKENS=16
MCQ_MIN_CALL_GAP=1.0

Current Stage 4 model priority:

1. `qwen/qwen3.8-27b` (primary)
2. `groq/compound-mini` (fallback)
3. `openai/gpt-oss-20b` (fallback)

At runtime, the script also appends any other models accessible to your Groq
key and automatically switches when a model is unavailable or quota-limited.

Other useful variables:

- `MCQ_DATA_DIR` — location of `train.csv` and `test.csv`
- `MCQ_SEARCH_LIMIT` — candidate titles per question in Stage 3a
- `MCQ_STAGE2_MODEL_PATH` — path to trained DeBERTa weights (useful for Kaggle kernels in Stage 5)

## Setup

This project uses `uv`.

uv sync

If you want to install packages manually:

uv add pandas numpy scikit-learn transformers torch datasets accelerate \
  sentence-transformers huggingface_hub sentencepiece protobuf requests \
  rank-bm25 faiss-cpu groq python-dotenv

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

Run:
uv run python -m src.stage1_baseline

### Stage 2 — DeBERTa multiple-choice fine-tuning
[src/stage2_deberta_train.py](src/stage2_deberta_train.py)

What it does:
- Reads train/test CSVs.
- Tokenizes each question-option pair.
- Fine-tunes `microsoft/deberta-v3-base` as a multiple-choice classifier.
- Evaluates accuracy and MAP@3 on a leakage-free validation split.
- Predicts test labels and writes a submission file.

*Note: This stage is heavy and intended for GPU environments (Kaggle/Colab).*

Run:
uv run python -m src.stage2_deberta_train

### Stage 3a — Build the local Wikipedia corpus
[src/stage3a_build_corpus.py](src/stage3a_build_corpus.py)

What it does:
- Cleans the question prompts, searches Wikipedia, and fetches article text.
- Saves a corpus to `data_enriched/corpus.jsonl`.

Run:
uv run python -m src.stage3a_build_corpus

### Stage 3b — Chunk the corpus and build indexes
[src/stage3b_build_index.py](src/stage3b_build_index.py)

What it does:
- Splits articles into overlapping chunks (~300 words).
- Embeds each chunk and builds a FAISS index at `data_enriched/chunks.faiss`.

Run:
uv run python -m src.stage3b_build_index

### Stage 3c — Hybrid retrieval for each MCQ
[src/stage3c_retrieve.py](src/stage3c_retrieve.py)

What it does:
- Searches the local chunks with BM25 and FAISS embeddings.
- Fuses rankings (Reciprocal Rank Fusion) and stores context to `data_enriched/train_with_context.csv` & `test_with_context.csv`.

Run:
uv run python -m src.stage3c_retrieve

### Stage 3d — Inspect retrieval quality
[src/stage3d_inspect_retrieval.py](src/stage3d_inspect_retrieval.py)

What it is for:
- Manual debugging to check passage relevance and chunking quality.

Run:
uv run python -m src.stage3d_inspect_retrieval

### Stage 4 — Groq LLM reasoning over retrieved evidence
[src/stage4_llm_reasoning.py](src/stage4_llm_reasoning.py)

What it does:
- Sends the question, options, and retrieved context to Groq API.
- Evaluates validation MAP@3 and writes `outputs/submission_stage4_llm_rag.csv`.
- Implements robust caching and automatic model fallback on rate limits.

Run:
uv run python -m src.stage4_llm_reasoning

### Stage 5 — DeBERTa + LLM-RAG Ensemble Blending
[src/stage5_ensemble.py](src/stage5_ensemble.py)

What it does:
- Loads Stage 2 DeBERTa probability distributions.
- Maps Stage 4 LLM ranked outputs to discrete probability vectors.
- Performs an alpha grid sweep (α ∈ [0.0, 1.0]) to find optimal weighting on the validation set.
- Evaluates MAP@3 for each alpha and writes the winning blend to `outputs/submission_stage5_ensemble.csv`.

Run:
uv run python -m src.stage5_ensemble

## Outputs

Typical generated files:

- `outputs/submission_baseline_tfidf.csv`
- `outputs/submission_stage2_deberta.csv`
- `outputs/submission_stage4_llm_rag.csv`
- `outputs/submission_stage5_ensemble.csv`
- `outputs/stage5_ensemble_results.csv`
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
4. Run Stage 4 to get LLM reasoning predictions on retrieved context.
5. Train Stage 2 DeBERTa multiple-choice classifier (requires GPU).
6. Run Stage 5 to sweep alphas and ensemble DeBERTa + LLM predictions.

## Current status

- Stage 1: baseline available.
- Stage 2: DeBERTa training available.
- Stage 3: local Wikipedia RAG pipeline available.
- Stage 4: Groq reasoning available with caching and model fallback.
- Stage 5: Alpha-sweep ensemble blending available.

## Results

Final verified scores using a leakage-free, group-aware cross-validation split:

| Stage | Method | Val MAP@3 | Public LB | Private LB |
|---|---|---|---|---|
| 1 | TF-IDF baseline | 0.3204 | - | - |
| 2 | DeBERTa-v3 fine-tuned | 0.4324 | - | - |
| 4 | Groq LLM + Wikipedia RAG | 0.6930 | - | - |
| 5 | Ensemble (DeBERTa + LLM, α=0.5) | **0.7041** | **0.64983** | **0.65211** |

*Note: The highly stable Public/Private Kaggle leaderboard scores demonstrate the robustness of the group-aware, leakage-free cross-validation strategy used during training.*