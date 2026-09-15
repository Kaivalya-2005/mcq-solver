"""
Stage 3c: For every MCQ, retrieve top evidence chunks using hybrid
BM25 + embedding search, fused with Reciprocal Rank Fusion (RRF).

Pure local computation -- no network calls, runs in seconds/minutes even
on CPU. Rerun freely while tuning top_k, fusion weights, etc.

Usage:
    uv run python -m src.stage3c_retrieve
"""
import os
import re
import json
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.rag.wiki_client import clean_query

DATA_DIR = os.environ.get("MCQ_DATA_DIR", "./data")
CORPUS_DIR = "./data_enriched"
CHUNKS_PATH = f"{CORPUS_DIR}/chunks.jsonl"
FAISS_INDEX_PATH = f"{CORPUS_DIR}/chunks.faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BM25_TOP_K = 10
FAISS_TOP_K = 10
FINAL_TOP_K = 4       # how many fused chunks to keep as context
RRF_K = 60             # standard RRF smoothing constant


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """rankings: list of ranked-id-lists (best first). Returns fused ranking."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def build_retriever():
    chunks = load_chunks()
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    embed_model = SentenceTransformer(EMBED_MODEL)
    # title -> id of its chunk_index==0 (the article intro)
    intro_chunk_by_title = {c["title"]: c["id"] for c in chunks if c.get("chunk_index") == 0}
    return chunks, bm25, faiss_index, embed_model, intro_chunk_by_title


def retrieve_for_query(query, chunks, bm25, faiss_index, embed_model, intro_chunk_by_title):
    # BM25 ranking
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = list(np.argsort(-bm25_scores)[:BM25_TOP_K])

    # FAISS ranking (cosine via normalized inner product)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, faiss_idx = faiss_index.search(q_emb.astype(np.float32), FAISS_TOP_K)
    faiss_ranked = list(faiss_idx[0])

    fused_ids = reciprocal_rank_fusion([bm25_ranked, faiss_ranked])

    # Dedupe by article title so we don't return 4 chunks of the same page.
    # For the FIRST (highest-ranked) title only, swap in its intro chunk --
    # for definitional "What is X" questions the intro is almost always the
    # most useful passage, and a plain top-fused chunk can land mid-article
    # (e.g. a "cultural references" section) even when the right article
    # was found.
    seen_titles = set()
    picked = []
    for doc_id in fused_ids:
        title = chunks[doc_id]["title"]
        if title in seen_titles:
            continue
        if not picked and title in intro_chunk_by_title:
            doc_id = intro_chunk_by_title[title]
        picked.append(doc_id)
        seen_titles.add(title)
        if len(picked) >= FINAL_TOP_K:
            break

    return [chunks[i]["text"] for i in picked], [chunks[i]["title"] for i in picked]


def add_retrieved_context(df, chunks, bm25, faiss_index, embed_model, intro_chunk_by_title):
    contexts, sources = [], []
    for prompt in df["prompt"]:
        q = clean_query(prompt)
        texts, titles = retrieve_for_query(q, chunks, bm25, faiss_index, embed_model, intro_chunk_by_title)
        contexts.append(" ".join(texts))
        sources.append("; ".join(titles))
    out = df.copy()
    out["retrieved_context"] = contexts
    out["retrieved_sources"] = sources
    return out


def main():
    print("Loading local index (BM25 + FAISS)...")
    chunks, bm25, faiss_index, embed_model, intro_chunk_by_title = build_retriever()
    print(f"{len(chunks)} chunks indexed.")

    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")

    print("Retrieving for train set...")
    train_ctx = add_retrieved_context(train, chunks, bm25, faiss_index, embed_model, intro_chunk_by_title)
    print("Retrieving for test set...")
    test_ctx = add_retrieved_context(test, chunks, bm25, faiss_index, embed_model, intro_chunk_by_title)

    train_ctx.to_csv(f"{CORPUS_DIR}/train_with_context.csv", index=False)
    test_ctx.to_csv(f"{CORPUS_DIR}/test_with_context.csv", index=False)

    empty_frac = (train_ctx["retrieved_context"].str.len() == 0).mean()
    avg_len = train_ctx["retrieved_context"].str.len().mean()
    print(f"\nSaved to {CORPUS_DIR}/")
    print(f"Rows with EMPTY retrieved context: {empty_frac:.1%}")
    print(f"Avg context length (chars): {avg_len:.0f}")
    print("\nSample:")
    print(train_ctx[["prompt", "retrieved_sources"]].iloc[0])


if __name__ == "__main__":
    main()