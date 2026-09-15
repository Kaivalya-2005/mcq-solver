"""
Stage 3b: Chunk the local corpus and build BM25 + FAISS indexes.

Pure local computation -- no network calls. Rerun freely while tuning
chunk size, overlap, or the embedding model.

Usage:
    uv run python -m src.stage3b_build_index
"""
import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_DIR = "./data_enriched"
CORPUS_PATH = f"{CORPUS_DIR}/corpus.jsonl"
CHUNKS_PATH = f"{CORPUS_DIR}/chunks.jsonl"
FAISS_INDEX_PATH = f"{CORPUS_DIR}/chunks.faiss"

CHUNK_WORDS = 300
OVERLAP_WORDS = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus():
    articles = []
    with open(CORPUS_PATH) as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_article(title, text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_words - overlap
    idx = 0
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < 30:  # skip tiny tail fragments
            break
        chunks.append({"title": title, "text": " ".join(piece), "chunk_index": idx})
        idx += 1
        if start + chunk_words >= len(words):
            break
    return chunks


def main():
    print("Loading corpus...")
    articles = load_corpus()
    print(f"{len(articles)} articles loaded.")

    print("Chunking...")
    chunks = []
    for art in articles:
        text = clean_text(art["text"])
        chunks.extend(chunk_article(art["title"], text))
    for i, c in enumerate(chunks):
        c["id"] = i
    print(f"{len(chunks)} chunks created (~{CHUNK_WORDS} words each, {OVERLAP_WORDS} overlap).")

    with open(CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Saved chunks to {CHUNKS_PATH}")

    print(f"\nEmbedding {len(chunks)} chunks with {EMBED_MODEL} (CPU is fine for this size)...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index ({index.ntotal} vectors, dim={embeddings.shape[1]}) to {FAISS_INDEX_PATH}")

    print("\nStage 3b done. BM25 is rebuilt on the fly from chunks.jsonl at retrieval "
          "time (cheap for a few thousand chunks) -- nothing else to save for it.")


if __name__ == "__main__":
    main()