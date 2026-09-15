"""
Stage 3a: Build the local knowledge base (ONE-TIME network cost).

Two phases, each cached and resumable:
  Phase 1 -- search: for every unique cleaned question, get candidate page
             titles (cheap call, titles only, no content).
  Phase 2 -- fetch:   for every unique title found across ALL questions,
             fetch the full article body ONCE (many questions share titles,
             e.g. "Quantum mechanics" gets hit by dozens of questions).

After this script finishes, stages 3b/3c never touch the network again.

Usage:
    uv run python -m src.stage3a_build_corpus
"""
import os
import json
import signal
import sys
import pandas as pd

from src.rag.wiki_client import build_session, clean_query, search_titles, opensearch_titles, fetch_full_text

DATA_DIR = os.environ.get("MCQ_DATA_DIR", "./data")
CORPUS_DIR = "./data_enriched"
TITLES_CACHE = f"{CORPUS_DIR}/titles_cache.json"
ARTICLES_CACHE = f"{CORPUS_DIR}/articles_cache.json"
SEARCH_LIMIT = int(os.environ.get("MCQ_SEARCH_LIMIT", "5"))
SAVE_EVERY = 25


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    titles_cache = load_json(TITLES_CACHE)   # query -> [titles]
    articles_cache = load_json(ARTICLES_CACHE)  # title -> full text
    print(f"Loaded {len(titles_cache)} cached queries, {len(articles_cache)} cached articles.")

    session = build_session()

    def handle_sigint(sig, frame):
        print("\nInterrupted -- saving caches before exit...")
        save_json(titles_cache, TITLES_CACHE)
        save_json(articles_cache, ARTICLES_CACHE)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ---- Collect all unique cleaned questions across train + test ----
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    all_prompts = pd.concat([train["prompt"], test["prompt"]]).tolist()
    unique_queries = sorted({clean_query(p) for p in all_prompts})
    print(f"{len(unique_queries)} unique questions to search (train+test combined).")

    # ---- Phase 1: search titles (full-text search + title-prefix search, unioned) ----
    print("\nPhase 1: searching for candidate page titles...")
    for i, q in enumerate(unique_queries):
        if q in titles_cache:
            continue
        try:
            fulltext = search_titles(session, q, limit=SEARCH_LIMIT)
        except Exception as e:
            print(f"  [search FAILED] '{q[:60]}...' -> {e}")
            fulltext = []
        try:
            prefix = opensearch_titles(session, q, limit=SEARCH_LIMIT)
        except Exception as e:
            print(f"  [opensearch FAILED] '{q[:60]}...' -> {e}")
            prefix = []
        # union, prefix-match titles first since they're usually more precise
        # for compound/technical terms (e.g. "Einstein@Home")
        combined = list(dict.fromkeys(prefix + fulltext))
        titles_cache[q] = combined
        if (i + 1) % SAVE_EVERY == 0:
            save_json(titles_cache, TITLES_CACHE)
            print(f"  searched {i + 1}/{len(unique_queries)}")
    save_json(titles_cache, TITLES_CACHE)

    # ---- Phase 2: fetch full articles for every unique title ----
    all_titles = sorted({t for titles in titles_cache.values() for t in titles})
    print(f"\nPhase 2: fetching full text for {len(all_titles)} unique articles...")
    for i, title in enumerate(all_titles):
        if title in articles_cache:
            continue
        try:
            articles_cache[title] = fetch_full_text(session, title)
        except Exception as e:
            print(f"  [fetch FAILED] '{title[:60]}...' -> {e}")
            articles_cache[title] = ""
        if (i + 1) % SAVE_EVERY == 0:
            save_json(articles_cache, ARTICLES_CACHE)
            print(f"  fetched {i + 1}/{len(all_titles)}")
    save_json(articles_cache, ARTICLES_CACHE)

    # ---- Write final corpus.jsonl ----
    corpus_path = f"{CORPUS_DIR}/corpus.jsonl"
    n_written = 0
    with open(corpus_path, "w") as f:
        for title, text in articles_cache.items():
            if text.strip():
                f.write(json.dumps({"title": title, "text": text}) + "\n")
                n_written += 1

    print(f"\nDone. Wrote {n_written} articles to {corpus_path}")
    empty = sum(1 for v in articles_cache.values() if not v.strip())
    print(f"({empty} title(s) returned no content, e.g. redirects/disambig pages)")


if __name__ == "__main__":
    main()