"""Shared Wikipedia access used ONLY by stage3a (corpus building).
Stages 3b/3c never touch the network -- they run on the local corpus.
"""
import re
import time
import requests
from requests.adapters import HTTPAdapter, Retry

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "mcq-solver-resume-project/0.2 (contact: none; personal project)"}
REQUEST_DELAY_SECONDS = 0.4


def build_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(HEADERS)
    return session


PREFIX_PATTERNS = [
    r"^select the most accurate option:\s*",
    r"^identify the correct statement:\s*",
    r"^choose the correct answer:\s*",
    r"^pick the best possible answer:\s*",
    r"^determine the correct option:\s*",
    r"^which of the following is correct\?\s*",
]
SUFFIX_PATTERNS = [
    r"\s*based on the given context\.?$",
    r"\s*among the listed options\.?$",
    r"\s*from the following choices\.?$",
    r"\s*carefully\.?$",
]


def clean_query(prompt):
    q = prompt.strip()
    for pat in PREFIX_PATTERNS:
        q = re.sub(pat, "", q, flags=re.IGNORECASE)
    for pat in SUFFIX_PATTERNS:
        q = re.sub(pat, "", q, flags=re.IGNORECASE)
    return q.strip()


def search_titles(session, query, limit=3):
    """ONE lightweight call: just page titles, no content yet."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    r = session.get(WIKI_API, params=params, timeout=15)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    data = r.json()
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def opensearch_titles(session, query, limit=3):
    """Prefix/near-title matching -- catches exact article names (e.g.
    'Einstein@Home') that plain full-text search under-ranks because the
    term is rare/compound and gets diluted by generic pages."""
    params = {
        "action": "opensearch",
        "search": query,
        "limit": limit,
        "namespace": 0,
        "format": "json",
    }
    r = session.get(WIKI_API, params=params, timeout=15)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    data = r.json()
    return data[1] if len(data) > 1 else []


# Cut article text at these boilerplate section headers (plaintext extracts
# keep them as literal "== References ==" style lines).
BOILERPLATE_HEADERS = [
    "== References ==", "== See also ==", "== External links ==",
    "== Further reading ==", "== Notes ==", "== Bibliography ==",
]


def fetch_full_text(session, title):
    """ONE call: full plaintext body of a single Wikipedia page."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    r = session.get(WIKI_API, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = page.get("extract", "")
        if not text:
            continue
        for header in BOILERPLATE_HEADERS:
            idx = text.find(header)
            if idx != -1:
                text = text[:idx]
        return text.strip()
    return ""