# Step 1: Set up the Knowledge Base and Implement Search Tools
# We will begin with a small corpus of a dataset as an example, and show how we can build a search method that finds the most related document given an input query. 
# We will use this corpus as the database and the method as the action space for building a GPT-based agent.


# First, let's import necessary packages

from __future__ import annotations
import sys
from pathlib import Path
# When running this module directly (python app/db/tfidf_search.py) ensure the
# repository root is on sys.path so absolute imports like `app.db.queries` work.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dataclasses import dataclass, field, asdict
import string
from typing import Callable, Dict, List, Tuple, Optional, Any
import json, math, re, textwrap, random, os, sys
import math
from collections import Counter, defaultdict
from .engine import get_session
from .queries import return_text_stream
import numpy as np

# Get initial data
def get_corpus(bucketLevel: int = None, subject: str = None, credits: int = None):
    with get_session() as session:
        # Collect recipe strings in chunks. `return_text_stream` yields strings;
        # convert each chunk to a list and extend the master list so `sentences`
        # is a flat list of strings (SentenceTransformer expects an indexable list).
        CORPUS = []

        for x in range(0, 6001, 1000):
            chunk = list(return_text_stream(session=session, bucketLevel=bucketLevel, credits=credits, subject=subject, offset=x, n=1000))

            # filter out any empty strings
            chunk = [s for s in chunk if s]
            CORPUS.extend(chunk)
        return CORPUS

# 1.  Tokenize the document into words
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

# irrelevent words to remove
STOPWORDS = {
    "a", "an", "the",
    "is", "are", "am", "was", "were", "be", "been", "being",
    "i", "me", "my", "you", "your", "u",
    "and", "or", "but", "so",
    "to", "of", "in", "on", "for", "at", "with",
    "this", "that", "it", "its",
    "because", "not",
    "rt"
}

def doc_tokens_vocab(CORPUS):
    #     Get all the words of each document in the corpus
    DOC_TOKENS = [[t for t in tokenize(doc) if t not in STOPWORDS] for doc in CORPUS]

    #     Get all the words from the corpus
    VOCAB = sorted(set(t for doc in DOC_TOKENS for t in doc))

    return DOC_TOKENS, VOCAB

# 2.  Compute term frequency (TF) for each doc
def compute_tf(tokens: List[str]) -> Dict[str, float]:
    # Input: A list of all the words in a document
    # Output: A dictionary of the frequency of each word
    result = {}
    for x in tokens:
        result[x] = result.get(x, 0) + 1
    return result

# 3.   Compute the document frequency across corpus: how many docs does a word appear?
def compute_df(doc_tokens: List[List[str]]) -> Dict[str, float]:
    # Input: A list of lists of tokens in each document
    # Output: A dictionary of the counts of each word appearing across the documents
    df = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] += 1
    return df

def compute_idf(DOC_TOKENS, VOCAB):
    #     Compute the inverse document frequency (higher for rarer terms), in which we use a smoothed variant
    DF = compute_df(DOC_TOKENS) # Get the DF
    N_DOC = len(DOC_TOKENS) # number of docs
    IDF = {t: math.log((N_DOC + 1) / (DF[t] + 0.5)) + 1 for t in VOCAB} # Inverse document frequency

    return IDF

# 4.   We compute TF-IDF vectors for each document, which is the product between
def tfidf_vector(tokens: List[str], IDF) -> Dict[str, float]:
    # Input: A list of words in a document
    # Output: A dictionary of tf-idf score of each word
    tf = compute_tf(tokens)
    vec = {t: tf[t] * IDF.get(t, 0.0) for t in tf}
    return vec

def doc_vecs(DOC_TOKENS, IDF):
    return [tfidf_vector(tokens, IDF) for tokens in DOC_TOKENS]

# 5.   We compute the cosine similarity for the search
def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Inputs: Two dictrionaries of tf-idf vectors of two document
    # Output: The cosine similarity scalar between the two vector

    if not a or not b:
        return 0.0

    # ===== TODO =====
    # Compute the cosine similarity between two tf-idf vectors
    # Notice that they are two dictionaries and could have missing keys
    common_words = filter(lambda x: x in b, a.keys())
    # compute dot product
    dot_prod = sum(a[word] * b[word] for word in common_words)

    
    # compute norms
    a_norm = np.linalg.norm(np.array(list(a.values())), ord=2)
    b_norm = np.linalg.norm(np.array(list(b.values())), ord=2)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    similarity = dot_prod/(a_norm * b_norm)
    # ===== TODO =====
    return similarity

# 6.   We implement a search method based on the cosine similarity, which finds the documents with the highest similarity scores as the top-k search results.
def search_corpus(query: str, DOC_VECS, CORPUS, IDF, k: int = 3) -> List[Dict[str, Any]]:
    qvec = tfidf_vector(tokenize(query), IDF)
    scored = [(cosine(qvec, v), i) for i, v in enumerate(DOC_VECS)]
    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:k]:
        text = CORPUS[idx]
        title = None
        try:
            after_colon = text.split(":", 1)[1].strip()
            title = after_colon.split(".", 1)[0].strip()
        except Exception:
            title = text[:80]

        d = {
            "id": int(idx),
            "title": title,
            "text": text,
            "score": float(score),
        }
        results.append(d)
    return results

#       Integrate the search method as a tool
def tool_search(query: str, bucketLevel: int = None, subject: str = None, credits: int = None, k: int = 3) -> Dict[str, Any]:
    CORPUS = get_corpus(bucketLevel, subject, credits)

    DOC_TOKENS, VOCAB = doc_tokens_vocab(CORPUS)
    IDF = compute_idf(DOC_TOKENS, VOCAB)
    DOC_VECS = doc_vecs(DOC_TOKENS, IDF)
    
    hits = search_corpus(query, DOC_VECS, CORPUS, IDF, k=k)
    
    # Return a concise, citation-friendly payload
    return {
        "tool": "search",
        "query": query,
        "results": [
            {"id": h["id"], "title": h["title"], "snippet": h["text"]}
            for h in hits
        ],
    }

TOOLS = {
    "search": {
        "schema": {"query": "str", "k": "int? (default=3)"},
        "fn": tool_search
    },
    "finish": {
        "schema": {"answer": "str"},
        "fn": lambda answer: {"tool": "finish", "answer": answer}
    }
}