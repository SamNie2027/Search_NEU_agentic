"""
TF-IDF search implementation for course data.

Provides keyword-based search using term frequency-inverse document frequency
with optional filtering by course level, subject, credits, and major requirements.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field, asdict
import string
from typing import Callable, Dict, List, Tuple, Optional, Any
import json
import math
import re
import textwrap
import random
import os
from collections import Counter, defaultdict
from app.database.engine import get_session
from app.database.queries import return_text_stream, get_courses_by_codes_safe, format_course_recipe
import numpy as np

try:
    from major_requirements.major_requirements import get_requirement_codes
    REQUIREMENTS_AVAILABLE = True
except ImportError:
    REQUIREMENTS_AVAILABLE = False


def get_corpus(bucketLevel: int = None, subject: str = None, credits: int = None, major_requirement: str = None):
    """
    Get corpus of course text recipes, optionally filtered by major requirements.
    
    If major_requirement is specified, only courses matching requirement codes are included.
    Otherwise, uses standard filters (bucketLevel, subject, credits).
    """
    # If major requirement is specified, filter by requirement codes first
    if major_requirement and REQUIREMENTS_AVAILABLE:
        try:
            course_codes = get_requirement_codes(major_requirement)
            
            if not course_codes:
                return []
            
            courses = get_courses_by_codes_safe(course_codes)
            
            if not courses:
                return []
            
            # Apply additional filters if specified (subject, credits, bucketLevel)
            filtered_courses = []
            for course in courses:
                if subject is not None:
                    course_subj = getattr(course, 'subject', None)
                    if course_subj is None or str(course_subj).upper() != subject.upper():
                        continue
                
                if credits is not None:
                    course_credits = getattr(course, 'min_credits', None)
                    if course_credits is None or course_credits < credits:
                        continue
                
                if bucketLevel is not None:
                    course_num = getattr(course, 'number', None)
                    if course_num is None:
                        continue
                    start = int(bucketLevel)
                    end = start + 1000
                    if not (start <= course_num < end):
                        continue
                
                recipe = format_course_recipe(course)
                if recipe:
                    filtered_courses.append(recipe)
            
            return filtered_courses
            
        except Exception as e:
            pass
    
    # Standard filtering path (no major requirement or fallback)
    with get_session() as session:
        # Collect recipe strings in chunks. `return_text_stream` yields strings;
        # convert each chunk to a list and extend the master list so `sentences`
        # is a flat list of strings (SentenceTransformer expects an indexable list).
        CORPUS = []

        for x in range(0, 6001, 1000):
            chunk = list(return_text_stream(session=session, bucketLevel=bucketLevel, credits=credits, subject=subject, offset=x, n=1000))
            chunk = [s for s in chunk if s]
            CORPUS.extend(chunk)
        return CORPUS

def tokenize(text: str) -> List[str]:
    """Tokenize text into words, extracting alphanumeric sequences."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

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
    """Extract tokens from corpus and build vocabulary."""
    DOC_TOKENS = [[t for t in tokenize(doc) if t not in STOPWORDS] for doc in CORPUS]
    VOCAB = sorted(set(t for doc in DOC_TOKENS for t in doc))
    return DOC_TOKENS, VOCAB


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for tokens in a document."""
    result = {}
    for x in tokens:
        result[x] = result.get(x, 0) + 1
    return result


def compute_df(doc_tokens: List[List[str]]) -> Dict[str, float]:
    """Compute document frequency: count of documents containing each token."""
    df = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] += 1
    return df


def compute_idf(DOC_TOKENS, VOCAB):
    """Compute inverse document frequency using smoothed variant."""
    DF = compute_df(DOC_TOKENS)
    N_DOC = len(DOC_TOKENS)
    IDF = {t: math.log((N_DOC + 1) / (DF[t] + 0.5)) + 1 for t in VOCAB}
    return IDF


def tfidf_vector(tokens: List[str], IDF) -> Dict[str, float]:
    """Compute TF-IDF vector for a document's tokens."""
    tf = compute_tf(tokens)
    vec = {t: tf[t] * IDF.get(t, 0.0) for t in tf}
    return vec

def doc_vecs(DOC_TOKENS, IDF):
    return [tfidf_vector(tokens, IDF) for tokens in DOC_TOKENS]

def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    if not a or not b:
        return 0.0

    common_words = filter(lambda x: x in b, a.keys())
    dot_prod = sum(a[word] * b[word] for word in common_words)
    
    a_norm = np.linalg.norm(np.array(list(a.values())), ord=2)
    b_norm = np.linalg.norm(np.array(list(b.values())), ord=2)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    similarity = dot_prod / (a_norm * b_norm)
    return similarity


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


def tool_search(query: str, bucketLevel: int = None, subject: str = None, credits: int = None, major_requirement: str = None, k: int = 3) -> Dict[str, Any]:
    """
    Search for courses using TF-IDF with optional filters.
    
    Args:
        query: Search query string
        bucketLevel: Course level bucket (1000, 2000, 4000, etc.)
        subject: Department code filter (e.g., "CS")
        credits: Minimum credits filter
        major_requirement: Major name to filter by requirement courses (e.g., "CS")
        k: Number of results to return
        
    Returns:
        Dictionary with search results in tool format
    """
    CORPUS = get_corpus(bucketLevel, subject, credits, major_requirement)
    
    if not CORPUS:
        return {
            "tool": "search",
            "query": query,
            "results": [],
        }

    DOC_TOKENS, VOCAB = doc_tokens_vocab(CORPUS)
    IDF = compute_idf(DOC_TOKENS, VOCAB)
    DOC_VECS = doc_vecs(DOC_TOKENS, IDF)
    
    hits = search_corpus(query, DOC_VECS, CORPUS, IDF, k=k)
    
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
        "schema": {"query": "str", "k": "int? (default=3)", "major_requirement": "str? (optional, e.g., 'CS')"},
        "fn": tool_search
    },
    "finish": {
        "schema": {"answer": "str"},
        "fn": lambda answer: {"tool": "finish", "answer": answer}
    }
}
