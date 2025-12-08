"""
Semantic search using sentence embeddings.

Provides cosine similarity search over course embeddings using the
sentence-transformers model for query encoding.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer
from app.search.load_embeddings import load_embeddings
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def embedding_search(query: str, courses: list, embeddings: list, k: int = 3):
    """
    Perform cosine similarity search over course embeddings.
    
    Args:
        query: Search query string
        courses: List of course text descriptions
        embeddings: List of embedding vectors corresponding to courses
        k: Number of top results to return
        
    Returns:
        List of dictionaries with id, title, text, and score for top k results
    """
    query_embedding = model.encode([query])
    if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 2:
        query_embedding = query_embedding[0]

    scores = []
    for i, e in enumerate(embeddings):
        e_arr = np.asarray(e)
        dot_prod = float(np.dot(query_embedding, e_arr))
        query_norm = float(np.linalg.norm(query_embedding, ord=2))
        e_norm = float(np.linalg.norm(e_arr, ord=2))
        if query_norm == 0.0 or e_norm == 0.0:
            score = 0.0
        else:
            score = dot_prod / (query_norm * e_norm)

        scores.append((score, i))

    scores.sort(reverse=True)

    results = []
    for score, i in scores[:k]:
        text = courses[i]
        title = None
        try:
            after_colon = text.split(":", 1)[1].strip()
            title = after_colon.split(".", 1)[0].strip()
        except Exception:
            title = text[:80]

        d = {
            "id": int(i),
            "title": title,
            "text": text,
            "score": float(score),
        }
        results.append(d)

    return results