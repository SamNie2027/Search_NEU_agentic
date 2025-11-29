import sys
from pathlib import Path

# Ensure repository root is on sys.path so `import app` works when running this script
# file is at app/db/embedding_similarity.py so parents[2] is the repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
     
from sentence_transformers import SentenceTransformer
from app.db.load_embeddings import load_embeddings
import pandas as pd
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def embedding_search(query: str, courses: list, embeddings: list, k: int = 3):
    """Minimal-change cosine similarity search using the original loop style.

    This keeps the original approach but fixes shape/matmul issues by
    ensuring `query_embedding` and each `e` are 1-D NumPy arrays and by
    correcting the sorting call.
    """
    # Compute query embedding and ensure it's a 1-D vector
    query_embedding = model.encode([query])
    if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 2:
        query_embedding = query_embedding[0]

    scores = []
    for i, e in enumerate(embeddings):
        # coerce element to 1-D numpy array
        e_arr = np.asarray(e)

        # compute dot product
        dot_prod = float(np.dot(query_embedding, e_arr))

        # compute norms
        query_norm = float(np.linalg.norm(query_embedding, ord=2))
        e_norm = float(np.linalg.norm(e_arr, ord=2))
        if query_norm == 0.0 or e_norm == 0.0:
            score = 0.0
        else:
            score = dot_prod / (query_norm * e_norm)

        scores.append((score, i))

    # sort by score descending
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

courses, embeddings = load_embeddings()
print(embedding_search("...whatever query you want!...", courses, embeddings))