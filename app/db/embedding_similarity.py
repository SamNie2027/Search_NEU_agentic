import sys
from pathlib import Path

# Ensure repository root is on sys.path so `import app` works when running this script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))
     
from sentence_transformers import SentenceTransformer
from app.db.queries import return_text_stream
from app.db.engine import get_session
import pandas as pd
import json
from datetime import datetime

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def embedding_similarity(query, courses, embeddings):
    query_embedding = model.encode([query])
