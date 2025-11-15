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


# obtain sentences from the queries module (assumes return_text_stream is iterable)

with get_session() as session:
    # Collect recipe strings in chunks. `return_text_stream` yields strings;
    # convert each chunk to a list and extend the master list so `sentences`
    # is a flat list of strings (SentenceTransformer expects an indexable list).
    sentences = []
    for x in range(0, 6001, 1000):
        chunk = list(return_text_stream(session, offset=x, n=1000))
        # filter out any empty strings
        chunk = [s for s in chunk if s]
        sentences.extend(chunk)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Encode in batches to avoid excessive memory use; show progress if available.
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
dim = embeddings.shape[1] if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1 else (len(embeddings[0]) if embeddings else 0)

# Persist embeddings and metadata
out_dir = Path(ROOT) / "data"
out_dir.mkdir(parents=True, exist_ok=True)

# Build DataFrame: text + embedding (as list)
df = pd.DataFrame({
    "text": sentences,
    "embedding": [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings],
})

parquet_path = out_dir / "course_embeddings.parquet"
df.to_parquet(parquet_path, index=False)

meta = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "date": datetime.utcnow().isoformat() + "Z",
    "dim": int(dim),
    "rows": len(df),
}

with open(out_dir / "embedding_meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)

print(f"Wrote {len(df)} embeddings to {parquet_path} (dim={dim})")
