"""
Load course embeddings from parquet files.

Provides functions to load pre-computed embeddings and course text data
for semantic search operations.
"""
import pandas as pd
from pathlib import Path
import numpy as np


def load_embeddings():
    """
    Load embeddings from parquet file, searching repository if needed.
    
    Returns:
        Tuple of (courses, embeddings) where courses is a list of course text
        and embeddings is a list of embedding vectors.
    """
    filename = "course_embeddings.parquet"
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent.parent
    matches = list(repo_root.rglob(filename))
    if matches:
        parquet_path = matches[0]
        df = pd.read_parquet(parquet_path, columns=["text", "embedding"])
        courses = df["text"].tolist()
        embeddings = df["embedding"].tolist()

        meta_path = parquet_path.with_name("embedding_meta.json")
        meta = None
        if meta_path.exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        data_dim = len(embeddings[0]) if embeddings else 0
        if meta and "dim" in meta:
            try:
                meta_dim = int(meta["dim"])
            except Exception:
                meta_dim = None
            if meta_dim is not None and meta_dim != data_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: metadata dim={meta_dim} but parquet data dim={data_dim}"
                )

        return courses, embeddings