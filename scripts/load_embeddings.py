import pandas as pd

def load_embeddings():
    return pd.read_parquet("data/course_embeddings.parquet", columns=["text", "embedding"])