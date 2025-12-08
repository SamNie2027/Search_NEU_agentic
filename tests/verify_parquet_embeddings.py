"""
Verify parquet embeddings file structure and metadata.

Checks that embeddings file exists, has correct dimensions, and matches metadata.
"""
import json
import sys
from pathlib import Path

import pandas as pd

p = Path("data/embeddings")
pq = p / "course_embeddings.parquet"
meta_file = p / "embedding_meta.json"

if not pq.exists():
    print("Parquet file missing:", pq)
    sys.exit(2)
if not meta_file.exists():
    print("Meta file missing:", meta_file)
    sys.exit(2)

print("Reading Parquet...")
df = pd.read_parquet(pq)
print("Loaded DataFrame:", df.shape)
print("Columns:", list(df.columns))

print("Reading metadata...")
meta = json.loads(meta_file.read_text(encoding="utf8"))
print("Meta:", meta)

# 1) row count vs meta
if df.shape[0] != meta.get("rows", df.shape[0]):
    print("WARNING: row count mismatch: parquet rows =", df.shape[0], "meta rows =", meta.get("rows"))
else:
    print("Row count matches meta:", df.shape[0])

# 2) embedding column exists and dims
if "embedding" not in df.columns:
    print("ERROR: 'embedding' column missing")
    sys.exit(2)

# count null embeddings
nulls = df['embedding'].isna().sum()
print("Null embeddings:", nulls)

# ensure every embedding has correct length
expected_dim = int(meta.get("dim", -1))
if expected_dim <= 0:
    print("Meta missing valid 'dim' field; skipping dim checks")
else:
    lens = df['embedding'].dropna().apply(lambda v: len(v) if v is not None else 0)
    bad = (lens != expected_dim).sum()
    if bad:
        print(f"ERROR: {bad} embeddings have wrong dimension (expected {expected_dim}).")
        # show first few problematic rows
        print(df[lens != expected_dim].head(5))
    else:
        print("All embeddings have expected dim:", expected_dim)

# 3) sample rows
print("Sample rows:")
print(df.head(5).to_dict(orient="records"))

print("OK: basic Parquet verification complete.")