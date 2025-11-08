"""Pytest conftest to ensure the project root is on sys.path during test collection.

This makes `import app` resolvable when running tests from the repository root.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
# Prepend repo root so tests can import `app` as a package
sys.path.insert(0, str(ROOT))
