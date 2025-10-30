# Project Roadmap — SearchNEU Agentic Chat (Oct–Nov 2025)

This roadmap breaks the project into 4 weekly sprints with concrete tickets, owners, steps, architecture/library recommendations, acceptance criteria, and risks. It assumes minimal ML experience and prioritizes simple, reliable tools.

Repo: `Search_NEU_agentic` (branch: `DB-connection`)

## Minimal Month Plan — Terminal MVP (fastest path)

This is the least-effort plan for a usable, terminal-only MVP. No servers, no web UI, no LLMs, no vector DB. It delivers a CLI that searches Northeastern courses using simple rules and TF-IDF.

Minimal stack
- Python 3.10+
- DB: PostgreSQL local; driver: `psycopg2-binary`; config via `.env` using `python-dotenv`
- Search: scikit-learn `TfidfVectorizer` + cosine similarity on course title+description
- Filters: rule-based parsing for subject (CS, POLS, …), level buckets (1000/2000/4000/grad), and basic requirement keywords
- Utilities: `pandas` (optional for convenience), `joblib` for caching index, `pytest` for 3–5 smoke tests

Minimal folders and commands
- `cli/` — small Python package with an entrypoint `neu.py`
- `data/` — cached TF-IDF artifacts (joblib) to avoid rebuilding every run
- `scripts/` — optional one-offs (e.g., rebuild index)
- `tests/` — tiny tests for DB connect, search, and filter parsing

CLI usage examples
- `python -m cli.neu search "upper-level computer science elective" --limit 10`
- `python -m cli.neu search "american politics" --subject POLS --level 1000`
- `python -m cli.neu reindex` (rebuild TF-IDF cache)
- `python -m cli.neu db-check` (sanity query to Postgres)

### Sprint 1 (Oct 31 – Nov 6): Local DB + First Query
Goal by 11/6: local Postgres restored; CLI can run a basic title search using SQL ILIKE; env configured; code paths aligned to reuse `knowledge_base.tool_search`.

- T0 — Bootstrap environment and layout (Owner: All)
  - Add `requirements.txt` (minimal): psycopg2-binary, python-dotenv, scikit-learn, joblib, pandas (optional), pytest
  - Add `.env.example` with `DATABASE_URL=postgresql+psycopg2://neu:<pwd>@localhost:5432/searchneu`
  - Add `.gitignore` entries for `data/*.joblib`, `.env`
  - Create `cli/__init__.py` so `python -m cli.neu` works cleanly

- T1 — Restore DB locally and add `.env` (Owner: All, Sam leads)
  - Copy `.env.example` → `.env` and fill credentials
  - Verify `courses` row count using a simple SQL (to be used by `db-check`): `SELECT COUNT(*) FROM courses;`
  - Document columns to use (subject, number, title, description) in `Resources/schema-notes.md`

- T2 — Minimal CLI scaffold wired to DB (Owner: Sam)
  - Create `cli/neu.py` with subcommands and function stubs:
 - [ ]  -  `db-check` → calls `search_backend.select_count_courses()`
 - [ ]  -  `reindex` → placeholder (no-op in Sprint 1; implemented in Sprint 2)
 - [ ]  -  `search <term>` → uses SQL ILIKE on title via `search_backend.select_courses_ilike(term, limit)`
  - Create `cli/search_backend.py` with thin psycopg2 connector using `.env`:
 - [ ]  -  `get_connection()` → returns a new connection with autocommit, sets statement timeout if desired
 - [ ]  -  `select_count_courses()` → returns integer
 - [ ]  -  `select_courses_ilike(term: str, limit: int = 10)` → returns list of dicts: `{subject, number, title, description}`
 - [ ]  -  `tool_search(query: str, k: int = 10)` → for now, wrap `select_courses_ilike` and return shape matching `knowledge_base.py`:
      `{ "tool": "search", "query": query, "results": [{"id": <code>, "title": title, "snippet": description[:240]+...}] }`
  - Ensure `cli/neu.py` pretty-prints results and exits non-zero with a helpful message if DB is unreachable

- T3 — Schema note + sample queries (Owner: All)
  - Add short `Resources/schema-notes.md` with 3–5 sample SQL snippets the CLI will rely on, e.g.:
 - [ ]  -  Titles ILIKE: `SELECT subject, number, title, description FROM courses WHERE title ILIKE '%' || %s || '%' LIMIT %s;`
 - [ ]  -  Subject filter: `... WHERE subject = %s`
    Level bucket (example): `... WHERE number BETWEEN 4000 AND 4999`

Acceptance:
- `python -m cli.neu db-check` prints a positive row count
- `python -m cli.neu search "data"` prints subject/number/title lines
- `tool_search` output shape matches `knowledge_base.py` for future agent compatibility

### Sprint 2 (Nov 7 – Nov 13): TF-IDF Search + Rule Filters
Goal by 11/13: CLI uses TF-IDF search over title+description; subject/level filters parsed and applied.

- T4 — Build TF-IDF index in backend (Owner: Angela)
  - In `cli/search_backend.py`, load courses from Postgres; use scikit-learn `TfidfVectorizer` on `title + " " + description`
  - Persist vectorizer and matrix with `joblib` to `data/tfidf.joblib`
  - Expose `tool_search(query: str, k: int=10)` returning the same structure as in `knowledge_base.py`
- T5 — CLI `search` calls `tool_search` (Owner: Sam)
  - Update `cli/neu.py search` to call `search_backend.tool_search`
  - Display subject, number, title, and a short description snippet
- T6 — Rule-based filters (Owner: Aria)
  - Add minimal parsing in `cli/search_backend.py` (or a tiny `cli/filters.py`) for `--subject` (dept code) and `--level` (1000/2000/4000/grad from course number)
  - Apply filters pre-rank (SQL WHERE) or post-rank (Python slice); prefer simplest working path

Acceptance: Queries like "machine learning" return sensible CS courses; `--subject POLS` narrows correctly; level buckets work.

### Sprint 3 (Nov 14 – Nov 20): Datatypes + Query Builder (Rules Only)
Goal by 11/20: minimal datatypes implemented; single place to compose filters + search.

- T7 — Datatypes module (Owner: Angela)
  - Create `cli/datatypes.py` with Subject codes and Level buckets; add a small keyword list for 1–2 requirements if desired
  - Keep names simple to reuse later if `agent_system.py` is integrated
- T8 — Query builder composition (Owner: Sam)
  - In `cli/search_backend.py`, centralize the flow: fetch -> optional rule filters -> TF-IDF rank -> top-k
  - Add support for `--limit` in `cli/neu.py`
- T9 — Tiny tests (Owner: All)
  - Add `tests/test_cli_smoke.py`: DB connects; filter parsing in `cli/datatypes.py`; TF-IDF returns ≥1 result for a known query

Acceptance: End-to-end CLI answers typical questions with filters, consistently.

### Sprint 4 (Nov 21 – Nov 27): Polish + Demo
Goal by 11/27: polished CLI demo; quick reindex; light docs.

- T10 — CLI UX polish (Owner: Aria)
  - Improve `cli/neu.py` help text (`-h`), align columns, truncate snippets; optional `colorama` for color (skip if time)
- T11 — Reindex + cache guardrails (Owner: Sam)
  - In `cli/search_backend.py`, detect missing/stale `data/tfidf.joblib`; print guidance to run `python -m cli.neu reindex` instead of crashing
- T12 — Demo scripts + README quickstart (Owner: All)
  - Add a short "Try it" section in `README.md` using the CLI commands
  - Include 3 demo queries in `Resources/demo-queries.md`

Acceptance: Team can run 3 demo queries live, zero code edits needed.

Milestone mapping
- 11/6: DB connected (T1) + CLI first query (T2)
- 11/13: TF-IDF search + subject/level filters (T4–T6)
- 11/20: Datatypes + query builder + tests (T7–T9)
- 11/27: Polished CLI and demo-ready (T10–T12)

De-scoped from MVP (optional later)
- Any web server (FastAPI), web UI (Streamlit), embeddings, FAISS/pgvector, LLM integration

---

## Repository reuse plan for the terminal MVP

Leverage as much of the existing starter code as possible while keeping effort low:

- knowledge_base.py
  - Reuse: overall TF–IDF search structure and function shapes (`search_corpus`, `tool_search`).
  - Minimal tweak: point the “corpus” to DB-backed course rows instead of the toy list. Internally switch implementation to scikit-learn `TfidfVectorizer` but keep `tool_search(query, k)` interface so later agent code can call it unchanged.
  - Optional: keep `tokenize` for simple normalization if needed.

- prompting_techniques.py
  - Reuse later (optional). For the MVP CLI, we won’t run ReAct. Keep as-is so we can integrate an agent later without rework.
  - If you want to align now, implement the TODO in `parse_action` (low effort) so the agent path is ready when/if needed.

- agent_system.py
  - Defer for MVP. It stays a clean extension path (ReAct loop) that can call the same `tool_search` shim you’ll expose from the CLI backend.

Language_model.py
  - Defer for MVP (LLM not required). Leave TODOs; optional swap-in later.

Thin shim to maximize reuse
- Add a tiny module (later) `cli/search_backend.py` that:
  Loads courses from Postgres (columns: subject, number, title, description)
  - Builds/loads a cached TF-IDF model (`data/tfidf.joblib`)
  - Exposes `tool_search(query: str, k: int=10) -> {"tool":"search", "query":..., "results":[...]}` matching the shape used in `knowledge_base.py`
  - This lets both the CLI and the future agent share the same “tool.”

Quick changes checklist
- Keep existing files unchanged where possible; add the CLI and backend as new files.
- Reuse names/signatures (`tool_search`) to keep a no-changes migration path to the agent later.
- Only modify `knowledge_base.py` if you prefer to literally host the DB-backed TF-IDF there; otherwise wrap it via `cli/search_backend.py` and import into future agent code.

---

## Expanded plan (optional) — Web/API and embeddings

- Runtime: Python 3.10+
- Service/API: FastAPI + Uvicorn
- Config: python-dotenv for `.env` (DB URL, LLM key, feature flags)
- Data layer: PostgreSQL (local) restored from AWS dump
  - ORM: SQLAlchemy 2.0 + `psycopg2-binary`
  - Optional migrations: Alembic (not required if DB is restored read-only)
- Embeddings: `sentence-transformers` (model: `all-MiniLM-L6-v2`) — light, fast, good quality
- Classifiers: scikit-learn (LogisticRegression/LinearSVC) + `train/test` scripts
- Semantic search index (choose one; start with simpler):
  - Option A (simple, no DB extension): FAISS CPU index built from embeddings
  - Option B (if extension enabled): `pgvector` in Postgres; store and query vectors in-db
- Orchestration/Agent: Start with a light custom “tool-calling” planner (no heavy framework). If needed, optionally add LangChain later for routing.
LLM (optional but helpful for answer formatting): OpenAI API (model: `gpt-4o-mini`) using environment key; keep the app usable without LLM by returning structured results.
- UI: Streamlit (fast to build) with a minimal chat and course-card result list
- Testing: pytest
- Quality: black + isort (optional), pre-commit (optional)

Folder suggestions (create as you go):
- `app/` (FastAPI app, routers, service layer)
- `app/db/` (SQLAlchemy engine/session/models)
- `app/ml/` (encoders, classifiers, indexing)
- `scripts/` (one-off scripts: restore, embed, build index)
- `ui/` (Streamlit app)
- `tests/` (pytest)

---

## Sprint 1 (Oct 31 – Nov 6): Local DB + Baseline Encoder

Goal by 11/6: code can query Postgres; baseline encoder exists to start extracting subject/level/requirement signals.

### S1-T1 — Install and restore local PostgreSQL
Owner: All (Sam leads)
Outcome: Local Postgres running with SearchNEU data restored; credentials in `.env`.
Steps:
 - [ ] Install PostgreSQL (Windows installer); ensure `psql` and `pg_restore` are on PATH.
 - [ ] Create role and database:
 - [ ]  -  Role: `neu` with password, CREATEDB off, LOGIN on.
 - [ ]  -  DB: `searchneu` (UTF8, owner `neu`).
 - [ ] Place AWS dump locally (e.g., `C:\data\searchneu.dump`) and restore:
 - [ ]  -  Custom format: `pg_restore -d searchneu -U neu -h localhost -p 5432 C:\data\searchneu.dump`
 - [ ]  -  Plain SQL: `psql -d searchneu -U neu -h localhost -p 5432 -f C:\data\searchneu.sql`
 - [ ] Validate schema and counts with `psql`:
 - [ ]  -  `\dt` shows tablesRisks;`SELECT COUNT(*) FROM courses;` returns > 0.
 - [ ]  -  Spot-check columns: subject, number, title, description, prerequisites, meeting info.
 - [ ] Create `.env` at repo root with `DATABASE_URL=postgresql+psycopg2://neu:<pwd>@localhost:5432/searchneu`.
 - [ ] Note any required extensions (e.g., `uuid-ossp`) and enable if dump references them.
Libraries/Resources: PostgreSQL 14+, official docs; Windows service manager.
Acceptance criteria: `SELECT COUNT(*)` on courses returns rows; schema documented in `Resources/` notes.
Risks: Dump compatibility; Mitigation: normalize with `pg_dump` format; if blocked, use CSV export to bootstrap.

### S1-T2 — Minimal DB access layer (SQLAlchemy)
Owner: Sam
Outcome: `app/db/engine.py` (engine/session), `app/db/models.py` (declarative models for core tables), `app/db/queries.py` (simple getters).
Steps:
 - [ ] `app/db/engine.py`:
 - [ ]  -  Read `DATABASE_URL` from `.env` (python-dotenv) and create `create_engine()` with pool_pre_ping=True.
 - [ ]  -  Provide `SessionLocal()` factory and a `get_session()` context helper.
 - [ ] `app/db/models.py`:
 - [ ]  -  Define Base = declarative_base().
 - [ ]  -  Create core models with essential fields only: `Course(id/code, subject, number, title, description, credits, attributes)`; add `__tablename__` and indices (subject+number).
 - [ ]  -  If schema mismatch unknown, use SQLAlchemy reflection for `Course` as fallback.
 - [ ] `app/db/queries.py`:
 - [ ]  -  `get_course_by_code(session, subject: str, number: int) -> Course | None`.
 - [ ]  -  `search_courses_by_title(session, term: str, limit: int = 10) -> list[Course]` using `ilike`.
 - [ ]  -  Optional: `random_courses(session, n=3)` for later health/sample.
 - [ ] `tests/test_db_smoke.py`:
 - [ ]  -  Test engine connects; test title search returns ≥ 1 row for a common term (e.g., "data").
 - [ ]  -  Test get by code for a known code if available; otherwise skip with marker.
Libraries: SQLAlchemy 2.0, psycopg2-binary, pytest, python-dotenv.
Acceptance criteria: Pytest can run 2-3 test queries successfully.

### S1-T3 — Baseline sentence encoder + embedding script
Owner: All
Outcome: Script computes embeddings for each course row (e.g., title + description) and saves artifacts.
Steps:
 - [ ] Choose model `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly) and install requirement.
 - [ ] Define text recipe: `f"{subject} {number}: {title}. {description}"`; strip HTML; cap description length (e.g., 1k chars).
 - [ ] `scripts/compute_embeddings.py`:
 - [ ]  -  Connect to DB (reuse `app/db/engine.py` or lightweight psycopg2).
 - [ ]  -  Query courses in batches (e.g., LIMIT/OFFSET 1000) to control memory.
 - [ ]  -  Encode with batch_size=64; store rows as DataFrame with columns: course_id/code, subject, number, title, desc, embedding (np.ndarray or list[float]).
 - [ ]  -  Persist to `data/course_embeddings.parquet` (arrow/pyarrow) and `data/embedding_meta.json` (model name, date, dim).
 - [ ] Indexing option:
 - [ ]  -  Option A (FAISS): build an IndexFlatIP on normalized embeddings; save to `data/faiss.index`.
 - [ ]  -  Option B (pgvector): create table with `vector` column; insert embeddings; create ivfflat index if enabled.
 - [ ] Add `--resume` flag to skip already processed IDs.
Acceptance criteria: Embeddings computed for >95% of courses; index built; script runtime < 20 min on laptop.
Risks: Large corpuses; Mitigation: batch compute (512–1k rows per batch) and persist progress.

### S1-T4 — Minimal FastAPI skeleton + health/db endpoints
Owner: Sam
Outcome: FastAPI app with `/healthz` and `/courses/sample` (returns 3 random courses).
Steps:
 - [ ] `app/main.py`:
 - [ ]  -  Instantiate FastAPI; add startup event to test DB connection; inject settings from `.env`.
 - [ ]  -  Implement `/healthz` returning `{status:"ok", db:true/false}` based on a trivial query.
 - [ ] `app/routers/courses.py`:
 - [ ]  -  Route `/courses/sample` using `random_courses(n=3)`; serialize minimal fields (subject, number, title).
 - [ ]  -  Add `/courses/search?term=...&limit=10` that calls `search_courses_by_title`.
 - [ ] Pydantic response models in `app/schemas.py` for CourseOut.
 - [ ] Local run: `uvicorn app.main:app --reload` and verify both endpoints.
Acceptance criteria: Endpoint returns valid JSON; cleanly logs DB errors if misconfigured.

### S1-T5 — Draft “datatype” taxonomy and dataset notes
Owner: All (Angela leads taxonomy)
Outcome: A short doc `Resources/datatypes.md` describing:
  - Subject (CS, POLS, …)
  Level buckets (1000, 2000, 4000, grad)
  - Major requirement labels (e.g., CCIS requirements; list initial targets)
Steps: Inspect DB columns, catalog site, and SearchNEU schema; list label sources or rules of thumb.
Acceptance criteria: Team agrees on initial labels and sources; doc committed.

---

## Sprint 2 (Nov 7 – Nov 13): Working Encoder (TF‑IDF) + Rule Filters

Goal by 11/13: a working “encoder” powering retrieval via TF‑IDF in the CLI, plus subject/level rule filters; reuse `knowledge_base.tool_search` shape.

### S2-T1 — Build TF‑IDF index in `cli/search_backend.py`
Owner: Angela
Outcome: Persisted TF‑IDF artifacts at `data/tfidf.joblib`.
Steps:
 - [ ] Load courses from Postgres (subject, number, title, description) using psycopg2 and `.env`.
 - [ ] Fit scikit-learn `TfidfVectorizer` on `title + " " + description`; store vectorizer and matrix with `joblib`.
 - [ ] Implement `tool_search(query: str, k: int=10)` that returns `{tool, query, results:[...]}` matching `knowledge_base.py`.
Acceptance: `python -m cli.neu reindex` completes and creates `data/tfidf.joblib`.

### S2-T2 — Wire CLI search to backend `tool_search`
Owner: Sam
Outcome: `python -m cli.neu search "..."` returns ranked results with subject/number/title/snippet.
Steps:
 - [ ] Update `cli/neu.py search` to import and call `search_backend.tool_search`.
 - [ ] Pretty-print top-k with truncated description.
Acceptance: Query "machine learning" yields sensible CS courses.

### S2-T3 — Rule-based filters for subject and level
Owner: Aria
Outcome: `--subject` (dept code) and `--level` (1000/2000/4000/grad) flags supported.
Steps:
 - [ ] Add parsing helpers in `cli/datatypes.py` (Subject enum, Level buckets) and import into `cli/search_backend.py`.
 - [ ] Apply filters either pre-rank (SQL WHERE) or post-rank (simple Python filter) — pick simplest working path.
Acceptance: `--subject POLS` narrows correctly; level buckets produce expected subsets.

### S2-T4 — Future-agent alignment (optional, low effort)
Owner: Aria
Outcome: `prompting_techniques.parse_action` TODO implemented so `agent_system.py` can reuse `tool_search` later.
Steps: Implement parse logic per the helpers in `prompting_techniques.py`; no runtime use in MVP.
Acceptance: Unit check that `Action: search[query="test", k=3]` parses to ("search", {query:"test", k:3}).

### S2-T5 — Dev quickstart
Owner: All
Outcome: `README` quickstart section for CLI.
Steps: List commands to run `db-check`, `reindex`, and `search`; note `.env` setup.
Acceptance: New dev runs a search in <15 minutes.

---

## Sprint 3 (Nov 14 – Nov 20): Datatypes + Query Builder (CLI)

Goal by 11/20: datatypes implemented; a single place in `cli/search_backend.py` composes filters + TF‑IDF rank.

### S3-T1 — Datatypes module
Owner: Angela
Outcome: `cli/datatypes.py` with Subject codes and Level buckets; optional minimal requirement keywords.
Steps: Define enums/constants and simple helpers; import into backend.
Acceptance: Parsing functions return expected buckets and codes.

### S3-T2 — Query builder composition in backend
Owner: Sam
Outcome: Centralized flow in `cli/search_backend.py`: fetch -> apply optional rule filters -> TF‑IDF rank -> top‑k.
Steps: Add `--limit` support (via `cli/neu.py`) and ensure deterministic output ordering.
Acceptance: Synthetic combinations (subject+level) return correct subsets.

### S3-T3 — Tiny tests
Owner: All
Outcome: `tests/test_cli_smoke.py` with 3–5 tests.
Steps: DB connects; datatypes parse; TF‑IDF returns ≥1 result for a known query.
Acceptance: Tests pass locally.

---

## Sprint 4 (Nov 21 – Nov 27): CLI Polish + Demo

Goal by 11/27: polished CLI demo; cache/reindex guardrails; docs.

### S4-T1 — CLI UX polish
Owner: Aria
Outcome: Better `cli/neu.py` help (`-h`), aligned columns, truncated snippets; optional color (skip if time).
Acceptance: Output is readable and consistent.

### S4-T2 — Reindex + cache guardrails
Owner: Sam
Outcome: `cli/search_backend.py` detects missing/stale `data/tfidf.joblib` and instructs `python -m cli.neu reindex` instead of crashing.
Acceptance: Running without cache yields helpful guidance and proceeds after reindex.

### S4-T3 — Demo scripts + README quickstart
Owner: All
Outcome: `Resources/demo-queries.md` with 3 demo flows; README "Try it" section for CLI.
Acceptance: Team can run a scripted demo end-to-end.

---

## Milestone alignment to your dates
- 10/30 Presentation (today): use current slides; demo target and plan using this roadmap.
- 11/6: DB connected (S1-T1–T2) + baseline encoder and embeddings (S1-T3). Health endpoint running (S1-T4).
- 11/13: Working encoder + subject classifier + search API (S2-T1–T3); basic prompts (S2-T4).
- 11/20: Datatypes done (Sam/Angela/Aria tickets in Sprint 3) and query builder integrated (S3-T4), Streamlit scaffold (S3-T5).
- 11/27: Integrated agent + usable app + demo polish (Sprint 4).
- 12/1: Final presentation and tidy repo.

---

## Risks and mitigations
- pgvector not available on Windows: Use FAISS index on disk (default path). Keep a small index refresh script.
- DB restore issues: Use CSV exports or limited table subset; create lightweight derived tables via scripts.
- Sparse labels for requirements: Start rule-based with keyword lists; enhance gradually.
- Model performance: Keep it simple; prefer explainable rules where signals are strong (course number → level).
- Team time constraints: Keep each ticket shippable; avoid deep frameworks until needed.

---

## Quick dependency list to pin (for later `requirements.txt`)
- fastapi, uvicorn
- sqlalchemy, psycopg2-binary, python-dotenv
- sentence-transformers, scikit-learn, numpy, pandas, joblib
- faiss-cpu (or `pgvector` if using in-db vectors)
- streamlit, pytest
- openai (optional)

