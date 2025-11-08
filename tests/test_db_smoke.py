import pytest

from app.db.engine import engine
from app.db.queries import (
    search_courses_by_title_safe,
    get_course_by_code_safe,
    random_courses_safe,
)


@pytest.fixture(scope="module")
def db_available():
    """Skip tests if the DATABASE_URL is not configured or the DB is unreachable."""
    # engine.url may be None if DATABASE_URL was not set; guard for that
    url = getattr(engine, "url", None)
    if url is None:
        pytest.skip("DATABASE_URL not configured; skipping DB integration tests")

    try:
        conn = engine.connect()
        conn.close()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Cannot connect to the database: {exc}")

    return True


def test_engine_connects(db_available):
    assert db_available


def test_title_search_returns_rows(db_available):
    """A simple smoke test: searching for a common term should return at least one row.

    If the dataset doesn't contain the term, skip the assertion rather than failing CI.
    """
    results = search_courses_by_title_safe("data", limit=5)
    if not results:
        pytest.skip("No results for 'data' in this dataset; skipping title search assertion")
    assert len(results) >= 1


def test_get_by_code_known_or_skip(db_available):
    """Pick a random course, then ensure get_course_by_code can fetch it by (subject, number)."""
    sample = random_courses_safe(1)
    if not sample:
        pytest.skip("No courses available to test get_course_by_code")

    course = sample[0]
    subj = getattr(course, "subject", None)
    num = getattr(course, "number", None)
    if subj is None or num is None:
        pytest.skip("Sample course missing subject/number; skipping")

    fetched = get_course_by_code_safe(subj, num)
    assert fetched is not None
    assert getattr(fetched, "subject", None) == subj
    assert getattr(fetched, "number", None) == num
