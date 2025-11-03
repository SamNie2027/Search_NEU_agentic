from __future__ import annotations

from typing import List, Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .models import Course, resolve_course_model
from .engine import get_session


def _course_model_from_session(session: Session):
	"""Return the appropriate Course model (declarative or reflected) for this session."""
	# Prefer the declarative model if it maps cleanly; otherwise reflect
	bind = session.get_bind() if hasattr(session, "get_bind") else session.bind
	if bind is None:
		# Fall back to declarative Course if no bind; caller likely misconfigured session
		return Course
	return resolve_course_model(bind)


def get_course_by_code(session: Session, subject: str, number: int) -> Optional[Course]:
	"""Return a single course by subject and number, or None if not found."""
	Model = _course_model_from_session(session) if session else session
	subj = subject.upper().strip()
	return (
		session.query(Model) 
		.filter(Model.subject == subj, Model.number == number)
		.first()
	)


def search_courses_by_title(
	session: Session, term: str, limit: int = 10
) -> List[Course]:
	"""Simple ILIKE title search with a LIMIT. Returns a list of Course rows."""
	Model = _course_model_from_session(session)
	pattern = f"%{term.strip()}%" if term else "%"
	q = (
		session.query(Model)  # type: ignore[arg-type]
		.filter(Model.title.ilike(pattern))
		.order_by(Model.title.asc())
		.limit(int(limit))
	)
	return list(q)


def random_courses(session: Session, n: int = 3) -> List[Course]:
	"""Return N random courses. Uses PostgreSQL random() for ordering."""
	Model = _course_model_from_session(session)
	# Prefer func.random() (portable on Postgres) rather than text('RANDOM()') literal.
	q = session.query(Model).order_by(func.random()).limit(int(n))  # type: ignore[arg-type]
	return list(q)


__all__ = [
	"get_course_by_code",
	"search_courses_by_title",
	"random_courses",
]


# Convenience wrappers that manage the Session lifecycle using get_session()

def get_course_by_code_safe(subject: str, number: int):
	"""
	Wrapper around get_course_by_code that opens/closes the session via get_session().

	Usage:
		course = get_course_by_code_safe("CS", 2500)
	"""
	with get_session() as session:
		return get_course_by_code(session, subject, number)


def search_courses_by_title_safe(term: str, limit: int = 10):
	"""
	Wrapper around search_courses_by_title using get_session() for session management.
	"""
	with get_session() as session:
		return search_courses_by_title(session, term, limit)


def random_courses_safe(n: int = 3):
	"""
	Wrapper around random_courses using get_session() for session management.
	"""
	with get_session() as session:
		return random_courses(session, n)

# all declares the module’s public API for star-imports.
__all__ += [
	"get_course_by_code_safe",
	"search_courses_by_title_safe",
	"random_courses_safe",
]
