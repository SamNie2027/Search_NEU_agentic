from __future__ import annotations

from typing import List, Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .models import Course, resolve_course_model
from .engine import get_session
from types import SimpleNamespace


def _detach_row(obj):
	"""Convert a SQLAlchemy ORM instance to a simple namespace (detached, safe to use after session close).

	Copies all column values found on the mapped table to attributes on a SimpleNamespace.
	"""
	if obj is None:
		return None
	# If obj is already a mapping/namespace, return as-is
	# (helps if callers pass through simple dicts)
	try:
		table = obj.__table__
		values = {c.name: getattr(obj, c.name) for c in table.columns}
		return SimpleNamespace(**values)
	except Exception:
		# Fallback: try vars()
		try:
			return SimpleNamespace(**vars(obj))
		except Exception:
			return obj


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
	Model = _course_model_from_session(session)
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

def return_text_stream(session: Session, bucketLevel: int = None, subject: str = None, credits: int = None, n: int = 1000, offset: int = 0):
	Model = _course_model_from_session(session)
	# Start a base query and apply filters incrementally so we don't need
	# separate branches for every combination of filters.
	q = session.query(Model)

	if bucketLevel is not None:
		start = int(bucketLevel)
		end = start + 1000
		q = q.filter((Model.number >= start) & (Model.number < end))

	if subject is not None:
		q = q.filter(Model.subject == subject.upper())

	if credits is not None:
		# Assume `credits` is a numeric column on the model
		q = q.filter(Model.min_credits >= int(credits))

	# Apply a stable ordering and pagination for chunked reads
	q = q.order_by(Model.subject.asc(), Model.number.asc()).offset(int(offset)).limit(int(n)).yield_per(100)

	for row in q:
		yield format_course_recipe(row)


__all__ = [
	"get_course_by_code",
	"search_courses_by_title",
	"random_courses",
	"return_text_stream"
]


# Convenience wrappers that manage the Session lifecycle using get_session()

def get_course_by_code_safe(subject: str, number: int):
	"""
	Wrapper around get_course_by_code that opens/closes the session via get_session().

	Usage:
		course = get_course_by_code_safe("CS", 2500)
	"""
	with get_session() as session:
		row = get_course_by_code(session, subject, number)
		return _detach_row(row)


def search_courses_by_title_safe(term: str, limit: int = 10):
	"""
	Wrapper around search_courses_by_title using get_session() for session management.
	"""
	with get_session() as session:
		rows = search_courses_by_title(session, term, limit)
		return [(_detach_row(r) if r is not None else None) for r in rows]


def random_courses_safe(n: int = 3):
	"""
	Wrapper around random_courses using get_session() for session management.
	"""
	with get_session() as session:
		rows = random_courses(session, n)
		return [(_detach_row(r) if r is not None else None) for r in rows]

# all declares the module’s public API for star-imports.
__all__ += [
	"get_course_by_code_safe",
	"search_courses_by_title_safe",
	"random_courses_safe",
]


def format_course_recipe(course) -> str:
	"""Return a normalized text recipe for a course.

	Format: "{subject} {number}: {title}. {description}". Missing parts are skipped cleanly.
	Accepts either a SimpleNamespace returned by the safe wrappers or an ORM instance.
	"""
	if course is None:
		return ""
	# Try attribute access safely
	subj = getattr(course, "subject", None)
	num = getattr(course, "number", None)
	title = getattr(course, "title", None)
	desc = getattr(course, "description", None)

	parts = []
	if subj:
		parts.append(str(subj).strip())
	if num is not None:
		parts.append(str(num).strip())

	head = " ".join(parts)
	if head:
		head = head + ":"

	body = ""
	if title:
		body += str(title).strip()
	if desc:
		if body:
			body += ". " + str(desc).strip()
		else:
			body = str(desc).strip()

	# Compose final recipe and normalize whitespace
	recipe = f"{head} {body}".strip()
	# Collapse multiple spaces
	return " ".join(recipe.split())


__all__ += [
	"format_course_recipe",
]
