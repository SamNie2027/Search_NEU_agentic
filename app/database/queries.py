"""
Database query functions for course data.

Provides functions to query courses by various criteria and format course data
for use in search and display.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func, text, or_
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
		try:
			return SimpleNamespace(**vars(obj))
		except Exception:
			return obj


def _course_model_from_session(session: Session):
	"""Return the appropriate Course model (declarative or reflected) for this session."""
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
	q = session.query(Model).order_by(func.random()).limit(int(n))  # type: ignore[arg-type]
	return list(q)

def get_courses_by_codes(session: Session, course_codes: List[Tuple[str, int]]) -> List[Course]:
	"""
	Return all courses matching the provided list of course codes.
	
	Args:
		session: SQLAlchemy session
		course_codes: List of (subject, number) tuples to match
		
	Returns:
		List of Course objects matching any of the provided codes
	"""
	if not course_codes:
		return []
	
	Model = _course_model_from_session(session)
	
	# Build OR conditions for each (subject, number) pair
	conditions = []
	for subject, number in course_codes:
		subj_upper = str(subject).upper().strip()
		try:
			num_int = int(number)
			conditions.append(
				(Model.subject == subj_upper) & (Model.number == num_int)  # type: ignore
			)
		except (ValueError, TypeError):
			# Skip invalid course codes
			continue
	
	if not conditions:
		return []
	
	# Combine all conditions with OR
	filter_condition = or_(*conditions)
	
	q = session.query(Model).filter(filter_condition)  # type: ignore[arg-type]
	return list(q)


def return_text_stream(session: Session, bucketLevel: int = None, subject: str = None, credits: int = None, n: int = 1000, offset: int = 0):
	"""Yield formatted course text recipes with optional filters."""
	Model = _course_model_from_session(session)
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

	q = q.order_by(Model.subject.asc(), Model.number.asc()).offset(int(offset)).limit(int(n)).yield_per(100)

	for row in q:
		yield format_course_recipe(row)


__all__ = [
	"get_course_by_code",
	"search_courses_by_title",
	"random_courses",
	"get_courses_by_codes",
	"return_text_stream"
]


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


def get_courses_by_codes_safe(course_codes: List[Tuple[str, int]]):
	"""
	Wrapper around get_courses_by_codes using get_session() for session management.
	
	Usage:
		codes = [("CS", 2500), ("CS", 3000)]
		courses = get_courses_by_codes_safe(codes)
	"""
	with get_session() as session:
		rows = get_courses_by_codes(session, course_codes)
		return [(_detach_row(r) if r is not None else None) for r in rows]

__all__ += [
	"get_course_by_code_safe",
	"search_courses_by_title_safe",
	"random_courses_safe",
	"get_courses_by_codes_safe",
]


def format_course_recipe(course) -> str:
	"""Return a normalized text recipe for a course.

	Format: "{subject} {number}: {title}. {description}". Missing parts are skipped cleanly.
	Accepts either a SimpleNamespace returned by the safe wrappers or an ORM instance.
	"""
	if course is None:
		return ""
	
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

	recipe = f"{head} {body}".strip()
	return " ".join(recipe.split())


__all__ += [
	"format_course_recipe",
]
