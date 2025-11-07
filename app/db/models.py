# Note: This file is very subject to change based on the actual schema of the DB and what shape of data we care to implement
# Therefore there is some dependency on issue #13 
from __future__ import annotations

from typing import Optional, Type, List

from sqlalchemy import Index, Integer, String, Text, inspect
from sqlalchemy.orm import Mapped, declarative_base, mapped_column
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.dialects.postgresql import ARRAY

# Declarative base for ORM models
# AKA a base for the table
Base = declarative_base()


class Course(Base):
	"""
	Declarative Course model with essential fields only.

	If the live database schema differs, use `resolve_course_model(engine)` to obtain
	a reflected/automapped class mapped to the existing `courses` table instead.
	"""

	__tablename__ = "courses"
	__table_args__ = (
		# Composite index to speed up subject/number lookups
		# Accelerates the common lookup pattern "find the course by subject and number"
		Index("ix_courses_subject_number", "subject", "number"),
	)

	# Matches raw_data/seed.py (Postgres) shape
	id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
	# Department code (TEXT)
	subject: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
	# Course number as an integer (e.g., 4100). The DB was changed to store numeric course numbers.
	number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
	# Title/description stored as TEXT
	title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
	description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
	# Credit range (INTEGER)
	min_credits: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
	max_credits: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
	# Instructors stored as TEXT[] (array of names)
	instructors: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text), nullable=True)

	def __repr__(self) -> str:  # pragma: no cover - debug helper
		return (
			f"Course(subject={self.subject!r}, number={self.number!r}, "
			f"title={self.title!r})"
		)


def resolve_course_model(engine) -> Type[Course]:
	"""
	Returns a mapped class for the `courses` table.

	- If the live DB has the expected columns, returns the declarative `Course` class.
	- If the schema is unknown/mismatched, returns an automapped class via reflection.

	Usage:
		from app.db.models import resolve_course_model
		CourseModel = resolve_course_model(engine)
		session.query(CourseModel).limit(1).all()
	"""

	inspector = inspect(engine)

	# If the table exists and required columns are present, use the declarative Course
	required = {"subject", "number", "title", "description"}
	try:
		if inspector.has_table("courses"):
			try:
				cols = {c["name"] for c in inspector.get_columns("courses")}
			except Exception:
				cols = set()
			if required.issubset(cols):
				return Course
	except Exception:
		# If inspection fails, fall through to automap
		pass

	# Fallback: automap the existing schema at runtime
	AutoBase = automap_base()
	AutoBase.prepare(autoload_with=engine)

	# Typical case: table name matches attribute on classes namespace
	try:
		return getattr(AutoBase.classes, "courses")  # type: ignore[attr-defined]
	except Exception:
		# As a last-ditch attempt, bind the reflected Table to a dynamic class
		try:
			table = AutoBase.metadata.tables["courses"]

			class CourseAuto(AutoBase):  # type: ignore[misc]
				__table__ = table  # type: ignore[assignment]

			return CourseAuto  # type: ignore[return-value]
		except Exception:
			# If we still can't reflect, return the declarative class to avoid import-time crashes.
			return Course

# all declares the module’s public API for star-imports.
__all__ = [
	"Base",
	"Course",
	"resolve_course_model",
]

