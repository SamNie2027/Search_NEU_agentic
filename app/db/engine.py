import os
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# Fail early if DATABASE_URL is missing
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set. Did you load .env from the project root?")

# When using sqlite we need special connect args; keep connect_args as a dict
# so we don't accidentally pass `None` into SQLAlchemy which can cause
# a TypeError ('NoneType' object is not iterable') in some versions.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Creating database engine
# Note: SQLAlchemy uses lazy connection; errors can show up when the engine
# establishes a connection later on.
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # checks connection health automatically
)

# Create a session
# The session makes it so that we can commit entries as python objects and it helps to
# handle committing the data to the actual DB through the engine (the bind)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Helper to get session securely
@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
