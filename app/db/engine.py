import os
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Creating database engine
# Note that SQLAlchemy uses lazy connection so we only know if it's successfull
# if we actually try doing something with the connection created by the engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True  # checks connection health automatically
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
