# """
# Engine/session setup. DATABASE_URL is read from .env via python-dotenv
# (config.py already calls load_dotenv() on import, so importing config here
# is enough to guarantee it's been loaded — no need to call it again).

# Example .env line:
#     DATABASE_URL=postgresql+psycopg2://kyc_user:kyc_pass@localhost:5432/kyc_db
# """
# import os

# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# from src2 import config  # noqa: F401  (import triggers load_dotenv())
# from src2.db.models import Base

# DATABASE_URL = os.environ.get(
#     "DATABASE_URL",
#     "postgresql+psycopg2://postgres:postgres@localhost:5432/kyc_db",
# )

# engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# # `expire_on_commit=False`: the background task commits, then the request
# # handler (or a later poll) may still want to read attributes off the same
# # object without triggering a fresh query against a session that's already closed.
# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


# def init_db():
#     """Create tables if they don't exist yet. Call once at app startup."""
#     Base.metadata.create_all(bind=engine)


# def get_db():
#     """FastAPI dependency: yields a session, always closes it."""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src2 import config  # noqa: F401  (import triggers load_dotenv())
from src2.db.models import Base


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./kyc_db.sqlite3",
)


if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db():
    """Create tables if they don't exist yet. Call once at app startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session, always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()