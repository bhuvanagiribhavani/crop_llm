"""
================================================================================
DATABASE CONNECTION MODULE
================================================================================
SQLAlchemy engine, session factory, and base model for PostgreSQL + PostGIS.

Uses environment variables from .env file for credentials.
================================================================================
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()

# Build database URL from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "crop_db")
DB_USER = os.getenv("DB_USER", "crop_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "crop_user")

# URL-encode the password to handle special characters like @
DATABASE_URL = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db():
    """
    Dependency that provides a database session.
    Yields a session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables defined by Base subclasses.
    Call this on FastAPI startup.
    """
    from models import CropPrediction  # noqa: F401 — import so table is registered
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created / verified")
