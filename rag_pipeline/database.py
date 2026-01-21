"""Database models and connection for request history."""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smartagent")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class RequestHistory(Base):
    """Model for storing request history."""
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    contract_code = Column(Text, nullable=False)
    contract_length = Column(Integer, nullable=False)  # character count
    token_count = Column(Integer, nullable=False)      # approximate tokens
    processing_time_ms = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)  # success / error
    result = Column(Text, nullable=True)
    error_message = Column(String(500), nullable=True)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

