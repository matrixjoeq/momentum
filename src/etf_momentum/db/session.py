from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def make_engine(*, db_url: str | None) -> Engine:
    """
    Create SQLAlchemy engine.

    Runtime uses MySQL only.
    """
    if not db_url:
        raise ValueError("db_url is required (runtime uses MySQL only)")
    return create_engine(
        db_url,
        future=True,
        pool_pre_ping=True,
    )


def make_sqlite_engine(sqlite_url: str = "sqlite+pysqlite:///:memory:") -> Engine:
    """
    Test-only helper: create a SQLite engine (in-memory by default).
    """
    if not sqlite_url.startswith("sqlite"):
        return create_engine(sqlite_url, future=True)
    connect_args = {"check_same_thread": False}
    # For in-memory SQLite, ensure all connections share the same DB.
    if sqlite_url.endswith(":memory:"):
        return create_engine(sqlite_url, future=True, connect_args=connect_args, poolclass=StaticPool)
    return create_engine(sqlite_url, future=True, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def session_scope(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    db = session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

