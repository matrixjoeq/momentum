from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(sqlite_path: str | None = None, *, db_url: str | None = None) -> Engine:
    """
    Create SQLAlchemy engine.

    - If db_url is provided: use it as-is (e.g. mysql+pymysql://...).
    - Else fallback to sqlite_path (local file).
    """
    if db_url:
        return create_engine(
            db_url,
            future=True,
            pool_pre_ping=True,
        )
    if not sqlite_path:
        raise ValueError("either db_url or sqlite_path must be provided")
    # sqlite needs check_same_thread=False for multi-threaded web usage
    return create_engine(
        f"sqlite+pysqlite:///{sqlite_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )


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

