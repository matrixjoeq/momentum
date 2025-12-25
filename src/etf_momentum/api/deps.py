from __future__ import annotations

from collections.abc import Generator

import akshare as ak
from fastapi import FastAPI, Request
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..db.init_db import init_db
from ..db.session import make_engine, make_session_factory, session_scope
from ..db.seed import ensure_default_policies
from ..settings import get_settings


def build_engine() -> Engine:
    settings = get_settings()
    engine = make_engine(str(settings.sqlite_path))
    init_db(engine)
    return engine


def init_app_state(app: FastAPI) -> None:
    if getattr(app.state, "engine", None) is not None and getattr(app.state, "session_factory", None) is not None:
        return

    engine = build_engine()
    session_factory = make_session_factory(engine)
    db = session_factory()
    try:
        ensure_default_policies(db)
        db.commit()
    finally:
        db.close()

    app.state.engine = engine
    app.state.session_factory = session_factory


def get_session(request: Request) -> Generator[Session, None, None]:
    init_app_state(request.app)
    sf: sessionmaker[Session] = request.app.state.session_factory
    yield from session_scope(sf)


def get_akshare():
    return ak

