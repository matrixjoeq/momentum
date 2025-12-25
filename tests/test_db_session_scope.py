from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from etf_momentum.db.session import session_scope


def test_session_scope_rolls_back_on_error(session_factory: sessionmaker) -> None:
    # create a simple table
    with session_factory() as db:
        db.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)"))
        db.commit()

    gen = session_scope(session_factory)
    db = next(gen)
    db.execute(text("INSERT INTO t (id, v) VALUES (1, 10)"))
    try:
        gen.throw(RuntimeError("boom"))
    except RuntimeError:
        pass

    with session_factory() as db:
        rows = db.execute(text("SELECT COUNT(*) FROM t")).fetchone()
        assert rows[0] == 0

