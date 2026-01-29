from __future__ import annotations

def test_deps_get_engine_and_get_session() -> None:
    from fastapi import FastAPI
    from sqlalchemy import text

    import etf_momentum.api.deps as deps
    from etf_momentum.db.init_db import init_db
    from etf_momentum.db.session import make_session_factory, make_sqlite_engine

    app = FastAPI()
    # Pre-seed app.state so init_app_state does not try to connect to MySQL in tests.
    eng = make_sqlite_engine()
    init_db(eng)
    sf = make_session_factory(eng)
    app.state.engine = eng
    app.state.session_factory = sf
    deps.init_app_state(app)
    assert app.state.engine is eng

    # exercise get_session
    class Req:
        def __init__(self, a):
            self.app = a

    gen1 = deps.get_session(Req(app))
    db1 = next(gen1)
    db1.execute(text("SELECT 1"))
    gen1.close()

    # second init should no-op
    deps.init_app_state(app)


def test_deps_get_akshare_returns_module() -> None:
    import etf_momentum.api.deps as deps

    ak = deps.get_akshare()
    assert hasattr(ak, "fund_etf_hist_em")

