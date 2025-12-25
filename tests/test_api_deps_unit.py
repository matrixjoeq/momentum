from __future__ import annotations

def test_deps_get_engine_and_get_session(tmp_path, monkeypatch) -> None:
    # isolate sqlite to tmp path
    db_path = tmp_path / "deps.sqlite3"
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(db_path))

    from fastapi import FastAPI
    from sqlalchemy import text

    import etf_momentum.api.deps as deps

    app = FastAPI()
    deps.init_app_state(app)
    assert str(db_path) in str(app.state.engine.url)

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

