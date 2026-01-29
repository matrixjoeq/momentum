from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from etf_momentum.db.init_db import init_db
from etf_momentum.db.seed import ensure_default_policies
from etf_momentum.db.session import make_session_factory, make_sqlite_engine


@pytest.fixture()
def engine():
    eng = make_sqlite_engine()
    init_db(eng)
    return eng


@pytest.fixture()
def session_factory(engine):
    return make_session_factory(engine)


@pytest.fixture()
def api_client(engine):
    """
    API client isolated from local filesystem DB:
    - uses sqlite in-memory (tests only)
    - overrides FastAPI dependencies for get_session/get_akshare
    """
    from etf_momentum.app import create_app
    from etf_momentum.db.session import session_scope
    import etf_momentum.api.routes as routes

    sf = make_session_factory(engine)
    with sf() as db:
        ensure_default_policies(db)
        db.commit()

    class FakeAk:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            return pd.DataFrame(
                {
                    "日期": ["2024-01-02", "2024-01-03"],
                    "开盘": [1.0, 1.02],
                    "最高": [1.05, 1.06],
                    "最低": [0.98, 1.00],
                    "收盘": [1.02, 1.03],
                    "成交量": [10.0, 20.0],
                    "成交额": [100.0, 200.0],
                }
            )

    app = create_app()
    # Prevent app lifespan from creating a MySQL engine during tests.
    app.state.engine = engine
    app.state.session_factory = sf

    def override_get_session():
        yield from session_scope(sf)

    def override_get_akshare():
        return FakeAk()

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare

    return TestClient(app)

