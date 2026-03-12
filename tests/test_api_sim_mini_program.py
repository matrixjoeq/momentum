import datetime as dt

import pytest
from fastapi.testclient import TestClient

from etf_momentum.app import create_app
from etf_momentum.db.models import EtfPrice
from etf_momentum.db.session import session_scope
from tests.helpers.rotation_case_data import get_json, post_json, request_json


@pytest.fixture()
def sim_client(session_factory, engine):
    """
    A TestClient with a seeded DB and NO external akshare calls.
    We seed enough hfq/none open/close data to run weekly5-open (lookback=20).
    """
    import etf_momentum.api.routes as routes

    sf = session_factory
    with sf() as db:
        codes = ["159915", "511010", "513100", "518880"]
        start = dt.date(2024, 1, 2)
        # 100 business days is enough for lookback + weekly decisions
        days = [d.date() for d in __import__("pandas").date_range(start, periods=120, freq="B")]
        # deterministic trends: 513100 grows fastest
        slopes = {"159915": 0.10, "511010": 0.01, "513100": 0.20, "518880": 0.05}
        base = 100.0
        for code in codes:
            for i, d in enumerate(days):
                px = base + slopes[code] * i
                for adj in ("hfq", "none"):
                    db.add(EtfPrice(code=code, trade_date=d, open=px, close=px, source="seed", adjust=adj))
        db.commit()

    app = create_app()
    # Prevent app lifespan from creating a MySQL engine during tests.
    app.state.engine = engine
    app.state.session_factory = sf

    def override_get_session():
        yield from session_scope(sf)

    # no akshare needed for these tests
    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = lambda: None
    return TestClient(app)


def test_sim_portfolio_init_generate_and_confirm_trade(sim_client):
    c = sim_client

    # 1) create portfolio
    pid = post_json(c, "/api/sim/portfolio", {"name": "默认账户", "initial_cash": 1000000})["id"]

    # 2) init fixed strategy + 5 variants
    out = request_json(c, method="post", path=f"/api/sim/portfolio/{pid}/init-fixed-strategy")
    assert out["portfolio_id"] == pid
    assert len(out["variant_ids"]) == 5
    vid0 = out["variant_ids"][0]

    # 3) generate decisions from backtest periods
    assert post_json(c, "/api/sim/decision/generate", {"portfolio_id": pid, "start": "20240301", "end": "20240430"})[
        "inserted"
    ] > 0

    # 4) list decisions, preview + confirm first one
    decisions = get_json(c, f"/api/sim/variant/{vid0}/decisions?start=20240301&end=20240430")["decisions"]
    assert len(decisions) > 0
    d0 = decisions[0]

    prev = post_json(c, "/api/sim/trade/preview", {"variant_id": vid0, "decision_id": d0["id"]})
    assert prev["decision_id"] == d0["id"]

    assert post_json(c, "/api/sim/trade/confirm", {"variant_id": vid0, "decision_id": d0["id"]})["ok"] is True

    # 5) mark-to-market for a short window and fetch nav
    request_json(c, method="post", path=f"/api/sim/mark-to-market?variant_id={vid0}&start=20240301&end=20240315")

    nav = get_json(c, f"/api/sim/variant/{vid0}/nav?start=20240301&end=20240315")
    assert len(nav["dates"]) > 0
    assert len(nav["dates"]) == len(nav["nav"])

