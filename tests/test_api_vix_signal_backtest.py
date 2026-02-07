import pytest
from fastapi.testclient import TestClient

from etf_momentum.app import create_app
from etf_momentum.db.init_db import init_db
from etf_momentum.db.seed import ensure_default_policies
from etf_momentum.db.session import make_session_factory, make_sqlite_engine, session_scope


@pytest.fixture()
def client_with_seeded_prices(monkeypatch):
    import etf_momentum.api.routes as routes
    from etf_momentum.db.models import EtfPrice

    engine = make_sqlite_engine()
    init_db(engine)
    sf = make_session_factory(engine)

    with sf() as db:
        ensure_default_policies(db)
        db.commit()

        code = "513100"
        days = [d.date() for d in __import__("pandas").date_range("2024-01-02", periods=120, freq="B")]
        for i, d in enumerate(days):
            px = 100.0 + 0.1 * i
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=px,
                    close=px,
                    high=px,
                    low=px,
                    volume=1000.0,
                    amount=100000.0,
                    source="seed",
                    adjust="hfq",
                )
            )
        db.commit()

    # Stub Cboe fetch to avoid network calls
    def fake_fetch_cboe_daily_close(req, *, timeout_s=15.0, retries=2):
        import pandas as pd

        us_days = pd.date_range("2023-01-02", "2024-12-31", freq="B")
        close = 20.0 + 0.001 * pd.Series(range(len(us_days)), dtype=float)
        return pd.DataFrame({"date": us_days.date, "close": close.to_numpy(dtype=float)})

    monkeypatch.setattr(routes, "fetch_cboe_daily_close", fake_fetch_cboe_daily_close)

    app = create_app()
    app.state.engine = engine
    app.state.session_factory = sf

    def override_get_session():
        yield from session_scope(sf)

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = lambda: None

    return TestClient(app)


def test_api_vix_signal_backtest_returns_extended_payload(client_with_seeded_prices):
    c = client_with_seeded_prices
    resp = c.post(
        "/api/analysis/vix-signal-backtest",
        json={
            "etf_code": "513100",
            "start": "20240102",
            "end": "20240628",
            "adjust": "hfq",
            "index": "VXN",
            "index_align": "cn_next_trading_day",
            "calendar": "XSHG",
            "exec_model": "open_open",
            "lookback_window": 20,
            "threshold_quantile": 0.01,
            "min_abs_ret": 0.0,
            "trade_cost_bps": 0.0,
            "initial_position": "cash",
            "initial_nav": 1.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "period_returns" in data
    assert "distributions" in data
    assert (data.get("meta") or {}).get("exec_model") == "open_open"
    trades = data.get("trades") or []
    assert len(trades) > 0
    assert "etf_ret_exec" in trades[0]

