from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_AKSHARE_LIVE") != "1", reason="set RUN_AKSHARE_LIVE=1 to run live akshare tests")
def test_live_fetch_etf_hist_em_small_range() -> None:
    """
    True functional test hitting akshare/eastmoney.

    This is intentionally opt-in because it depends on network + upstream availability.
    """
    import akshare as ak

    from etf_momentum.data.akshare_fetcher import FetchRequest, fetch_etf_daily_qfq

    rows = fetch_etf_daily_qfq(ak, FetchRequest(code="510300", start_date="20240102", end_date="20240110", adjust="qfq"))
    assert len(rows) > 0
    assert all(r.code == "510300" for r in rows)
    assert all(r.trade_date.isoformat() >= "2024-01-02" for r in rows)
    assert all((r.close is None) or (r.close > 0) for r in rows)

