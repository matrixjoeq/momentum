import pandas as pd

from etf_momentum.data.akshare_sina_fetcher import FetchRequest, fetch_etf_daily_sina_none_and_adjusted


class _AkMock:
    def __init__(self, hist_df: pd.DataFrame, div_df: pd.DataFrame):
        self._hist = hist_df
        self._div = div_df

    def fund_etf_hist_sina(self, *args, **kwargs):
        return self._hist.copy()

    def fund_etf_dividend_sina(self, *args, **kwargs):
        return self._div.copy()


def test_sina_derives_hfq_and_qfq_from_none_and_dividend_factor():
    # 3 days, with cumulative dividend stepping on day2
    hist = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": [100, 100, 100],
        }
    )
    div = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "累计分红": [0.0, 0.2]})
    ak = _AkMock(hist, div)

    packs = fetch_etf_daily_sina_none_and_adjusted(ak, FetchRequest(code="159915", start_date="20240102", end_date="20240104"))
    none_rows = packs["none"]
    hfq_rows = packs["hfq"]
    qfq_rows = packs["qfq"]

    assert len(none_rows) == 3
    assert len(hfq_rows) == 3
    assert len(qfq_rows) == 3

    # hfq should be >= raw (dividend added)
    assert hfq_rows[0].close == none_rows[0].close
    assert hfq_rows[1].close > none_rows[1].close

    # qfq is scaled so last close matches raw last close
    assert abs(qfq_rows[-1].close - none_rows[-1].close) < 1e-12

