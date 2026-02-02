from __future__ import annotations

import math

import numpy as np
import pandas as pd

from etf_momentum.analysis.vol_proxy import VolProxySpec, compute_vol_proxy_levels
from etf_momentum.analysis.vol_timing import backtest_tiered_exposure_by_level


def _make_synth_ohlc(n: int = 400, seed: int = 7) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    # Random walk on log-price
    r = rng.normal(0.0002, 0.015, size=n)
    logp = np.cumsum(r) + math.log(1.0)
    close = np.exp(logp)

    # Construct OHLC around close with random ranges (ensure positivity)
    open_ = close * np.exp(rng.normal(0.0, 0.002, size=n))
    intraday = np.abs(rng.normal(0.0, 0.01, size=n))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * np.maximum(0.0001, (1.0 - intraday))

    s = lambda x: pd.Series(x, index=dates).astype(float)
    return {"open": s(open_), "high": s(high), "low": s(low), "close": s(close)}


def test_vol_proxy_levels_shapes_and_finiteness() -> None:
    ohlc = _make_synth_ohlc()
    kinds = [
        "rv_close",
        "ewma_close",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "yang_zhang",
        "har_rv",
    ]
    for k in kinds:
        spec = VolProxySpec(kind=k, window=20, ann=252)
        lvl = compute_vol_proxy_levels(ohlc, spec=spec)
        assert isinstance(lvl, pd.Series)
        assert len(lvl) == len(ohlc["close"])
        # should have some finite values after warmup
        assert int(np.isfinite(lvl.to_numpy(dtype=float)).sum()) > 50


def test_tier_backtest_runs_on_vol_proxy() -> None:
    ohlc = _make_synth_ohlc()
    lvl = compute_vol_proxy_levels(ohlc, spec=VolProxySpec(kind="rv_close", window=20, ann=252))

    close = ohlc["close"]
    etf_ret = np.log(close).diff()
    df = pd.DataFrame({"etf_ret": etf_ret}).dropna()
    df2 = df.join(pd.DataFrame({"idx_close": lvl}), how="inner").dropna()

    levels = pd.to_numeric(df2["idx_close"], errors="coerce")
    lv = levels.to_numpy(dtype=float)
    lv = lv[np.isfinite(lv)]
    qs = [0.8, 0.9]
    thr_abs = [float(np.quantile(lv, q)) for q in qs]
    exposures = [1.0, 0.5, 0.2]

    out = backtest_tiered_exposure_by_level(df2, levels, thresholds_abs=thr_abs, exposures=exposures, cost_bps=10, ann=252)
    assert out["ok"] is True
    assert len(out["dates"]) == len(out["nav_strategy"]) == len(out["nav_buy_hold"]) == len(out["exp"])
    ms = out["metrics"]["strategy"]
    assert "cagr" in ms and "vol" in ms and "sharpe" in ms and "max_drawdown" in ms

