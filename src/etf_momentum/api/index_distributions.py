from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..data.cboe_fetcher import FetchRequest as CboeFetchRequest
from ..data.cboe_fetcher import fetch_cboe_daily_close


Window = Literal["1y", "3y", "5y", "10y", "all"]


@dataclass(frozen=True)
class IndexDistributionInputs:
    symbol: str  # VXN/GVZ/VIX
    window: Window = "all"
    end_date: dt.date | None = None
    bins: int = 60
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _hist(samples: np.ndarray, *, bins: int) -> dict[str, Any]:
    x = samples[np.isfinite(samples)]
    if x.size == 0:
        return {"bins": [], "counts": [], "n": 0}
    b = int(max(10, min(200, bins)))
    counts, edges = np.histogram(x, bins=b)
    return {"bins": edges.astype(float).tolist(), "counts": counts.astype(int).tolist(), "n": int(x.size)}


def _qs(samples: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    x = samples[np.isfinite(samples)]
    if x.size == 0:
        return {f"q{int(q * 100)}": float("nan") for q in qs}
    vals = np.quantile(x, qs)
    return {f"q{int(q * 100)}": float(v) for q, v in zip(qs, vals, strict=False)}


def compute_cboe_index_distribution(inp: IndexDistributionInputs) -> dict[str, Any]:
    sym = str(inp.symbol or "").strip().upper()
    if sym.startswith("^"):
        sym = sym[1:]
    if sym not in {"VIX", "GVZ", "VXN"}:
        return {"ok": False, "error": "unsupported_symbol", "meta": {"symbol": sym}}

    end = inp.end_date or dt.date.today()
    # fetch a wide range, then slice by window
    df = fetch_cboe_daily_close(CboeFetchRequest(index=sym, start_date="19900101", end_date=end.strftime("%Y%m%d")))
    if df is None or df.empty:
        return {"ok": False, "error": "empty_series", "meta": {"symbol": sym}}

    s = pd.Series(df["close"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]).dt.date.to_list(), dtype=float).dropna()
    if s.empty:
        return {"ok": False, "error": "empty_close", "meta": {"symbol": sym}}

    if inp.window != "all":
        years = int(str(inp.window).replace("y", ""))
        start = end - dt.timedelta(days=365 * years + 10)
        s = s[s.index >= start]

    if s.empty or len(s) < 20:
        return {"ok": False, "error": "insufficient_window", "meta": {"symbol": sym, "window": inp.window}}

    close = s.to_numpy(dtype=float)
    close_dates = list(s.index)
    close_current = float(close[-1])
    close_current_date = max(s.index).isoformat()

    # log returns aligned to the "to" date
    ret_s = np.log(pd.Series(close, index=pd.to_datetime(close_dates))).diff()
    ret_s.index = pd.to_datetime(close_dates)
    ret_s = pd.to_numeric(ret_s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret_s.to_numpy(dtype=float)
    ret_current = float(ret_s.iloc[-1]) if not ret_s.empty else float("nan")
    ret_current_date = ret_s.index[-1].date().isoformat() if not ret_s.empty else None

    out = {
        "ok": True,
        "meta": {
            "symbol": sym,
            "window": inp.window,
            "start": min(s.index).isoformat(),
            "end": max(s.index).isoformat(),
            "n_close": int(len(close)),
            "n_ret": int(len(ret)),
        },
        "series": {
            "dates": [d.isoformat() for d in close_dates],
            "close": close.astype(float).tolist(),
        },
        "close": {
            "hist": _hist(close, bins=int(inp.bins)),
            "quantiles": _qs(close, inp.quantiles),
            "current": close_current,
            "current_date": close_current_date,
        },
        "ret_log": {
            "hist": _hist(ret, bins=int(inp.bins)),
            "quantiles": _qs(ret, inp.quantiles),
            "current": ret_current,
            "current_date": ret_current_date,
        },
    }
    return out

