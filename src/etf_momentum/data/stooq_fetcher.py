from __future__ import annotations

import datetime as dt
import io
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchRequest:
    symbol: str  # e.g. XAUUSD / GC.F
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def fetch_stooq_daily_close(
    req: FetchRequest,
    *,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch Stooq CSV daily history and return a dataframe with columns: date, close.

    Stooq endpoint returns full history; we filter by date client-side.
    """
    sym0 = str(req.symbol or "").strip()
    sym = sym0.lower()
    meta: dict[str, Any] = {"provider": "stooq", "symbol": sym0}
    if not sym:
        return pd.DataFrame(), {**meta, "error": "empty_symbol"}

    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame(), {**meta, "error": "end_before_start"}

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    meta["url"] = url

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            req2 = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
            with urlopen(req2, timeout=float(timeout_s)) as resp:  # noqa: S310
                raw = resp.read()
            text = "" if raw is None else raw.decode("utf-8", errors="replace")
            if "Date" not in text or "Close" not in text:
                return pd.DataFrame(), {**meta, "error": "unexpected_csv"}

            df = pd.read_csv(io.StringIO(text))
            if df is None or df.empty:
                return pd.DataFrame(), {**meta, "error": "empty_csv"}
            if "Date" not in df.columns or "Close" not in df.columns:
                return pd.DataFrame(), {**meta, "error": "missing_columns"}

            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
            out["close"] = pd.to_numeric(df["Close"], errors="coerce")
            out = out.dropna(subset=["date"]).sort_values("date", ascending=True)
            out = out[(out["date"] >= start) & (out["date"] <= end)].copy()
            if out.empty:
                return pd.DataFrame(), {**meta, "error": "empty_in_range"}
            return out[["date", "close"]], meta
        except (HTTPError, URLError, ValueError, TypeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                continue
            logger.warning("stooq fetch failed symbol=%s err=%s", sym0, e)
            break

    return pd.DataFrame(), {**meta, "error": str(last_err) if last_err else "failed"}

