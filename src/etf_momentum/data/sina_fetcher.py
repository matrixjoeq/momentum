from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchRequest:
    symbol: str  # e.g. DINIW
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def fetch_sina_forex_day_kline_daily_close(
    req: FetchRequest,
    *,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch daily OHLC history from Sina's forex API and return a dataframe with columns: date, close.

    Works for symbols like:
    - fx_susdcny
    - DINIW (USD index as used by Sina/FX pages)

    Endpoint format (jsonp-like):
      https://vip.stock.finance.sina.com.cn/forex/api/jsonp.php/data=/NewForexService.getDayKLine?symbol=<symbol>

    Response looks like:
      data=("1985-11-08,129.2200,128.9100,129.6600,129.1300,|1985-11-11,...")
    where each record is: date, open, high, low, close, (trailing comma), separated by '|'.
    """
    sym = str(req.symbol or "").strip()
    meta: dict[str, Any] = {"provider": "sina", "symbol": sym}
    if not sym:
        return pd.DataFrame(), {**meta, "error": "empty_symbol"}

    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame(), {**meta, "error": "end_before_start"}

    url = f"https://vip.stock.finance.sina.com.cn/forex/api/jsonp.php/data=/NewForexService.getDayKLine?symbol={sym}"
    meta["url"] = url

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            req2 = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"})  # noqa: S310
            with urlopen(req2, timeout=float(timeout_s)) as resp:  # noqa: S310
                text = (resp.read() or b"").decode("utf-8", errors="replace")

            m = re.search(r'\(\s*"(?P<body>.*)"\s*\)\s*;?\s*$', text, flags=re.DOTALL)
            if not m:
                # Some responses start with "/*<script>...*/" comments; still should end with (...).
                m = re.search(r'\(\s*"(?P<body>.*)"\s*\)', text, flags=re.DOTALL)
            if not m:
                return pd.DataFrame(), {**meta, "error": "unexpected_payload"}

            body = m.group("body")
            if not body:
                return pd.DataFrame(), {**meta, "error": "empty_payload"}

            rows: list[dict[str, Any]] = []
            for rec in body.split("|"):
                rec = rec.strip()
                if not rec:
                    continue
                parts = [p.strip() for p in rec.split(",")]
                if len(parts) < 5:
                    continue
                d0, _, _, _, c0 = parts[0], parts[1], parts[2], parts[3], parts[4]
                try:
                    d = dt.date.fromisoformat(d0[:10])
                except ValueError:
                    continue
                try:
                    c = float(c0)
                except (TypeError, ValueError):
                    c = float("nan")
                rows.append({"date": d, "close": c})

            df = pd.DataFrame(rows)
            if df.empty:
                return pd.DataFrame(), {**meta, "error": "empty_parsed"}
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date", ascending=True)
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
            if df.empty:
                return pd.DataFrame(), {**meta, "error": "empty_in_range"}
            return df[["date", "close"]], meta
        except (HTTPError, URLError, ValueError, TypeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                continue
            logger.warning("sina forex day kline fetch failed symbol=%s err=%s", sym, e)
            break

    return pd.DataFrame(), {**meta, "error": str(last_err) if last_err else "failed"}

