from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# Sina "global futures" service also serves London spot metals. Map the common
# spot tickers (XAUUSD/XAGUSD) to Sina's internal symbols (XAU/XAG).
SINA_GLOBAL_SYMBOL_ALIASES: dict[str, str] = {
    "XAUUSD": "XAU",
    "XAGUSD": "XAG",
    "XAU": "XAU",
    "XAG": "XAG",
}


def normalize_sina_global_symbol(symbol: str) -> str:
    """Map a user-facing spot ticker to Sina's global-futures symbol (e.g. XAUUSD->XAU)."""
    key = str(symbol or "").strip().upper()
    return SINA_GLOBAL_SYMBOL_ALIASES.get(key, key)


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
            timeout = httpx.Timeout(float(timeout_s), connect=float(timeout_s))
            r = httpx.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://finance.sina.com.cn",
                },
            )
            r.raise_for_status()
            text = (r.content or b"").decode("utf-8", errors="replace")

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
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                # light exponential backoff to avoid hammering Sina
                time.sleep(min(2.0**attempt, 8.0))
                continue
            logger.warning("sina forex day kline fetch failed symbol=%s err=%s", sym, e)
            break

    return pd.DataFrame(), {**meta, "error": str(last_err) if last_err else "failed"}


def _parse_sina_global_kline(text: str) -> list[dict[str, Any]]:
    """
    Parse Sina's GlobalFuturesService.getGlobalFuturesDailyKLine JSONP payload.

    The body looks like:
      /*<script>...*/\nvar _=XAU=([{"date":"2006-06-19","open":"580.350",...}]);
    i.e. a JSON array of OHLC dicts wrapped in a JSONP assignment. We extract the
    array between the first '[' and the last ']' and json-decode it.
    """
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [rec for rec in data if isinstance(rec, dict)]


def fetch_sina_global_futures_day_kline_daily_close(
    req: FetchRequest,
    *,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch daily OHLC history from Sina's global-futures API and return a dataframe
    with columns: date, close.

    Covers London spot metals and global futures, e.g.:
    - XAU (London spot gold, exposed as XAUUSD)
    - XAG (London spot silver, exposed as XAGUSD)
    - GC / SI / HG (COMEX gold / silver / copper)

    Endpoint (JSONP):
      https://stock.finance.sina.com.cn/futures/api/jsonp_v2.php/var%20_=<sym>=/
        GlobalFuturesService.getGlobalFuturesDailyKLine?symbol=<sym>

    The full history is returned by Sina; we filter by [start, end] client-side.
    """
    sym0 = str(req.symbol or "").strip()
    sym = normalize_sina_global_symbol(sym0)
    meta: dict[str, Any] = {"provider": "sina_global", "symbol": sym0, "sina_symbol": sym}
    if not sym:
        return pd.DataFrame(), {**meta, "error": "empty_symbol"}

    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame(), {**meta, "error": "end_before_start"}

    url = (
        "https://stock.finance.sina.com.cn/futures/api/jsonp_v2.php/"
        f"var%20_={sym}=/GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={sym}"
    )
    meta["url"] = url

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            timeout = httpx.Timeout(float(timeout_s), connect=float(timeout_s))
            r = httpx.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://finance.sina.com.cn",
                },
            )
            r.raise_for_status()
            text = (r.content or b"").decode("utf-8", errors="replace")

            records = _parse_sina_global_kline(text)
            if not records:
                return pd.DataFrame(), {**meta, "error": "unexpected_payload"}

            rows: list[dict[str, Any]] = []
            for rec in records:
                d0 = rec.get("date")
                c0 = rec.get("close")
                if not d0:
                    continue
                try:
                    d = dt.date.fromisoformat(str(d0)[:10])
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
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                # light exponential backoff to avoid hammering Sina
                time.sleep(min(2.0**attempt, 8.0))
                continue
            logger.warning(
                "sina global futures kline fetch failed symbol=%s err=%s", sym0, e
            )
            break

    return pd.DataFrame(), {**meta, "error": str(last_err) if last_err else "failed"}
