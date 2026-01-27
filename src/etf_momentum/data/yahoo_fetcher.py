from __future__ import annotations

import datetime as dt
import io
import logging
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import pandas as pd
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchRequest:
    symbol: str  # e.g. "^VIX", "^GVX"
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _date_from_epoch_seconds(x: int | float | None) -> dt.date | None:
    if x is None:
        return None
    try:
        return dt.datetime.utcfromtimestamp(int(x)).date()
    except (OSError, OverflowError, TypeError, ValueError):  # pragma: no cover
        return None


def _extract_chart_series(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Parse Yahoo chart API response into a dataframe with columns: date, close.
    """
    chart = payload.get("chart") if isinstance(payload, dict) else None
    result = (chart.get("result") if isinstance(chart, dict) else None) or []
    if not result or not isinstance(result, list):
        return pd.DataFrame()
    r0 = result[0] if isinstance(result[0], dict) else None
    if not r0:
        return pd.DataFrame()

    ts = r0.get("timestamp") or []
    ind = r0.get("indicators") or {}
    quote = (ind.get("quote") if isinstance(ind, dict) else None) or []
    q0 = quote[0] if quote and isinstance(quote[0], dict) else None
    if not q0:
        return pd.DataFrame()

    close = q0.get("close") or []
    if not ts or not close:
        return pd.DataFrame()

    n = min(len(ts), len(close))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        d = _date_from_epoch_seconds(ts[i])
        if d is None:
            continue
        v = close[i]
        try:
            fv = float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            fv = np.nan
        rows.append({"date": d, "close": fv})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=True)
    return df


def fetch_yahoo_daily_close(
    req: FetchRequest,
    *,
    timeout_s: float = 10.0,
    retries: int = 2,
) -> pd.DataFrame:
    """
    Fetch Yahoo Finance daily close series via the public chart API:
    - endpoint: https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
    - params: interval=1d, period1/period2 (epoch seconds)
    """
    sym = str(req.symbol or "").strip()
    if not sym:
        return pd.DataFrame()

    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame()

    # Yahoo uses exclusive end timestamp; add 1 day so the end date is included.
    period1 = int(dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc).timestamp())
    period2 = int(
        dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc).timestamp()
        + 24 * 3600
    )

    # Yahoo expects special symbols to be URL-encoded in the path, e.g. "^VIX" -> "%5EVIX".
    sym_path = quote(sym, safe="")
    params = {"interval": "1d", "period1": str(period1), "period2": str(period2)}

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://finance.yahoo.com/",
            }
            with httpx.Client(timeout=timeout_s, headers=headers, follow_redirects=True) as client:
                # Try chart API (two hosts).
                for host in ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"):
                    url = f"{host}/v8/finance/chart/{sym_path}"
                    resp = client.get(url, params=params)
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as se:
                        # 403/429: fall back to CSV download flow (cookie+crumb).
                        code = int(se.response.status_code)
                        if code in {403, 429}:
                            df_csv = _fetch_yahoo_download_csv(
                                client,
                                symbol_path=sym_path,
                                period1=period1,
                                period2=period2,
                            )
                            if not df_csv.empty:
                                df_csv = df_csv[(df_csv["date"] >= start) & (df_csv["date"] <= end)].copy()
                                return df_csv
                        # Try next host or retry loop.
                        continue

                    payload = resp.json()
                    df = _extract_chart_series(payload)
                    if not df.empty:
                        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
                        return df
                    # Empty is treated as empty result (no retries needed).
                    return pd.DataFrame()

                # If both hosts failed, last fallback attempt: CSV flow once.
                df_csv = _fetch_yahoo_download_csv(client, symbol_path=sym_path, period1=period1, period2=period2)
                if not df_csv.empty:
                    df_csv = df_csv[(df_csv["date"] >= start) & (df_csv["date"] <= end)].copy()
                    return df_csv
                return pd.DataFrame()
        except (httpx.HTTPError, ValueError, TypeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                continue
            logger.warning("yahoo fetch failed symbol=%s err=%s", sym, e)
            break

    if last_err is not None:
        return pd.DataFrame()
    return pd.DataFrame()


def _fetch_yahoo_download_csv(
    client: httpx.Client,
    *,
    symbol_path: str,
    period1: int,
    period2: int,
) -> pd.DataFrame:
    """
    Best-effort Yahoo CSV download flow:
    1) Visit finance.yahoo.com quote page to set cookies
    2) Get crumb from query1.finance.yahoo.com/v1/test/getcrumb
    3) Download CSV from /v7/finance/download/{symbol}
    """
    try:
        # Step 1: set cookies
        client.get(f"https://finance.yahoo.com/quote/{symbol_path}/history")
        # Step 2: crumb (plain text)
        crumb_resp = client.get("https://query1.finance.yahoo.com/v1/test/getcrumb")
        crumb = (crumb_resp.text or "").strip() if crumb_resp.status_code == 200 else ""
        if not crumb:
            return pd.DataFrame()

        # Step 3: download CSV
        dl_url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol_path}"
        dl_params = {
            "period1": str(period1),
            "period2": str(period2),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
            "crumb": crumb,
        }
        r = client.get(dl_url, params=dl_params)
        r.raise_for_status()
        text = r.text or ""
        if "Date" not in text or "Close" not in text:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text))
        if df is None or df.empty:
            return pd.DataFrame()
        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        out["close"] = pd.to_numeric(df["Close"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date", ascending=True)
        return out[["date", "close"]]
    except (httpx.HTTPError, ValueError, TypeError):
        return pd.DataFrame()


def fetch_yahoo_daily_close_with_alias(
    req: FetchRequest,
    *,
    aliases: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch Yahoo close series with best-effort symbol aliases (e.g. GVX -> GVZ).
    Returns (df, meta).
    """
    sym0 = str(req.symbol or "").strip()
    cand = [sym0]
    if aliases:
        key = sym0.upper()
        key2 = key[1:] if key.startswith("^") else key
        for k in (key, key2):
            if k in aliases:
                cand.extend(aliases[k])
                break
    seen: set[str] = set()
    tried: list[str] = []
    for s in cand:
        ss = str(s).strip()
        if not ss or ss in seen:
            continue
        seen.add(ss)
        tried.append(ss)
        df = fetch_yahoo_daily_close(FetchRequest(symbol=ss, start_date=req.start_date, end_date=req.end_date))
        if not df.empty:
            return df, {"symbol_used": ss, "symbols_tried": tried}
    return pd.DataFrame(), {"symbol_used": None, "symbols_tried": tried}

