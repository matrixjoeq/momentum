from __future__ import annotations

import datetime as dt
import io
import logging
from dataclasses import dataclass

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


_CBOE_CSV = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "GVZ": "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv",
    "VXN": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv",
}


@dataclass(frozen=True)
class FetchRequest:
    index: str  # VIX/GVZ/VXN
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _parse_cboe_history_csv(text: str) -> pd.DataFrame:
    """
    Parse Cboe daily history CSV into dataframe with columns: date, close.

    Expected columns include:
    - DATE
    - CLOSE (or other "*Close*" column)
    """
    if not text or "DATE" not in text.upper():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    if df is None or df.empty:
        return pd.DataFrame()

    cols = {str(c): c for c in df.columns}
    date_col = None
    for c in cols:
        if str(c).strip().lower() == "date":
            date_col = cols[c]
            break
    if date_col is None:
        # fallback: first column
        date_col = df.columns[0]

    close_col = None
    for c in df.columns:
        if "close" == str(c).strip().lower():
            close_col = c
            break
    if close_col is None:
        for c in df.columns:
            if "close" in str(c).strip().lower():
                close_col = c
                break
    if close_col is None:
        # Some Cboe files (e.g. GVZ_History.csv) contain columns like: DATE,GVZ
        # In that case, use the first non-date column as the value column.
        non_date_cols = [c for c in df.columns if str(c) != str(date_col)]
        if len(non_date_cols) == 1:
            close_col = non_date_cols[0]
    if close_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date", ascending=True)
    return out[["date", "close"]]


def fetch_cboe_daily_close(req: FetchRequest, *, timeout_s: float = 15.0, retries: int = 2) -> pd.DataFrame:
    idx = str(req.index or "").strip().upper()
    url = _CBOE_CSV.get(idx)
    if not url:
        return pd.DataFrame()

    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame()

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            r = httpx.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = _parse_cboe_history_csv(r.text)
            if df.empty:
                return pd.DataFrame()
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
            return df
        except (httpx.HTTPError, ValueError, TypeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                continue
            logger.warning("cboe fetch failed index=%s err=%s", idx, e)
            break

    if last_err is not None:
        return pd.DataFrame()
    return pd.DataFrame()

