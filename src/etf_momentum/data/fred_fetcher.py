from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchRequest:
    series_id: str  # e.g. DGS2/DGS5/DGS10/DGS30
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _iso_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def fetch_fred_daily_close(
    req: FetchRequest,
    *,
    api_key: str | None = None,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch FRED series observations (daily) and return a dataframe with columns: date, close.

    Notes:
    - FRED API typically requires an api_key.
    - Missing values are represented as "." in the API.
    """
    series_id = str(req.series_id or "").strip().upper()
    meta: dict[str, Any] = {"provider": "fred", "series_id": series_id}
    if not series_id:
        return pd.DataFrame(), {**meta, "error": "empty_series_id"}

    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame(), {**meta, "error": "end_before_start"}

    if not api_key:
        # Without key, FRED often returns 403 or an error payload. Fail fast so
        # the caller can return a clear error to the UI.
        return pd.DataFrame(), {**meta, "error": "missing_api_key"}

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": _iso_date(start),
        "observation_end": _iso_date(end),
        # keep default frequency; DGS* are daily business days
    }

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            timeout = httpx.Timeout(float(timeout_s), connect=float(timeout_s))
            r = httpx.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and (payload.get("error_code") or payload.get("error_message")):
                msg = str(payload.get("error_message") or payload.get("error_code") or "fred_error")
                return pd.DataFrame(), {**meta, "error": msg}
            obs = payload.get("observations") if isinstance(payload, dict) else None
            if not isinstance(obs, list) or not obs:
                return pd.DataFrame(), {**meta, "error": "empty_observations"}

            rows: list[dict[str, Any]] = []
            for o in obs:
                if not isinstance(o, dict):
                    continue
                d0 = o.get("date")
                v0 = o.get("value")
                if not d0:
                    continue
                d = pd.to_datetime(d0, errors="coerce")
                if pd.isna(d):
                    continue
                # FRED uses "." for missing
                v = pd.to_numeric(v0 if v0 != "." else None, errors="coerce")
                rows.append({"date": d.date(), "close": float(v) if pd.notna(v) else None})

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
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max(1, int(retries) + 1) - 1:
                # light exponential backoff to avoid hammering FRED
                time.sleep(min(2.0 ** attempt, 8.0))
                continue
            logger.warning("fred fetch failed series_id=%s err=%s", series_id, e)
            break

    return pd.DataFrame(), {**meta, "error": str(last_err) if last_err else "failed"}

