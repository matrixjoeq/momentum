from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..data.fred_fetcher import FetchRequest as FredFetchRequest
from ..data.fred_fetcher import fetch_fred_daily_close
from ..data.sina_fetcher import FetchRequest as SinaFetchRequest
from ..data.sina_fetcher import fetch_sina_forex_day_kline_daily_close
from ..data.stooq_fetcher import FetchRequest as StooqFetchRequest
from ..data.stooq_fetcher import fetch_stooq_daily_close
from ..data.yahoo_fetcher import FetchRequest as YahooFetchRequest
from ..data.yahoo_fetcher import fetch_yahoo_daily_close
from ..db.repo import (
    MacroPriceRow,
    create_macro_ingestion_batch,
    upsert_macro_prices,
    upsert_macro_series_meta,
    update_macro_ingestion_batch,
)
from ..settings import get_settings


@dataclass(frozen=True)
class MacroSeriesSpec:
    series_id: str
    provider: str  # fred|stooq|yahoo
    provider_symbol: str
    name: str | None = None
    category: str | None = None
    unit: str | None = None
    timezone: str | None = None
    calendar: str | None = None


DEFAULT_STEP1_SERIES: list[MacroSeriesSpec] = [
    MacroSeriesSpec(series_id="DGS2", provider="fred", provider_symbol="DGS2", name="US Treasury 2Y", category="rates", unit="%"),
    MacroSeriesSpec(series_id="DGS5", provider="fred", provider_symbol="DGS5", name="US Treasury 5Y", category="rates", unit="%"),
    MacroSeriesSpec(series_id="DGS10", provider="fred", provider_symbol="DGS10", name="US Treasury 10Y", category="rates", unit="%"),
    MacroSeriesSpec(series_id="DGS30", provider="fred", provider_symbol="DGS30", name="US Treasury 30Y", category="rates", unit="%"),
    # DXY / USDX: use Sina's DINIW which matches common quote conventions (e.g. ~96 not ~62).
    MacroSeriesSpec(series_id="DINIW", provider="sina", provider_symbol="DINIW", name="US Dollar Index (DXY)", category="fx", unit="index"),
    MacroSeriesSpec(series_id="XAUUSD", provider="stooq", provider_symbol="XAUUSD", name="Gold Spot (XAUUSD)", category="gold_spot", unit="USD/oz"),
    # Gold futures: Yahoo fallback; may be blocked depending on network policy.
    MacroSeriesSpec(series_id="GC_FUT", provider="yahoo", provider_symbol="GC=F", name="Gold Futures (GC=F)", category="gold_fut", unit="USD/oz"),
]


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _df_to_rows(df: pd.DataFrame, *, series_id: str, source: str) -> list[MacroPriceRow]:
    if df is None or df.empty:
        return []
    out: list[MacroPriceRow] = []
    for d, v in zip(df["date"].to_list(), df["close"].to_list(), strict=False):
        if not isinstance(d, dt.date):
            continue
        # pandas may represent missing values as NaN; MySQL driver cannot bind NaN.
        if v is None or pd.isna(v):
            fv = None
        else:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = None
        out.append(MacroPriceRow(series_id=series_id, trade_date=d, close=fv, source=source))
    return out


def fetch_macro_daily_close(
    spec: MacroSeriesSpec,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prov = str(spec.provider).strip().lower()
    sym = str(spec.provider_symbol).strip()
    if prov == "fred":
        settings = get_settings()
        return fetch_fred_daily_close(FredFetchRequest(series_id=sym, start_date=start, end_date=end), api_key=settings.fred_api_key)
    if prov == "sina":
        return fetch_sina_forex_day_kline_daily_close(SinaFetchRequest(symbol=sym, start_date=start, end_date=end))
    if prov == "stooq":
        return fetch_stooq_daily_close(StooqFetchRequest(symbol=sym, start_date=start, end_date=end))
    if prov == "yahoo":
        df = fetch_yahoo_daily_close(YahooFetchRequest(symbol=sym, start_date=start, end_date=end))
        return df, {"provider": "yahoo", "symbol": sym}
    return pd.DataFrame(), {"provider": prov, "symbol": sym, "error": "unsupported_provider"}


def ingest_macro_series(
    db: Session,
    *,
    spec: MacroSeriesSpec,
    start: str,
    end: str,
) -> dict[str, Any]:
    """
    Full refresh ingestion: fetch [start,end] and upsert into macro_prices.
    Conflicts are overwritten by fetched data (updated wins).
    """
    b = create_macro_ingestion_batch(db, series_id=spec.series_id, provider=spec.provider, start_date=start, end_date=end)
    db.commit()

    upsert_macro_series_meta(
        db,
        series_id=spec.series_id,
        provider=spec.provider,
        provider_symbol=spec.provider_symbol,
        name=spec.name,
        category=spec.category,
        unit=spec.unit,
        timezone=spec.timezone,
        calendar=spec.calendar,
    )
    db.commit()

    df, meta = fetch_macro_daily_close(spec, start=start, end=end)
    if df is None or df.empty:
        err = str((meta or {}).get("error") or "empty_fetch")
        update_macro_ingestion_batch(db, batch_id=b.id, status="failed", message=err)
        db.commit()
        return {"ok": False, "batch_id": b.id, "series_id": spec.series_id, "error": err, "meta": meta}

    rows = _df_to_rows(df, series_id=spec.series_id, source=str((meta or {}).get("provider") or spec.provider))
    n = upsert_macro_prices(db, rows)
    update_macro_ingestion_batch(db, batch_id=b.id, status="success", message=f"rows={len(rows)} upserted={n}")
    db.commit()
    return {"ok": True, "batch_id": b.id, "series_id": spec.series_id, "upserted": n, "meta": meta}

