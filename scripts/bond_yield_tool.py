#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Bond Yield Tool: CN10Y, CN30Y, US10Y

- Baseline start date: 2006-03-01
- Incremental SQLite upsert per instrument
- Sources: AkShare (CN 10Y/30Y), US10Y via FRED API (DGS10)
- Outputs: CSV and optional K-line PNG per instrument
- Plot/percentile window: all (default), 10y, 5y, 1y
"""

from __future__ import annotations

import argparse
import sys
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import io
import logging
import os

try:
    import requests  # type: ignore
except Exception:
    print("Missing dependency 'requests'. Install with: python3 -m pip install requests", file=sys.stderr)
    raise


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "database" / "bond_yields.db"
BASE_START = date(2006, 3, 1)

INSTRUMENTS = ["CN10Y", "CN30Y", "US10Y"]
TABLE_BY_INSTRUMENT: Dict[str, str] = {
    "CN10Y": "cn10y_yield",
    "CN30Y": "cn30y_yield",
    "US10Y": "us10y_yield",
}
FRED_API_KEY_VALUE: Optional[str] = None


@dataclass(frozen=True)
class FetchResult:
    dataframe: pd.DataFrame  # columns: [date, yield_pct]
    source: str


def _fetch_cn_from_akshare(tenor_label: str, start: str, end: Optional[str]) -> Optional[FetchResult]:
    logging.info(f"[fetch-akshare] tenor={tenor_label} start={start} end={end}")
    try:
        import akshare as ak  # type: ignore
    except Exception:
        logging.warning("[fetch-akshare] akshare not available")
        return None
    try:
        end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        start_dt = pd.to_datetime(start).date()
        end_dt = pd.to_datetime(end_str).date()
        def to_str(d: date) -> str:
            return pd.Timestamp(d).strftime("%Y-%m-%d")
        frames: List[pd.DataFrame] = []
        cur = start_dt
        while cur <= end_dt:
            year_end = pd.Timestamp(year=cur.year, month=12, day=31).date()
            if year_end > end_dt:
                year_end = end_dt
            try:
                df_part = ak.bond_china_yield(start_date=to_str(cur), end_date=to_str(year_end))
            except Exception:
                df_part = ak.bond_china_yield(start_date=to_str(cur).replace("-", ""), end_date=to_str(year_end).replace("-", ""))
            if df_part is not None and not df_part.empty:
                frames.append(df_part)
            logging.debug(f"[fetch-akshare] collected year {cur.year}, part_rows={(0 if df_part is None else len(df_part))}")
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1).date()
        if not frames:
            logging.warning("[fetch-akshare] empty frames")
            return None
        df_all = pd.concat(frames, ignore_index=True)
        if "日期" not in df_all.columns:
            return None
        if "曲线名称" in df_all.columns:
            mask = df_all["曲线名称"].astype(str).str.contains("国债", na=False)
            if mask.any():
                df_all = df_all.loc[mask]
        tenor_candidates = [tenor_label, tenor_label.replace("年", "年期"), "10Y" if tenor_label.startswith("10") else "30Y"]
        sel = None
        for t in tenor_candidates:
            if t in df_all.columns:
                sel = t
                break
        if sel is None:
            return None
        df = df_all[["日期", sel]].rename(columns={"日期": "date", sel: "yield_pct"}).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")
        df = df.dropna(subset=["date", "yield_pct"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            logging.warning("[fetch-akshare] empty df after cleaning")
            return None
        # apply start bound
        sdt = pd.to_datetime(start).date()
        df = df[df["date"] >= sdt].reset_index(drop=True)
        logging.info(f"[fetch-akshare] ok rows={len(df)}")
        return FetchResult(df, source="akshare_bond_china_yield")
    except Exception:
        logging.exception("[fetch-akshare] exception")
        return None


def _fetch_investing_hist(urls: List[str]) -> Optional[FetchResult]:
    # Removed: no web-scraping fallbacks allowed
    return None


def fetch_history(instrument: str, start: str, end: Optional[str]) -> FetchResult:
    logging.info(f"[fetch] instrument={instrument} start={start} end={end}")
    if instrument == "CN10Y":
        res = _fetch_cn_from_akshare("10年", start, end)
        if res is not None:
            return res
        raise RuntimeError("AkShare fetch failed for CN10Y")
    elif instrument == "CN30Y":
        res = _fetch_cn_from_akshare("30年", start, end)
        if res is not None:
            return res
        raise RuntimeError("AkShare fetch failed for CN30Y")
    elif instrument == "US10Y":
        # FRED API only (series DGS10)
        try:
            from fredapi import Fred  # type: ignore
        except Exception:
            logging.error("fredapi not installed. Install with: python3 -m pip install fredapi")
            raise
        api_key = FRED_API_KEY_VALUE
        if not api_key:
            raise RuntimeError("Missing FRED API key. Provide via --fred-api-key")
        fred = Fred(api_key=api_key)
        series = fred.get_series("DGS10", observation_start=start)
        if series is None or series.empty:
            raise RuntimeError("FRED returned empty series for DGS10")
        df = series.reset_index()
        df.columns = ["date", "yield_pct"]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")
        df = df.dropna(subset=["date", "yield_pct"]).sort_values("date").reset_index(drop=True)
        logging.info(f"[fetch-fredapi] ok rows={len(df)}")
        return FetchResult(df, source="fredapi_DGS10")
    else:
        raise ValueError(f"Unsupported instrument: {instrument}")
    raise RuntimeError(f"All sources failed for {instrument}")


def ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT PRIMARY KEY,
            yield_pct REAL NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_last_date(conn: sqlite3.Connection, table_name: str) -> Optional[date]:
    cur = conn.execute(f"SELECT MAX(date) FROM {table_name}")
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    try:
        return pd.to_datetime(row[0]).date()
    except Exception:
        return None


def upsert_rows(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame, source: str) -> int:
    if df.empty:
        return 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(pd.to_datetime(d).strftime("%Y-%m-%d"), float(v), source, ts) for d, v in zip(df["date"], df["yield_pct"]) ]
    conn.executemany(
        f"""
        INSERT INTO {table_name}(date, yield_pct, source, updated_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            yield_pct=excluded.yield_pct,
            source=excluded.source,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()
    logging.info(f"[db] upserted rows={len(rows)} into {table_name}")
    return len(rows)


def derive_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    s = df["yield_pct"].astype(float)
    open_series = s.shift(1).fillna(s)
    close_series = s
    high_series = np.maximum(open_series.values, close_series.values)
    low_series = np.minimum(open_series.values, close_series.values)
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["date"]),
            "Open": open_series.values,
            "High": high_series,
            "Low": low_series,
            "Close": close_series.values,
        }
    )
    out = out.set_index("Date")
    return out


def compute_percentile(value: float, series: pd.Series) -> float:
    arr = series.astype(float).values
    if arr.size == 0:
        return float("nan")
    return float(((arr <= value).sum() / float(arr.size)) * 100.0)


def _csv_path_for(instrument: str) -> Path:
    return TOOLS_DIR / f"{instrument.lower()}_history.csv"


def _png_path_for(instrument: str) -> Path:
    return TOOLS_DIR / f"{instrument.lower()}_kline.png"


def _pick_fallback_path(path: Path) -> Path:
    base = path.with_suffix("")
    suffix = path.suffix or ".csv"
    for i in range(1, 100):
        cand = Path(f"{base}-alt{i}{suffix}")
        if not cand.exists():
            return cand
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(f"{base}-alt-{ts}{suffix}")


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except Exception:
        alt = _pick_fallback_path(path)
        df.to_csv(alt, index=False)
        return alt


def plot_kline(ohlc: pd.DataFrame, out_path: Path, title: str) -> Tuple[bool, Optional[str]]:
    try:
        import mplfinance as mpf  # type: ignore
    except Exception:
        return False, "Missing mplfinance. Install with: python3 -m pip install mplfinance"
    try:
        mpf.plot(
            ohlc,
            type="candle",
            style="yahoo",
            title=title,
            ylabel="Yield (%)",
            savefig=dict(fname=str(out_path), dpi=140, bbox_inches="tight"),
        )
        return True, None
    except Exception as e:
        return False, str(e)


def _apply_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty:
        return df
    window = (window or "all").lower()
    if window == "all":
        return df
    dates = pd.to_datetime(df["date"], errors="coerce")
    last_ts = dates.iloc[-1]
    if pd.isna(last_ts):
        return df
    if window == "10y":
        cutoff = last_ts - pd.DateOffset(years=10)
    elif window == "5y":
        cutoff = last_ts - pd.DateOffset(years=5)
    elif window == "1y":
        cutoff = last_ts - pd.DateOffset(years=1)
    else:
        return df
    mask = dates >= cutoff
    return df.loc[mask].reset_index(drop=True)


def run_for_instrument(instrument: str, db_path: Path, do_plot: bool, window: str) -> None:
    table = TABLE_BY_INSTRUMENT[instrument]
    with sqlite3.connect(db_path) as conn:
        ensure_table(conn, table)
        last = get_last_date(conn, table)
        if last is None:
            fetch_start = BASE_START
        else:
            next_day = last + timedelta(days=1)
            fetch_start = BASE_START if next_day < BASE_START else next_day
    logging.info(f"[run] {instrument} fetch_start={fetch_start}")
    result = None
    df = None
    try:
        result = fetch_history(instrument, start=str(fetch_start), end=None)
        df = result.dataframe
    except Exception as e:
        # Graceful no-op if no new data available (today or beyond last available date)
        today = datetime.utcnow().date()
        if instrument in ("CN10Y", "CN30Y", "US10Y") and (fetch_start >= today or (last is not None and fetch_start > last)):
            logging.info(f"[run] {instrument} no new data for {fetch_start}, skipping fetch and keeping DB unchanged")
            with sqlite3.connect(db_path) as conn:
                df_all = pd.read_sql_query(f"SELECT date, yield_pct FROM {table} ORDER BY date", conn, parse_dates=["date"])  # type: ignore
            # Summarize and return
            csv_path_used = save_csv(df_all, _csv_path_for(instrument))
            df_win = _apply_window(df_all, window)
            ohlc = derive_ohlc(df_win)
            latest_date = pd.to_datetime(df_all["date"].iloc[-1]).date()
            latest_yield = float(df_all["yield_pct"].iloc[-1])
            pct = compute_percentile(latest_yield, df_win["yield_pct"])  # percentage of window <= current
            print(f"=== {instrument} Yield Summary ===")
            print(f"Source            : (no new data)")
            print(f"History rows      : {df_all.shape[0]}")
            print(f"First date        : {pd.to_datetime(df_all['date'].iloc[0]).date()}")
            print(f"Last date         : {latest_date}")
            print(f"Latest yield (%)  : {latest_yield:.3f}")
            print(f"Window            : {window}")
            print(f"Historical percentile (window)  : {pct:.2f}%")
            print(f"CSV saved to      : {csv_path_used.resolve()}")
            if do_plot:
                plotted, plot_err = plot_kline(ohlc, _png_path_for(instrument), title=f"{instrument} Yield [{window}] (up to {latest_date})")
                if not plotted:
                    logging.warning(f"[plot] failed: {plot_err}")
            return
        # For other errors, re-raise
        raise
    df_incr = df[df["date"] >= fetch_start].copy()
    with sqlite3.connect(db_path) as conn:
        ensure_table(conn, table)
        upsert_rows(conn, table, df_incr, result.source)
        df_all = pd.read_sql_query(f"SELECT date, yield_pct FROM {table} ORDER BY date", conn, parse_dates=["date"])  # type: ignore
    # DB stats
    cnt = len(df_all)
    min_d = pd.to_datetime(df_all["date"].iloc[0]).date() if cnt > 0 else None
    max_d = pd.to_datetime(df_all["date"].iloc[-1]).date() if cnt > 0 else None
    logging.info(f"[db] {instrument} total_rows={cnt} range=[{min_d}..{max_d}]")
    csv_path_used = save_csv(df_all, _csv_path_for(instrument))
    df_win = _apply_window(df_all, window)
    ohlc = derive_ohlc(df_win)
    latest_date = pd.to_datetime(df_all["date"].iloc[-1]).date()
    latest_yield = float(df_all["yield_pct"].iloc[-1])
    pct = compute_percentile(latest_yield, df_win["yield_pct"])  # percentage of window <= current
    plotted = True
    plot_err: Optional[str] = None
    if do_plot:
        plotted, plot_err = plot_kline(ohlc, _png_path_for(instrument), title=f"{instrument} Yield [{window}] (up to {latest_date})")
        if not plotted:
            logging.warning(f"[plot] failed: {plot_err}")
    print(f"=== {instrument} Yield Summary ===")
    print(f"Source            : {result.source}")
    print(f"History rows      : {df_all.shape[0]}")
    print(f"First date        : {pd.to_datetime(df_all['date'].iloc[0]).date()}")
    print(f"Last date         : {latest_date}")
    print(f"Latest yield (%)  : {latest_yield:.3f}")
    print(f"Window            : {window}")
    print(f"Historical percentile (window)  : {pct:.2f}%")
    print(f"CSV saved to      : {csv_path_used.resolve()}")
    if do_plot:
        if plotted:
            print(f"PNG saved to      : {_png_path_for(instrument).resolve()}")
        else:
            print(f"Plotting skipped/failed: {plot_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified bond yields tool (CN10Y, CN30Y, US10Y)")
    parser.add_argument("--instruments", default="CN10Y,CN30Y,US10Y", help="Comma list from CN10Y,CN30Y,US10Y or 'all'")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--window", choices=["all", "10y", "5y", "1y"], default="all", help="Plotting/percentile window")
    parser.add_argument("--full-update", action="store_true", help="Purge selected instruments' tables before refetching full history")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--fred-api-key", default=None, help="FRED API key for US10Y")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    instr = args.instruments
    if instr.strip().lower() == "all":
        targets = INSTRUMENTS
    else:
        targets = [x.strip().upper() for x in instr.split(",") if x.strip()]
        for t in targets:
            if t not in INSTRUMENTS:
                print(f"Unsupported instrument: {t}", file=sys.stderr)
                return 2

    db_path = Path(args.db)
    # Capture FRED key
    global FRED_API_KEY_VALUE
    FRED_API_KEY_VALUE = args.fred_api_key
    had_error = False
    for t in targets:
        try:
            if args.full_update:
                table = TABLE_BY_INSTRUMENT[t]
                with sqlite3.connect(db_path) as conn:
                    ensure_table(conn, table)
                    logging.info(f"[full-update] purging table {table}")
                    conn.execute(f"DELETE FROM {table}")
                    conn.commit()
            run_for_instrument(t, db_path, do_plot=(not args.no_plot), window=args.window)
        except Exception:
            had_error = True
            logging.exception(f"[run] instrument {t} failed")
    # DB summary after run
    try:
        with sqlite3.connect(db_path) as conn:
            for t in targets:
                table = TABLE_BY_INSTRUMENT[t]
                ensure_table(conn, table)
                df_sum = pd.read_sql_query(f"SELECT MIN(date) as mind, MAX(date) as maxd, COUNT(1) as cnt FROM {table}", conn)
                mind = df_sum.loc[0, 'mind']
                maxd = df_sum.loc[0, 'maxd']
                cnt = int(df_sum.loc[0, 'cnt']) if not pd.isna(df_sum.loc[0, 'cnt']) else 0
                logging.info(f"[summary] {t}: rows={cnt} range=[{mind}..{maxd}]")
    except Exception:
        logging.exception("[summary] failed to read DB summary")
    if had_error:
        logging.warning("One or more instruments failed. See logs above. Partial updates may have been applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


