from __future__ import annotations

"""
Check whether AkShare alternative sources can cover Eastmoney ETF price content.

This script focuses on daily K-line (OHLCV) data for ETF codes like 159915.

Sources tested (as available in your installed akshare):
- eastmoney: fund_etf_hist_em (baseline; may fail if network blocked)
- sina: fund_etf_hist_sina
- tencent: stock_zh_a_hist_tx (treat ETF as A-share code; may or may not support)

Notes:
- AkShare does NOT expose "163/sohu/xueqiu" ETF daily K-line APIs in your current version.
- This script reports per-source availability, date-range coverage, and available columns.
"""

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


DEFAULT_CODES = ["159915", "511010", "513100", "518880"]


def _ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _parse_any_date(x: Any) -> dt.date | None:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    s = str(x).strip()
    if not s:
        return None
    # try ISO "YYYY-MM-DD"
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        pass
    # try "YYYYMMDD"
    try:
        return dt.datetime.strptime(s[:8], "%Y%m%d").date()
    except Exception:
        return None


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    rows: int
    start: str | None
    end: str | None
    cols: list[str]
    err: str | None = None


def _summarize_df(df: pd.DataFrame, date_col_candidates: list[str]) -> FetchResult:
    if df is None or df.empty:
        return FetchResult(ok=True, rows=0, start=None, end=None, cols=[])
    cols = list(df.columns)
    dcol = None
    for c in date_col_candidates:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        # maybe index is date
        if isinstance(df.index, pd.DatetimeIndex):
            s = df.index.min().date().isoformat()
            e = df.index.max().date().isoformat()
            return FetchResult(ok=True, rows=int(len(df)), start=s, end=e, cols=cols)
        return FetchResult(ok=True, rows=int(len(df)), start=None, end=None, cols=cols)
    dates = df[dcol].apply(_parse_any_date).dropna()
    if dates.empty:
        return FetchResult(ok=True, rows=int(len(df)), start=None, end=None, cols=cols)
    return FetchResult(ok=True, rows=int(len(df)), start=min(dates).isoformat(), end=max(dates).isoformat(), cols=cols)


def _try_fetch(fn: Callable[[], pd.DataFrame]) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = fn()
        if not isinstance(df, pd.DataFrame):
            return None, f"not a DataFrame: {type(df)}"
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(DEFAULT_CODES), help="comma-separated ETF codes")
    ap.add_argument("--start", default="20240101", help="YYYYMMDD")
    ap.add_argument("--end", default=_ymd(dt.date.today()), help="YYYYMMDD")
    ap.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none"], help="adjust basis to request when supported")
    args = ap.parse_args()

    import akshare as ak  # local import

    codes = [c.strip() for c in str(args.codes).split(",") if c.strip()]
    start = str(args.start).strip()
    end = str(args.end).strip()
    adj = "" if args.adjust == "none" else args.adjust

    def _with_ex(code6: str, *, prefer: str | None = None) -> str:
        """
        Expand 6-digit code into exchange-prefixed symbols for sources that require it.
        """
        c = str(code6).strip()
        if c.lower().startswith(("sh", "sz")):
            return c
        if prefer in {"sh", "sz"}:
            return f"{prefer}{c}"
        # heuristic for CN instruments: 5/6/9 -> sh else sz
        exch = "sh" if c and c[0] in {"5", "6", "9"} else "sz"
        return f"{exch}{c}"

    # Detect available functions in your akshare build
    has_em = hasattr(ak, "fund_etf_hist_em")
    has_sina = hasattr(ak, "fund_etf_hist_sina")
    has_tx = hasattr(ak, "stock_zh_a_hist_tx")

    print("=== AkShare alt-source coverage check (ETF daily K-line) ===")
    print(f"codes={codes}")
    print(f"range={start}..{end} adjust={args.adjust}")
    print(f"available: eastmoney={has_em} sina={has_sina} tencent={has_tx}")
    print("")

    for code in codes:
        print(f"--- {code} ---")

        if has_em:
            df, err = _try_fetch(lambda: ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=end, adjust=adj))
            if err:
                r = FetchResult(ok=False, rows=0, start=None, end=None, cols=[], err=err)
            else:
                r = _summarize_df(df, ["日期", "date"])
            print(f"[eastmoney/fund_etf_hist_em] ok={r.ok} rows={r.rows} range={r.start}..{r.end} cols={r.cols[:8]}{'...' if len(r.cols)>8 else ''} err={r.err}")
        else:
            print("[eastmoney/fund_etf_hist_em] not available")

        if has_sina:
            # fund_etf_hist_sina signature differs across versions; try minimal.
            sym = _with_ex(code)
            df, err = _try_fetch(lambda: ak.fund_etf_hist_sina(symbol=sym))
            if err:
                r = FetchResult(ok=False, rows=0, start=None, end=None, cols=[], err=err)
            else:
                r = _summarize_df(df, ["date", "日期"])
            print(f"[sina/fund_etf_hist_sina] symbol={sym} ok={r.ok} rows={r.rows} range={r.start}..{r.end} cols={r.cols[:8]}{'...' if len(r.cols)>8 else ''} err={r.err}")
        else:
            print("[sina/fund_etf_hist_sina] not available")

        if has_tx:
            # Tencent A-share history API; treat ETF code as "A-share code".
            # Some versions accept adjust like 'qfq'/'hfq'/'': we try with period and adjust if supported.
            def _tx_call():
                try:
                    return ak.stock_zh_a_hist_tx(symbol=_with_ex(code), start_date=start, end_date=end, adjust=adj)
                except TypeError:
                    return ak.stock_zh_a_hist_tx(symbol=_with_ex(code))

            df, err = _try_fetch(_tx_call)
            if err:
                r = FetchResult(ok=False, rows=0, start=None, end=None, cols=[], err=err)
            else:
                r = _summarize_df(df, ["date", "日期"])
            print(f"[tencent/stock_zh_a_hist_tx] ok={r.ok} rows={r.rows} range={r.start}..{r.end} cols={r.cols[:8]}{'...' if len(r.cols)>8 else ''} err={r.err}")
        else:
            print("[tencent/stock_zh_a_hist_tx] not available")

        print("")

    print("=== Notes ===")
    print("- Your installed akshare does not expose 163/sohu/xueqiu ETF daily K-line APIs (no matching functions).")
    print("- If you need those sources, we'd need to use non-akshare clients or upgrade/change akshare version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

