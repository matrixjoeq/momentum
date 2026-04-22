"""
Synthesize dominant continuous futures prices (88/888/889) from deliverable-month contract data.

This module implements the logic from futures_continuous_replay.py to synthesize:
- 88: synthetic no adjustment (raw/no adjustment)
- 888: synthetic forward adjustment (qfq)
- 889: synthetic backward adjustment (hfq)

The synthesized data is stored in the same futures_prices table with adjust='none', 'qfq', 'hfq'
respectively, and code suffixes 88, 888, 889.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..db.futures_repo import FuturesPriceRow, upsert_futures_prices
from ..db.models import FuturesPool

logger = logging.getLogger(__name__)

NUM_COLS = ["open", "high", "low", "close", "volume", "hold", "settle", "amount"]
PRICE_COLS = ["open", "high", "low", "close", "settle"]


def _symbol_root_from_main(code: str) -> str:
    """Extract the alphabetical root from a futures main symbol (e.g., RB0 -> RB)."""
    c = str(code or "").strip().upper()
    i = 0
    while i < len(c) and c[i].isalpha():
        i += 1
    return c[:i] if i else c


def _month_iter(start_yymm: str, end_yymm: str) -> list[str]:
    """Generate list of YYMM strings from start to end (inclusive)."""
    sy = int(start_yymm[:2])
    sm = int(start_yymm[2:])
    ey = int(end_yymm[:2])
    em = int(end_yymm[2:])
    out: list[str] = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append(f"{y:02d}{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _load_contract_data(
    db: Session,
    pool_id: int,
    root: str,
    contract_codes: list[str],
    start_date: dt.date,
    end_date: dt.date,
) -> dict[str, pd.DataFrame]:
    """
    Load price data for given contract codes from the database.
    Returns a dict mapping contract_code -> DataFrame with date index.
    """
    from ..db.futures_repo import list_futures_prices

    data: dict[str, pd.DataFrame] = {}
    for contract_code in contract_codes:
        rows = list_futures_prices(
            db,
            code=contract_code,
            start_date=start_date,
            end_date=end_date,
            adjust="none",
            limit=100000,
        )
        if not rows:
            continue
        df = pd.DataFrame(
            [
                {
                    "date": r.trade_date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "settle": r.settle,
                    "volume": r.volume,
                    "amount": r.amount,
                    "hold": r.hold,
                }
                for r in rows
            ]
        )
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        data[contract_code] = df
    return data


def _build_hold_table(contract_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a pivot table of holdings by date and contract."""
    pieces: list[pd.DataFrame] = []
    for sym, df in contract_data.items():
        if "hold" not in df.columns:
            continue
        x = df[["hold"]].copy()
        x["symbol"] = sym
        pieces.append(x)
    if not pieces:
        return pd.DataFrame()
    panel = pd.concat(pieces, ignore_index=True)
    tbl = panel.pivot_table(
        index="date", columns="symbol", values="hold", aggfunc="last"
    ).sort_index()
    return tbl


def _build_quote_panel(
    contract_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build a dict of quote panels indexed by date."""
    out: dict[str, pd.DataFrame] = {}
    for sym, df in contract_data.items():
        out[sym] = df.copy()
    return out


def _argmax_hold(holds: pd.Series) -> str | None:
    """Return the contract code with maximum holding, or None if empty."""
    v = holds.dropna()
    if v.empty:
        return None
    return str(v.idxmax())


def _replay_dominant(
    contract_data: dict[str, pd.DataFrame], switch_threshold: float = 1.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replay dominant continuous contract based on open interest.
    Returns (replay_df, switch_df).
    """
    holds_tbl = _build_hold_table(contract_data)
    panel = _build_quote_panel(contract_data)
    if holds_tbl.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = list(holds_tbl.index)
    chosen: list[str] = []
    prev_dom: str | None = None

    for i, d in enumerate(dates):
        today_holds = holds_tbl.loc[d]
        if i == 0:
            cur = _argmax_hold(today_holds)
            if cur is None:
                continue
            chosen.append(cur)
            prev_dom = cur
            continue

        prev_d = dates[i - 1]
        prev_holds = holds_tbl.loc[prev_d]
        cur = prev_dom
        if cur is None:
            cur = _argmax_hold(today_holds)
        else:
            cur_hold = prev_holds.get(cur, np.nan)
            best = _argmax_hold(prev_holds)
            if best is not None and pd.notna(cur_hold):
                best_hold = prev_holds.get(best, np.nan)
                if (
                    best != cur
                    and pd.notna(best_hold)
                    and float(best_hold) > switch_threshold * float(cur_hold)
                ):
                    cur = best
        if cur is None or cur not in panel or d not in panel[cur].index:
            fallback = _argmax_hold(today_holds)
            if fallback is None:
                continue
            cur = fallback
        chosen.append(cur)
        prev_dom = cur

    replay_rows: list[dict[str, object]] = []
    for d, sym in zip(dates[: len(chosen)], chosen, strict=True):
        row = panel[sym].loc[d]
        replay_rows.append(
            {
                "date": d,
                "dominant_symbol": sym,
                "open": row.get("open", np.nan),
                "high": row.get("high", np.nan),
                "low": row.get("low", np.nan),
                "close": row.get("close", np.nan),
                "volume": row.get("volume", np.nan),
                "hold": row.get("hold", np.nan),
                "settle": row.get("settle", np.nan),
            }
        )
    replay_df = pd.DataFrame(replay_rows).sort_values("date")

    switches: list[dict[str, object]] = []
    if not replay_df.empty:
        prev = None
        for r in replay_df.itertuples(index=False):
            if prev is not None and r.dominant_symbol != prev:
                switches.append(
                    {
                        "date": str(r.date.date()),
                        "from_symbol": prev,
                        "to_symbol": r.dominant_symbol,
                    }
                )
            prev = r.dominant_symbol
    switch_df = pd.DataFrame(switches)
    return replay_df, switch_df


def _calc_price_diff(
    panel: dict[str, pd.DataFrame],
    old_sym: str,
    new_sym: str,
    date: pd.Timestamp,
    field: str,
) -> float | None:
    """Calculate price difference between two contracts at a specific date."""
    old_df = panel.get(old_sym)
    new_df = panel.get(new_sym)
    if old_df is None or new_df is None:
        return None
    if date not in old_df.index or date not in new_df.index:
        return None
    a = old_df.loc[date].get(field, np.nan)
    b = new_df.loc[date].get(field, np.nan)
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def _build_adjusted_continuous(
    replay88_df: pd.DataFrame, switch_df: pd.DataFrame, panel: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build forward-adjusted (888/qfq) and backward-adjusted (889/hfq) continuous series.
    """
    if replay88_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = replay88_df.copy().sort_values("date").reset_index(drop=True)
    base_idx = base.set_index("date")
    idx = base_idx.index
    loc_by_date = {d: i for i, d in enumerate(idx)}

    pre_adj = pd.Series(0.0, index=idx)
    post_adj = pd.Series(0.0, index=idx)

    if not switch_df.empty:
        for r in switch_df.itertuples(index=False):
            d = pd.to_datetime(r.date)
            if d not in loc_by_date:
                continue
            i = loc_by_date[d]
            if i <= 0:
                continue
            prev_d = idx[i - 1]
            # Forward adjustment: apply diff before the switch date
            pre_delta = _calc_price_diff(
                panel, r.from_symbol, r.to_symbol, prev_d, "close"
            )
            if pre_delta is not None:
                pre_adj.loc[idx <= prev_d] += pre_delta
            # Backward adjustment: apply diff on or after the switch date
            post_delta = _calc_price_diff(panel, r.from_symbol, r.to_symbol, d, "open")
            if post_delta is not None:
                post_adj.loc[idx >= d] += post_delta

    def apply_adj(adj: pd.Series) -> pd.DataFrame:
        out = base_idx.copy()
        for c in PRICE_COLS:
            if c not in out.columns:
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce") + adj
        # amount is not adjusted
        out["amount"] = 0.0
        return out.reset_index()

    replay888 = apply_adj(pre_adj)  # forward adjusted (qfq)
    replay889 = apply_adj(post_adj)  # backward adjusted (hfq)
    return replay888, replay889


def _df_to_price_rows(
    df: pd.DataFrame, code: str, adjust: str, pool_id: int | None
) -> list[FuturesPriceRow]:
    """Convert DataFrame to list of FuturesPriceRow."""
    rows: list[FuturesPriceRow] = []
    if df.empty:
        return rows
    for _, r in df.iterrows():
        date_val = r["date"]
        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        elif isinstance(date_val, str):
            date_val = pd.to_datetime(date_val).date()
        rows.append(
            FuturesPriceRow(
                code=code,
                trade_date=date_val,
                open=r.get("open"),
                high=r.get("high"),
                low=r.get("low"),
                close=r.get("close"),
                settle=r.get("settle"),
                volume=r.get("volume"),
                amount=r.get("amount"),
                hold=r.get("hold"),
                source="synthetic",
                adjust=adjust,
                pool_id=pool_id,
            )
        )
    return rows


def synthesize_continuous_for_pool(db: Session, pool: FuturesPool) -> dict[str, Any]:
    """
    Synthesize dominant continuous prices (88/888/889) for a single futures pool.
    Returns dict with 'ok', 'error', 'counts' keys.
    """
    root = _symbol_root_from_main(pool.code)

    # Determine date range from pool
    start_date = pool.start_date
    end_date = pool.end_date
    if not start_date or not end_date:
        return {"ok": False, "error": "pool missing start_date or end_date"}

    try:
        start_dt = dt.datetime.strptime(str(start_date), "%Y%m%d").date()
        end_dt = dt.datetime.strptime(str(end_date), "%Y%m%d").date()
    except ValueError:
        return {"ok": False, "error": "invalid date format"}

    # Extend end date for contract enumeration
    extend_days = pool.contract_extend_calendar_days or 366
    end_with_extend = end_dt + dt.timedelta(days=extend_days)

    # Generate contract codes to try
    start_yymm = start_dt.strftime("%y%m")
    end_yymm = end_with_extend.strftime("%y%m")
    contract_yyMMs = _month_iter(start_yymm, end_yymm)
    contract_codes = [f"{root}{m}" for m in contract_yyMMs]

    # Load contract data from database
    contract_data = _load_contract_data(
        db, int(pool.id), root, contract_codes, start_dt, end_dt
    )
    if not contract_data:
        return {"ok": False, "error": "no contract data found"}

    # Replay dominant continuous
    replay_df, switch_df = _replay_dominant(contract_data, switch_threshold=1.1)
    if replay_df.empty:
        return {"ok": False, "error": "failed to replay dominant contract"}

    # Build quote panel
    panel = _build_quote_panel(contract_data)

    # Build adjusted series
    replay888_df, replay889_df = _build_adjusted_continuous(replay_df, switch_df, panel)

    # Prepare to insert into database
    pool_id = int(pool.id)

    # Insert 88 (synthetic none)
    rows_88 = _df_to_price_rows(replay_df, f"{root}88", "none", pool_id)
    if rows_88:
        upsert_futures_prices(db, rows_88)

    # Insert 888 (synthetic qfq)
    rows_888 = _df_to_price_rows(replay888_df, f"{root}888", "qfq", pool_id)
    if rows_888:
        upsert_futures_prices(db, rows_888)

    # Insert 889 (synthetic hfq)
    rows_889 = _df_to_price_rows(replay889_df, f"{root}889", "hfq", pool_id)
    if rows_889:
        upsert_futures_prices(db, rows_889)

    db.commit()

    return {
        "ok": True,
        "counts": {
            "88": len(rows_88),
            "888": len(rows_888),
            "889": len(rows_889),
        },
    }
