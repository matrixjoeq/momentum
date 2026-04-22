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

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.futures_repo import FuturesPriceRow, upsert_futures_prices
from ..db.models import FuturesPool, FuturesPrice

logger = logging.getLogger(__name__)

NUM_COLS = ["open", "high", "low", "close", "volume", "hold", "settle", "amount"]
PRICE_COLS = ["open", "high", "low", "close", "settle"]
ERROR_FIELDS = ["open", "high", "low", "close", "volume", "amount", "hold", "settle"]
KEY_FIELDS = ["open", "high", "low", "close", "settle"]

USABLE_REL_MEAN_MAX = 0.005
USABLE_REL_P95_MAX = 0.02
USABLE_MIN_FIELDS = 4


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
    contract_codes: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Load price data for given contract codes from the database.
    Returns a dict mapping contract_code -> DataFrame with `date` column.
    This intentionally matches futures_continuous_replay.py so downstream
    hold-table and quote-panel logic stay consistent.
    """
    from ..db.futures_repo import list_futures_prices

    data: dict[str, pd.DataFrame] = {}
    for contract_code in contract_codes:
        rows = list_futures_prices(
            db,
            code=contract_code,
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
        df = df.sort_values("date")
        data[contract_code] = df
    return data


def _discover_contract_codes(db: Session, *, pool: FuturesPool, root: str) -> list[str]:
    """
    Discover deliverable-month contract codes already stored for this pool.
    This avoids requiring pool.start_date/end_date.
    """
    pid = int(pool.id)
    pool_code = str(pool.code or "").strip().upper()
    root_u = str(root or "").strip().upper()
    patt = re.compile(rf"^{re.escape(root_u)}(\d{{3,4}})$")

    rows = db.execute(
        select(FuturesPrice.code).where(
            FuturesPrice.pool_id == pid,
            FuturesPrice.adjust == "none",
        )
    ).all()
    seen: set[str] = set()
    out: list[str] = []
    for (code_raw,) in rows:
        code = str(code_raw or "").strip().upper()
        if not code or code == pool_code:
            continue
        m = patt.match(code)
        if m is None:
            continue
        suffix = m.group(1)
        if suffix in {"88", "888", "889"}:
            continue
        if code not in seen:
            seen.add(code)
            out.append(code)
    return sorted(out)


def _build_hold_table(contract_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a pivot table of holdings by date and contract."""
    pieces: list[pd.DataFrame] = []
    for sym, df in contract_data.items():
        if "date" not in df.columns or "hold" not in df.columns:
            continue
        x = df[["date", "hold"]].copy()
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
        out[sym] = df.copy().set_index("date").sort_index()
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

    # Discover contract codes from already fetched/landed data for this pool.
    contract_codes = _discover_contract_codes(db, pool=pool, root=root)
    if not contract_codes:
        return {"ok": False, "error": "no deliverable contract data found"}

    # Load contract data from database
    contract_data = _load_contract_data(db, contract_codes)
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


def _load_price_df(db: Session, *, code: str, adjust: str) -> pd.DataFrame:
    from ..db.futures_repo import list_futures_prices

    rows = list_futures_prices(db, code=code, adjust=adjust, limit=200000)
    if not rows:
        return pd.DataFrame(columns=["date", *NUM_COLS])
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
        return pd.DataFrame(columns=["date", *NUM_COLS])
    df["date"] = pd.to_datetime(df["date"])
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", *NUM_COLS]].sort_values("date")


def _build_joined_for_error(
    replay88_df: pd.DataFrame, main_df: pd.DataFrame
) -> pd.DataFrame:
    if replay88_df.empty or main_df.empty:
        return pd.DataFrame()
    left = replay88_df.copy().set_index("date")
    right = main_df.copy().set_index("date")
    return left.join(right, how="inner", lsuffix="_replay88", rsuffix="_main0")


def _calc_error_stats(joined_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for f in ERROR_FIELDS:
        col_a = f"{f}_replay88"
        col_b = f"{f}_main0"
        a_raw = (
            joined_df[col_a]
            if col_a in joined_df.columns
            else pd.Series(np.nan, index=joined_df.index)
        )
        b_raw = (
            joined_df[col_b]
            if col_b in joined_df.columns
            else pd.Series(np.nan, index=joined_df.index)
        )
        a = pd.to_numeric(a_raw, errors="coerce")
        b = pd.to_numeric(b_raw, errors="coerce")
        valid = a.notna() & b.notna()
        n = int(valid.sum())
        if n == 0:
            rows.append(
                {
                    "field": f,
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "max_abs": np.nan,
                    "mape": np.nan,
                    "p95_ape": np.nan,
                }
            )
            continue
        diff = (a[valid] - b[valid]).astype(float)
        abs_diff = diff.abs()
        denom = b[valid].replace(0, np.nan).abs()
        ape = abs_diff / denom
        mape = ape.mean(skipna=True)
        p95_ape = ape.quantile(0.95)
        rows.append(
            {
                "field": f,
                "n": n,
                "mae": float(abs_diff.mean()),
                "rmse": float(np.sqrt((diff**2).mean())),
                "max_abs": float(abs_diff.max()),
                "mape": float(mape) if pd.notna(mape) else np.nan,
                "p95_ape": float(p95_ape) if pd.notna(p95_ape) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_usability(err_df: pd.DataFrame, *, compare_ok: bool) -> dict[str, Any]:
    if not compare_ok:
        return {
            "usable": False,
            "reason": "comparison not available",
            "rule": {
                "rel_mean_max": USABLE_REL_MEAN_MAX,
                "rel_p95_max": USABLE_REL_P95_MAX,
                "min_fields": USABLE_MIN_FIELDS,
            },
            "covered_fields": [],
            "failed_fields": [],
        }

    covered: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for f in KEY_FIELDS:
        row = err_df[err_df["field"] == f]
        if row.empty:
            continue
        r = row.iloc[0]
        n = int(r.get("n", 0) or 0)
        if n <= 0:
            continue
        mape = float(r.get("mape", np.nan))
        p95 = float(r.get("p95_ape", np.nan))
        item = {"field": f, "n": n, "mape": mape, "p95_ape": p95}
        covered.append(item)
        if (not np.isnan(mape) and mape > USABLE_REL_MEAN_MAX) or (
            not np.isnan(p95) and p95 > USABLE_REL_P95_MAX
        ):
            failed.append(item)
    usable = len(covered) >= USABLE_MIN_FIELDS and len(failed) == 0
    reason = (
        "pass"
        if usable
        else (
            "insufficient covered fields"
            if len(covered) < USABLE_MIN_FIELDS
            else "relative error threshold exceeded"
        )
    )
    return {
        "usable": usable,
        "reason": reason,
        "rule": {
            "rel_mean_max": USABLE_REL_MEAN_MAX,
            "rel_p95_max": USABLE_REL_P95_MAX,
            "min_fields": USABLE_MIN_FIELDS,
        },
        "covered_fields": covered,
        "failed_fields": failed,
    }


def validate_synthesized_for_pool(db: Session, pool: FuturesPool) -> dict[str, Any]:
    """
    Validate synthesized continuous futures for one pool.
    Skip when no synthesized data exists.
    """
    root = _symbol_root_from_main(pool.code)
    code_main = str(pool.code or "").strip().upper()
    code_88 = f"{root}88"
    code_888 = f"{root}888"
    code_889 = f"{root}889"

    df_88 = _load_price_df(db, code=code_88, adjust="none")
    df_888 = _load_price_df(db, code=code_888, adjust="qfq")
    df_889 = _load_price_df(db, code=code_889, adjust="hfq")

    if df_88.empty and df_888.empty and df_889.empty:
        return {
            "code": code_main,
            "status": "skipped",
            "conclusion": "跳过：无合成数据",
            "details": {"reason": "no synthesized data"},
        }

    if df_88.empty:
        return {
            "code": code_main,
            "status": "failed",
            "conclusion": "失败：缺少无复权合成数据",
            "details": {"reason": "missing none-adjust synthesized series"},
        }

    none_dates = pd.Index(sorted(pd.to_datetime(df_88["date"]).dropna().unique()))
    if none_dates.empty:
        return {
            "code": code_main,
            "status": "failed",
            "conclusion": "失败：无复权合成数据为空",
            "details": {"reason": "empty none-adjust synthesized date set"},
        }
    none_start = pd.Timestamp(none_dates.min())
    none_end = pd.Timestamp(none_dates.max())

    def _coverage_for(df_adj: pd.DataFrame, adj_name: str) -> dict[str, Any]:
        adj_dates = pd.Index([])
        if not df_adj.empty:
            all_dates = pd.to_datetime(df_adj["date"]).dropna()
            in_span = all_dates[(all_dates >= none_start) & (all_dates <= none_end)]
            adj_dates = pd.Index(sorted(in_span.unique()))
        missing = none_dates.difference(adj_dates)
        extra = adj_dates.difference(none_dates)
        return {
            "adjust": adj_name,
            "rows_total": int(len(df_adj)),
            "rows_in_none_span": int(len(adj_dates)),
            "start": (
                str(pd.Timestamp(adj_dates.min()).date())
                if len(adj_dates) > 0
                else None
            ),
            "end": (
                str(pd.Timestamp(adj_dates.max()).date())
                if len(adj_dates) > 0
                else None
            ),
            "missing_days": int(len(missing)),
            "extra_days": int(len(extra)),
            "missing_dates_sample": [
                str(pd.Timestamp(x).date()) for x in list(missing[:5])
            ],
            "extra_dates_sample": [
                str(pd.Timestamp(x).date()) for x in list(extra[:5])
            ],
            "same_span": bool(
                len(adj_dates) > 0
                and pd.Timestamp(adj_dates.min()) == none_start
                and pd.Timestamp(adj_dates.max()) == none_end
            ),
            "same_date_set": bool(len(missing) == 0 and len(extra) == 0),
        }

    cov_888 = _coverage_for(df_888, "qfq")
    cov_889 = _coverage_for(df_889, "hfq")
    coverage_pass = bool(
        cov_888["same_date_set"]
        and cov_888["same_span"]
        and cov_889["same_date_set"]
        and cov_889["same_span"]
    )

    df_main = _load_price_df(db, code=code_main, adjust="none")
    joined = _build_joined_for_error(df_88, df_main)
    compare_ok = not joined.empty
    err_df = _calc_error_stats(joined) if compare_ok else pd.DataFrame()
    usability = _evaluate_usability(err_df, compare_ok=compare_ok)
    error_pass = bool(usability.get("usable", False))

    overall_pass = bool(coverage_pass and error_pass)
    status = "passed" if overall_pass else "failed"
    conclusion = (
        "通过：合成数据校验通过" if overall_pass else "失败：合成数据校验未通过"
    )

    compare_summary: dict[str, Any]
    if compare_ok:
        joined_x = joined.copy()
        joined_x["close_abs_diff"] = (
            joined_x["close_replay88"] - joined_x["close_main0"]
        ).abs()
        joined_x["close_rel_diff"] = (
            joined_x["close_abs_diff"]
            / joined_x["close_main0"].replace(0, np.nan).abs()
        )
        compare_summary = {
            "ok": True,
            "overlap_days": int(len(joined_x)),
            "date_start": str(pd.Timestamp(joined_x.index.min()).date()),
            "date_end": str(pd.Timestamp(joined_x.index.max()).date()),
            "close_rel_diff_mean": float(joined_x["close_rel_diff"].mean(skipna=True)),
            "close_rel_diff_p95": float(joined_x["close_rel_diff"].quantile(0.95)),
        }
    else:
        compare_summary = {"ok": False, "reason": "no overlap with main0"}

    return {
        "code": code_main,
        "status": status,
        "conclusion": conclusion,
        "details": {
            "rule_source": "futures_continuous_replay.py usability thresholds",
            "coverage_check": {
                "none_start": str(none_start.date()),
                "none_end": str(none_end.date()),
                "none_days": int(len(none_dates)),
                "qfq": cov_888,
                "hfq": cov_889,
                "passed": coverage_pass,
            },
            "main_compare_check": {
                "passed": error_pass,
                "summary": compare_summary,
                "usability": usability,
                "error_fields": (
                    []
                    if err_df.empty
                    else [
                        {
                            "field": str(r["field"]),
                            "n": int(r["n"]),
                            "mae": (None if pd.isna(r["mae"]) else float(r["mae"])),
                            "rmse": (None if pd.isna(r["rmse"]) else float(r["rmse"])),
                            "max_abs": (
                                None if pd.isna(r["max_abs"]) else float(r["max_abs"])
                            ),
                            "mape": (None if pd.isna(r["mape"]) else float(r["mape"])),
                            "p95_ape": (
                                None if pd.isna(r["p95_ape"]) else float(r["p95_ape"])
                            ),
                        }
                        for _, r in err_df.iterrows()
                    ]
                ),
            },
        },
    }
