"""
Deliverable-month futures contract daily ingestion for a pooled main symbol (e.g. RB0).

Tencent Finance ``newfqkline``-style endpoints return empty domestic contract series in tests; we do not use them.

**AkShare (Sina futures_zh_daily_sina):** contract fetches are **serial only** (parallel fixed to 1), with **1 second**
between each **contract** download, and **fail-fast**: the first network/parse error aborts the rest of that pool’s
contracts and any further pools scheduled in the same batch job.

**Official exchange daily bars (via AkShare, opt-in only):** when ``MOMENTUM_FUTURES_CONTRACT_USE_OFFICIAL_EXCHANGE=1``,
deliverable months use ``get_shfe_daily`` / ``get_ine_daily`` / ``get_gfex_daily`` instead of Sina (one HTTP per
trading day, **1 second** between requests). By default this is **off**. There is **no** automatic switch from
Sina to exchange when Sina returns empty: enumerated months are heuristic; empty series means “no such contract
for this code” and ingestion **continues** to the next contract. ``amount`` maps from turnover on official panels.

**CTP:** real-time and settlement feeds use the **broker CTP (FTD) binary protocol**, not a public REST API.
Integrating CTP would require a native SDK, broker credentials, and deployment outside this HTTP service; it is not
implemented here.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..db.futures_repo import (
    FuturesPriceRow,
    delete_contract_fetch_status,
    delete_futures_prices,
    get_futures_date_range,
    get_futures_last_trade_date,
    get_futures_pool_by_code,
    mark_futures_contract_pool_fetch,
    record_contract_fetch_status,
    upsert_futures_prices,
)
from ..settings import get_settings
from .futures_ingestion import (
    _fetch_futures_daily_sina_df,
    _normalize_futures_df,
    _parse_yyyymmdd,
    _to_float,
)

logger = logging.getLogger(__name__)


class ContractFetchAborted(Exception):
    """Raised when a contract fetch fails and subsequent work must stop (fail-fast)."""


def _symbol_root_from_main(code: str) -> str:
    c = str(code or "").strip().upper()
    i = 0
    while i < len(c) and c[i].isalpha():
        i += 1
    return c[:i] if i else c


def _month_iter(start_yymm: str, end_yymm: str) -> list[str]:
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


def _yymm_from_yyyymmdd(s: str) -> str:
    d = _parse_yyyymmdd(s)
    return d.strftime("%y%m")


def _align_weekend_forward(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _contract_codes_in_window(*, root: str, main_start: str, main_end: str, extend_calendar_days: int) -> list[str]:
    end_raw = _parse_yyyymmdd(main_end) + dt.timedelta(days=int(extend_calendar_days))
    end_d = _align_weekend_forward(end_raw)
    start_yymm = _yymm_from_yyyymmdd(main_start)
    end_yymm = end_d.strftime("%y%m")
    months = _month_iter(start_yymm, end_yymm)
    r = root.upper()
    return [f"{r}{m}" for m in months]


def _trading_calendar_set() -> set[str]:
    from akshare.futures import cons as ak_cons

    return set(ak_cons.get_calendar())


def _official_exchange_kind(root: str) -> str | None:
    """
    Route to exchange-published daily panel APIs (AkShare) when the variety trades there.
    INE products are listed under ``shfe`` in akshare.cons but use ``get_ine_daily``.
    """
    from akshare.futures import cons as ak_cons

    r = root.upper()
    ine_roots = {"SC", "NR", "LU", "BC"}
    if r in ine_roots:
        return "ine"
    gfex = set(ak_cons.market_exchange_symbols.get("gfex", []))
    if r in gfex:
        return "gfex"
    shfe = set(ak_cons.market_exchange_symbols.get("shfe", []))
    if r in shfe and r not in ine_roots:
        return "shfe"
    return None


def _invoke_official_daily(ak: Any, *, exchange: str, date_str: str) -> pd.DataFrame:
    if exchange == "shfe":
        fn = getattr(ak, "get_shfe_daily", None)
    elif exchange == "ine":
        fn = getattr(ak, "get_ine_daily", None)
    elif exchange == "gfex":
        fn = getattr(ak, "get_gfex_daily", None)
    else:
        raise ValueError(f"unknown official exchange {exchange}")
    if fn is None:
        raise ValueError(f"AkShare daily API unavailable for {exchange}")
    try:
        out = fn(date_str)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as e:
        raise RuntimeError(f"official daily failed {exchange} {date_str}: {e}") from e
    if out is None:
        raise RuntimeError(f"official daily returned None ({exchange} {date_str})")
    return out


def _official_daily_series(
    ak: Any,
    *,
    exchange: str,
    contract_code: str,
    start_d: dt.date,
    end_d: dt.date,
) -> pd.DataFrame:
    """
    One HTTP request per trading day in [start_d, end_d]; 1s delay between requests (AkShare throttling).
    """
    cal = _trading_calendar_set()
    sym_key = contract_code.upper()
    rows: list[dict[str, Any]] = []
    d = start_d
    first_http = True
    while d <= end_d:
        ds = d.strftime("%Y%m%d")
        if ds not in cal:
            d += dt.timedelta(days=1)
            continue
        if not first_http:
            time.sleep(1.0)
        first_http = False
        df = _invoke_official_daily(ak, exchange=exchange, date_str=ds)
        if df.empty:
            d += dt.timedelta(days=1)
            continue
        if "symbol" not in df.columns:
            raise RuntimeError(f"official daily missing symbol column ({exchange} {ds})")
        m = df[df["symbol"].astype(str).str.upper() == sym_key]
        if m.empty:
            d += dt.timedelta(days=1)
            continue
        r = m.iloc[0]
        amt = r.get("turnover")
        rows.append(
            {
                "trade_date": d,
                "open": _to_float(r.get("open")),
                "high": _to_float(r.get("high")),
                "low": _to_float(r.get("low")),
                "close": _to_float(r.get("close")),
                "settle": _to_float(r.get("settle")),
                "volume": _to_float(r.get("volume")),
                "hold": _to_float(r.get("open_interest")),
                "amount": _to_float(amt),
            }
        )
        d += dt.timedelta(days=1)
    if not rows:
        return pd.DataFrame(
            columns=["trade_date", "open", "high", "low", "close", "settle", "volume", "hold", "amount"]
        )
    out = pd.DataFrame(rows)
    return out.sort_values("trade_date", ascending=True)


def _download_sina_full_history(ak: Any, contract_code: str) -> pd.DataFrame:
    raw = _fetch_futures_daily_sina_df(ak=ak, symbol=contract_code)
    return _normalize_futures_df(raw)


def _fetch_contract_series(
    ak: Any,
    *,
    root: str,
    contract_code: str,
    start_d: dt.date,
    end_d: dt.date,
    use_official_exchange: bool,
) -> tuple[pd.DataFrame, str]:
    kind = _official_exchange_kind(root)
    if use_official_exchange and kind is not None:
        df = _official_daily_series(
            ak,
            exchange=kind,
            contract_code=contract_code,
            start_d=start_d,
            end_d=end_d,
        )
        return df, f"official_{kind}"

    df = _download_sina_full_history(ak, contract_code)
    return df, "sina"


@dataclass(frozen=True)
class _ContractWork:
    contract_code: str
    start_d: dt.date
    end_d: dt.date


def _plan_contract_windows(
    db: Session,
    *,
    contracts: list[str],
    window_start: str,
    window_end: str,
    fetch_type: str,
) -> list[_ContractWork]:
    ui_fetch_type = str(fetch_type or "incremental").strip().lower()
    ws = _parse_yyyymmdd(window_start)
    we = _parse_yyyymmdd(window_end)
    out: list[_ContractWork] = []
    for sym in contracts:
        last_trade = get_futures_last_trade_date(db, code=sym, adjust="none")
        if last_trade is None:
            out.append(_ContractWork(sym, ws, we))
        else:
            if ui_fetch_type == "full":
                if ws <= we:
                    out.append(_ContractWork(sym, ws, we))
            else:
                start_use = max(ws, last_trade + dt.timedelta(days=1))
                if start_use <= we:
                    out.append(_ContractWork(sym, start_use, we))
    return out


def ingest_contracts_for_pool(
    db: Session,
    *,
    ak: Any,
    pool_code: str,
    main_fetch_type: str,
) -> str:
    pool = get_futures_pool_by_code(db, pool_code)
    if pool is None:
        return "pool not found"

    main_rng = get_futures_date_range(db, code=pool_code, adjust="none")
    if main_rng[0] is None or main_rng[1] is None:
        msg = "skip contracts: no local main continuous range"
        mark_futures_contract_pool_fetch(db, code=pool_code, status="skipped", message=msg)
        db.commit()
        return msg

    root = _symbol_root_from_main(pool_code)
    extend_days = int(pool.contract_extend_calendar_days or 366)

    end_raw = _parse_yyyymmdd(main_rng[1]) + dt.timedelta(days=extend_days)
    end_eff = _align_weekend_forward(end_raw)
    window_start = main_rng[0]
    window_end = end_eff.strftime("%Y%m%d")

    contracts = _contract_codes_in_window(
        root=root,
        main_start=main_rng[0],
        main_end=main_rng[1],
        extend_calendar_days=extend_days,
    )
    ui_fetch_type = str(main_fetch_type or "incremental").strip().lower()

    if ui_fetch_type == "full":
        for c in contracts:
            delete_futures_prices(db, code=c, adjust="none")

    planned = _plan_contract_windows(db, contracts=contracts, window_start=window_start, window_end=window_end, fetch_type=ui_fetch_type)
    if not planned:
        msg = f"no contract work (fetch_type={ui_fetch_type})"
        mark_futures_contract_pool_fetch(db, code=pool_code, status="success", message=msg)
        db.commit()
        return msg

    pid = int(pool.id)

    def _apply_df(
        sym: str, norm: pd.DataFrame, start_d: dt.date, end_d: dt.date, *, source: str
    ) -> tuple[str, int, str, str]:
        request_range = f"range={start_d.strftime('%Y%m%d')}~{end_d.strftime('%Y%m%d')}"
        if norm.empty:
            return sym, 0, "skipped", f"{request_range}"
        norm2 = norm[(norm["trade_date"] >= start_d) & (norm["trade_date"] <= end_d)].copy()
        if norm2.empty:
            return sym, 0, "success", f"{request_range}"
        actual_start = norm2["trade_date"].min().strftime("%Y%m%d")
        actual_end = norm2["trade_date"].max().strftime("%Y%m%d")
        actual_range = f"{actual_start}~{actual_end}"
        rows = [
            FuturesPriceRow(
                code=sym,
                trade_date=row.trade_date,
                open=_to_float(row.open),
                high=_to_float(row.high),
                low=_to_float(row.low),
                close=_to_float(row.close),
                settle=_to_float(row.settle),
                volume=_to_float(row.volume),
                amount=_to_float(getattr(row, "amount", None)),
                hold=_to_float(row.hold),
                source=source,
                adjust="none",
                pool_id=pid,
            )
            for row in norm2.itertuples(index=False)
        ]
        n = upsert_futures_prices(db, rows)
        return sym, int(n), "success", f"range={actual_range}; rows={len(rows)}"

    use_official = get_settings().futures_contract_use_official_exchange
    ok = 0
    for idx, w in enumerate(planned):
        if idx > 0:
            time.sleep(1.0)
        try:
            norm, src = _fetch_contract_series(
                ak,
                root=root,
                contract_code=w.contract_code,
                start_d=w.start_d,
                end_d=w.end_d,
                use_official_exchange=use_official,
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as e:
            range_txt = f"range={w.start_d.strftime('%Y%m%d')}~{w.end_d.strftime('%Y%m%d')}"
            err_msg = f"{range_txt}; {e}"
            last_td = get_futures_last_trade_date(db, code=w.contract_code, adjust="none")
            if last_td is None:
                delete_contract_fetch_status(db, pool_id=pid, contract_code=w.contract_code)
            else:
                record_contract_fetch_status(
                    db,
                    pool_id=pid,
                    contract_code=w.contract_code,
                    status="failed",
                    message=err_msg,
                    rows_upserted=0,
                )
            mark_futures_contract_pool_fetch(
                db,
                code=pool_code,
                status="failed",
                message=f"contract {w.contract_code}: {err_msg}",
            )
            db.commit()
            logger.warning("contract fetch aborted pool=%s contract=%s err=%s", pool_code, w.contract_code, e)
            raise ContractFetchAborted(str(e)) from e
        sym, n, st, msg = _apply_df(w.contract_code, norm, w.start_d, w.end_d, source=src)
        if st == "success":
            ok += 1
        last_td = get_futures_last_trade_date(db, code=sym, adjust="none")
        if last_td is None and n <= 0:
            delete_contract_fetch_status(db, pool_id=pid, contract_code=sym)
        else:
            record_contract_fetch_status(
                db,
                pool_id=pid,
                contract_code=sym,
                status=st,
                message=msg,
                rows_upserted=n,
            )
        db.commit()

    summary = (
        f"contracts={len(contracts)} planned={len(planned)} ok_status={ok} fetch_type={ui_fetch_type} "
        f"range={window_start}~{window_end} parallel=1 (AkShare serial)"
    )
    mark_futures_contract_pool_fetch(db, code=pool_code, status="success", message=summary)
    db.commit()
    return summary


def run_contract_fetch_job(pool_code: str, main_fetch_type: str, session_factory: Any) -> None:
    """Background entrypoint: separate Session, never raises to Starlette."""
    import akshare as ak

    db = session_factory()
    try:
        ingest_contracts_for_pool(db, ak=ak, pool_code=pool_code, main_fetch_type=main_fetch_type)
    except ContractFetchAborted:
        pass
    except Exception as e:
        db.rollback()
        mark_futures_contract_pool_fetch(db, code=pool_code, status="failed", message=str(e))
        db.commit()
    finally:
        db.close()


def run_contract_fetch_sequential_job(pool_codes: list[str], main_fetch_type: str, session_factory: Any) -> None:
    """
    Run contract ingestion for several pools **in order**. Stops after the first :class:`ContractFetchAborted`
    (failed contract in a pool), so later pools are not processed.
    """
    import akshare as ak

    for code in pool_codes:
        db = session_factory()
        try:
            ingest_contracts_for_pool(db, ak=ak, pool_code=code, main_fetch_type=main_fetch_type)
        except ContractFetchAborted:
            break
        except Exception as e:
            db.rollback()
            mark_futures_contract_pool_fetch(db, code=code, status="failed", message=str(e))
            db.commit()
            break
        finally:
            db.close()
