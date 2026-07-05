from __future__ import annotations

# pylint: disable=broad-exception-caught

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..data.akshare_tencent_fetcher import (
    FetchRequest as TencentFetchRequest,
    fetch_etf_daily_tencent,
)
from ..data.global_benchmark_defaults import (
    default_series_spec_by_code,
)
from ..data.sina_fetcher import (
    FetchRequest as SinaFetchRequest,
    fetch_sina_forex_day_kline_daily_close,
    fetch_sina_global_futures_day_kline_daily_close,
)
from ..data.stooq_fetcher import FetchRequest as StooqFetchRequest
from ..data.stooq_fetcher import fetch_stooq_daily_close
from ..data.yahoo_fetcher import FetchRequest as YahooFetchRequest
from ..data.yahoo_fetcher import fetch_yahoo_daily_close_with_alias
from ..db.global_benchmark_repo import (
    GlobalBenchmarkPriceRow,
    get_fallback_sources_for_pool_item,
    get_global_benchmark_pool_by_code,
    mark_global_benchmark_fetch_status,
    normalize_series_kind,
    update_global_benchmark_pool_data_range,
    upsert_global_benchmark_prices,
)
from ..settings import get_settings


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    symbol: str
    status: str
    sample_days: int
    span_days: int
    continuity: float
    latency_ms: int
    score: float | None
    error: str | None = None


@dataclass(frozen=True)
class GlobalBenchmarkIngestResult:
    code: str
    series_kind: str
    inserted_or_updated: int
    status: str
    message: str | None
    code_format: str
    final_provider: str | None
    final_symbol: str | None
    attempts: list[ProviderAttempt]


@dataclass(frozen=True)
class _SourceCandidate:
    provider: str
    symbol: str


def _parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(str(x), "%Y%m%d").date()


def detect_code_format(code: str, code_format_hint: str | None = None) -> str:
    hint = str(code_format_hint or "").strip().lower()
    if hint and hint != "auto":
        return hint
    c = str(code or "").strip()
    u = c.upper()
    if not c:
        return "unknown"
    if u.startswith("MSCI"):
        return "msci"
    if c.isdigit() and len(c) == 6:
        return "cn_6"
    if u in {"HSI", "HSCEI", "HSTECH", "HSHCI", "HSCCI"}:
        return "hk_index"
    if c.startswith("^") or "." in c:
        return "yahoo"
    if c.isalpha() and len(c) <= 10:
        return "symbol"
    return "unknown"


def resolve_provider_candidates(
    *,
    code_format: str,
    provider_hint: str | None = None,
) -> list[str]:
    hint = str(provider_hint or "").strip().lower()
    if hint and hint != "auto":
        return [hint]
    base_map: dict[str, list[str]] = {
        "cn_6": ["tencent", "yahoo", "stooq"],
        "hk_index": ["yahoo", "stooq", "tencent"],
        "yahoo": ["yahoo", "stooq", "tencent"],
        "msci": ["sina_global", "yahoo", "stooq"],
        "symbol": ["yahoo", "stooq", "sina_global", "tencent"],
        "unknown": ["yahoo", "stooq", "tencent", "sina_global"],
    }
    out = list(base_map.get(str(code_format), base_map["unknown"]))
    dedup: list[str] = []
    seen: set[str] = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def _price_rows_from_close_df(
    code: str,
    series_kind: str,
    source: str,
    df: pd.DataFrame,
) -> list[GlobalBenchmarkPriceRow]:
    if df is None or df.empty:
        return []
    if "date" not in df.columns or "close" not in df.columns:
        return []
    out: list[GlobalBenchmarkPriceRow] = []
    for _, r in df.iterrows():
        d = pd.to_datetime(r.get("date"), errors="coerce")
        if pd.isna(d):
            continue
        c = pd.to_numeric(r.get("close"), errors="coerce")
        if pd.isna(c):
            continue
        px = float(c)
        out.append(
            GlobalBenchmarkPriceRow(
                code=code,
                series_kind=series_kind,
                trade_date=d.date(),
                open=px,
                high=px,
                low=px,
                close=px,
                volume=None,
                amount=None,
                source=source,
                adjust="none",
            )
        )
    dedup = {x.trade_date: x for x in out}
    return [dedup[d] for d in sorted(dedup)]


_HK_YAHOO_ALIAS: dict[str, str] = {
    "HSI": "^HSI",
    "HSCEI": "^HSCE",
    "HSTECH": "^HSTECH",
    "HSHCI": "^HSHCI",
    "HSCCI": "^HSCCI",
}


def _to_yahoo_symbol(code: str, *, code_format: str) -> str:
    c = str(code or "").strip()
    if not c:
        return c
    u = c.upper()
    if str(code_format) == "hk_index":
        if u in _HK_YAHOO_ALIAS:
            return _HK_YAHOO_ALIAS[u]
        if c.startswith("^"):
            return c
        return f"^{u}"
    if str(code_format) == "cn_6":
        if c.isdigit() and len(c) == 6:
            suffix = ".SS" if c.startswith(("5", "6", "9")) else ".SZ"
            return f"{c}{suffix}"
    return c


def _tencent_symbol_candidates(code: str) -> list[str]:
    c = str(code or "").strip()
    if not c:
        return []
    cand = [c]
    if c.isdigit() and len(c) == 6:
        cand.extend([f"sh{c}", f"sz{c}"])
    if c.lower().startswith(("sh", "sz")) and len(c) == 8:
        raw6 = c[2:]
        if raw6.isdigit():
            cand.extend([raw6, f"sh{raw6}", f"sz{raw6}"])
    out: list[str] = []
    seen: set[str] = set()
    for x in cand:
        k = str(x).strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(str(x).strip())
    return out


def _fetch_rows_by_source_candidate(
    *,
    ak: Any,
    code: str,
    series_kind: str,
    candidate: _SourceCandidate,
    start: str,
    end: str,
) -> tuple[list[GlobalBenchmarkPriceRow], str | None]:
    provider = str(candidate.provider or "").strip().lower()
    symbol = str(candidate.symbol or "").strip()
    if not provider or not symbol:
        return [], "empty_provider_or_symbol"
    if provider == "tencent":
        errs: list[str] = []
        for sym in _tencent_symbol_candidates(symbol):
            try:
                rows = fetch_etf_daily_tencent(
                    ak,
                    TencentFetchRequest(
                        code=sym,
                        start_date=start,
                        end_date=end,
                        adjust="none",
                    ),
                )
            except Exception as e:
                errs.append(f"{sym}:{e}")
                continue
            if not rows:
                errs.append(f"{sym}:empty")
                continue
            out = [
                GlobalBenchmarkPriceRow(
                    code=code,
                    series_kind=series_kind,
                    trade_date=r.trade_date,
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                    amount=r.amount,
                    source="tencent",
                    adjust="none",
                )
                for r in rows
            ]
            return out, None
        return [], "; ".join(errs) if errs else "empty"
    if provider == "yahoo":
        try:
            df, meta = fetch_yahoo_daily_close_with_alias(
                YahooFetchRequest(symbol=symbol, start_date=start, end_date=end),
                aliases=None,
            )
        except Exception as e:
            return [], str(e)
        rows = _price_rows_from_close_df(code, series_kind, "yahoo", df)
        if rows:
            return rows, None
        return [], str((meta or {}).get("error") or "empty")
    if provider == "stooq":
        try:
            df, meta = fetch_stooq_daily_close(
                StooqFetchRequest(symbol=symbol, start_date=start, end_date=end)
            )
        except Exception as e:
            return [], str(e)
        rows = _price_rows_from_close_df(code, series_kind, "stooq", df)
        if rows:
            return rows, None
        return [], str((meta or {}).get("error") or "empty")
    if provider == "sina_global":
        try:
            df, meta = fetch_sina_global_futures_day_kline_daily_close(
                SinaFetchRequest(symbol=symbol, start_date=start, end_date=end)
            )
        except Exception as e:
            return [], str(e)
        rows = _price_rows_from_close_df(code, series_kind, "sina_global", df)
        if rows:
            return rows, None
        return [], str((meta or {}).get("error") or "empty")
    if provider == "sina":
        try:
            df, meta = fetch_sina_forex_day_kline_daily_close(
                SinaFetchRequest(symbol=symbol, start_date=start, end_date=end)
            )
        except Exception as e:
            return [], str(e)
        rows = _price_rows_from_close_df(code, series_kind, "sina", df)
        if rows:
            return rows, None
        return [], str((meta or {}).get("error") or "empty")
    return [], f"unsupported provider={provider}"


def _metrics_for_rows(rows: list[GlobalBenchmarkPriceRow]) -> tuple[int, int, float]:
    if not rows:
        return (0, 0, 0.0)
    dates = sorted({x.trade_date for x in rows})
    n = len(dates)
    span = (dates[-1] - dates[0]).days + 1 if n > 0 else 0
    continuity = float(n / span) if span > 0 else 0.0
    return (n, span, continuity)


def _score_attempt(*, sample_days: int, continuity: float, latency_ms: int) -> float:
    return float(sample_days + continuity * 120.0 - float(latency_ms) / 200.0)


def _build_auto_candidates(code: str, *, code_format: str) -> list[_SourceCandidate]:
    out: list[_SourceCandidate] = []
    yahoo_symbol = _to_yahoo_symbol(code, code_format=code_format)
    for provider in resolve_provider_candidates(
        code_format=code_format, provider_hint="auto"
    ):
        symbol = code
        if provider in {"yahoo", "stooq"}:
            symbol = yahoo_symbol
        out.append(_SourceCandidate(provider=provider, symbol=symbol))
    return out


def _source_candidates_for_pool_item(pool_item) -> list[_SourceCandidate]:
    # 1) locked source first
    if bool(getattr(pool_item, "source_locked", False)):
        provider = str(getattr(pool_item, "provider_hint", "") or "").strip().lower()
        symbol = str(getattr(pool_item, "provider_symbol", "") or "").strip()
        if provider and symbol:
            out = [_SourceCandidate(provider=provider, symbol=symbol)]
            for x in get_fallback_sources_for_pool_item(pool_item):
                out.append(
                    _SourceCandidate(
                        provider=str(x.get("provider") or "").strip(),
                        symbol=str(x.get("symbol") or "").strip(),
                    )
                )
            return [x for x in out if x.provider and x.symbol]

    # 2) default spec candidates
    spec = default_series_spec_by_code(
        str(pool_item.code), str(getattr(pool_item, "series_kind", "price"))
    )
    if spec is not None and spec.candidates:
        return [
            _SourceCandidate(provider=x.provider, symbol=x.symbol)
            for x in spec.candidates
            if x.provider and x.symbol
        ]

    # 3) fallback to auto by code_format
    code_format = detect_code_format(
        str(pool_item.code), str(getattr(pool_item, "code_format", "") or "")
    )
    return _build_auto_candidates(str(pool_item.code), code_format=code_format)


def ingest_one_global_benchmark_series(
    db,
    *,
    ak: Any,
    code: str,
    series_kind: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> GlobalBenchmarkIngestResult:
    kind = normalize_series_kind(series_kind)
    pool = get_global_benchmark_pool_by_code(db, code, series_kind=kind)
    if pool is None:
        raise ValueError(f"global benchmark {code} ({kind}) not found in pool")
    settings = get_settings()
    start = str(start_date or pool.start_date or settings.default_start_date)
    end = str(end_date or pool.end_date or settings.default_end_date)
    _ = _parse_yyyymmdd(start)
    _ = _parse_yyyymmdd(end)

    code_format = detect_code_format(str(code), str(pool.code_format or ""))
    candidates = _source_candidates_for_pool_item(pool)
    attempts: list[ProviderAttempt] = []
    success_rows_by_source: dict[tuple[str, str], list[GlobalBenchmarkPriceRow]] = {}
    for c in candidates:
        t0 = time.monotonic()
        rows, err = _fetch_rows_by_source_candidate(
            ak=ak,
            code=code,
            series_kind=kind,
            candidate=c,
            start=start,
            end=end,
        )
        latency_ms = int((time.monotonic() - t0) * 1000.0)
        sample_days, span_days, continuity = _metrics_for_rows(rows)
        if sample_days > 0:
            score = _score_attempt(
                sample_days=sample_days,
                continuity=continuity,
                latency_ms=latency_ms,
            )
            success_rows_by_source[(c.provider, c.symbol)] = rows
            attempts.append(
                ProviderAttempt(
                    provider=c.provider,
                    symbol=c.symbol,
                    status="success",
                    sample_days=sample_days,
                    span_days=span_days,
                    continuity=continuity,
                    latency_ms=latency_ms,
                    score=score,
                    error=None,
                )
            )
        else:
            attempts.append(
                ProviderAttempt(
                    provider=c.provider,
                    symbol=c.symbol,
                    status="failed",
                    sample_days=0,
                    span_days=0,
                    continuity=0.0,
                    latency_ms=latency_ms,
                    score=None,
                    error=err or "empty",
                )
            )

    successes = [x for x in attempts if x.status == "success"]
    if not successes:
        msg = "all providers failed: " + ", ".join(
            f"{x.provider}/{x.symbol}:{x.error}" for x in attempts
        )
        mark_global_benchmark_fetch_status(
            db,
            code=code,
            series_kind=kind,
            status="failed",
            message=msg,
        )
        db.commit()
        return GlobalBenchmarkIngestResult(
            code=code,
            series_kind=kind,
            inserted_or_updated=0,
            status="failed",
            message=msg,
            code_format=code_format,
            final_provider=None,
            final_symbol=None,
            attempts=attempts,
        )
    best = sorted(successes, key=lambda x: float(x.score or -1e18), reverse=True)[0]
    final_rows = success_rows_by_source.get((best.provider, best.symbol), [])
    n = upsert_global_benchmark_prices(db, final_rows)
    _ = update_global_benchmark_pool_data_range(
        db, code=code, series_kind=kind, adjust="none"
    )
    # solidify discovered source
    pool.provider_hint = str(best.provider)
    pool.provider_symbol = str(best.symbol)
    pool.source_locked = True
    msg = (
        f"rows={len(final_rows)} upserted={n} final_provider={best.provider} "
        f"symbol={best.symbol} score={best.score:.3f} attempts={len(attempts)}"
    )
    mark_global_benchmark_fetch_status(
        db,
        code=code,
        series_kind=kind,
        status="success",
        message=msg,
    )
    db.commit()
    return GlobalBenchmarkIngestResult(
        code=code,
        series_kind=kind,
        inserted_or_updated=int(n),
        status="success",
        message=msg,
        code_format=code_format,
        final_provider=best.provider,
        final_symbol=best.symbol,
        attempts=attempts,
    )


def ingest_one_global_benchmark(
    db,
    *,
    ak: Any,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> GlobalBenchmarkIngestResult:
    # Backward-compatible helper: fetch price series by default.
    return ingest_one_global_benchmark_series(
        db,
        ak=ak,
        code=code,
        series_kind="price",
        start_date=start_date,
        end_date=end_date,
    )
