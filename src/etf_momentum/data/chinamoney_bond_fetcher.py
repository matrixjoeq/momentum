from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# 中国货币网 (China Foreign Exchange Trade System) 收盘收益率曲线历史数据接口.
# 该接口覆盖国债与各信用评级 (AAA/AA+/AA/...) 的到期收益率曲线, 并提供 2 年关键期限点,
# 因此可以满足 "同久期国债 + 信用债" 的需求. akshare 自带的 ``bond_china_close_return``
# 由于把 pageSize/pageNum 写死 (只能取到约一天的整条曲线), 无法按日期区间分页, 故这里
# 直接访问原始接口并自行分页 (见 AGENTS.md "优先使用成熟库, 无法满足语义时自建并说明差距").
_HIS_URL = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/ClsYldCurvHis"
_CURVE_MAP_URL = (
    "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/ClsYldCurvCurvGO"
)
_REGISTER_PAGE = "https://www.chinamoney.com.cn/chinese/bkcurvclosedyhis/?bondType=CYCC000&reference=1"
_APPLY_URL = "https://www.chinamoney.com.cn/dqs/rest/cm-u-rbt/apply"
_SESSION_USER_URL = "https://www.chinamoney.com.cn/lss/rest/cm-s-account/getSessionUser"
# 该接口对 pageSize 有上限, 实测 >50 会返回 403; 使用 50 并翻页.
_PAGE_SIZE = 50
# 单次查询区间需 <= 1 个月.
_CHUNK_DAYS = 28
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class FetchRequest:
    symbol_label: str  # ChinaMoney 曲线中文名, e.g. "国债" / "中短期票据(AAA)"
    tenor_years: float  # 关键期限点 (年), e.g. 2 / 10 / 30
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _register_session(session: requests.Session) -> None:
    """Whitelist this IP for ~24h so the history endpoint returns JSON.

    Mirrors akshare's ``bond_china_money.__bond_register_service`` flow; without
    it the endpoint answers with a 403 HTML page instead of JSON.
    """
    session.get(_REGISTER_PAGE, timeout=30)
    cookies = "; ".join(f"{k}={v}" for k, v in session.cookies.get_dict().items())
    session.post(
        _APPLY_URL,
        data={"key": "TThwSjc2NWkzV0VSOVRzOA=="},
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Cookie": cookies,
            "Origin": "https://www.chinamoney.com.cn",
            "Referer": _REGISTER_PAGE,
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=30,
    )
    cookies = "; ".join(f"{k}={v}" for k, v in session.cookies.get_dict().items())
    session.post(
        _SESSION_USER_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Cookie": cookies,
            "Origin": "https://www.chinamoney.com.cn",
            "Referer": _REGISTER_PAGE,
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=30,
    )


def _resolve_curve_code(session: requests.Session, symbol_label: str) -> str | None:
    """Resolve a Chinese curve label to its ChinaMoney bondType code (e.g. CYCC000)."""
    try:
        r = session.get(_CURVE_MAP_URL, timeout=30)
        records = r.json().get("records") or []
    except (requests.RequestException, ValueError):
        _register_session(session)
        try:
            r = session.get(_CURVE_MAP_URL, timeout=30)
            records = r.json().get("records") or []
        except (requests.RequestException, ValueError):
            return None
    for rec in records:
        if str(rec.get("cnLabel")) == symbol_label:
            return str(rec.get("value"))
    return None


def _iter_chunks(start: dt.date, end: dt.date):
    """Yield (chunk_start, chunk_end) newest-first, each <= _CHUNK_DAYS wide."""
    cur_end = end
    while cur_end >= start:
        cur_start = max(start, cur_end - dt.timedelta(days=_CHUNK_DAYS - 1))
        yield cur_start, cur_end
        cur_end = cur_start - dt.timedelta(days=1)


def _fetch_curve_chunk(
    session: requests.Session,
    *,
    code: str,
    chunk_start: dt.date,
    chunk_end: dt.date,
) -> list[dict[str, Any]]:
    """Fetch all paginated records for one curve/date-chunk (auto re-register on block)."""
    out: list[dict[str, Any]] = []
    page_num = 1
    registered = False
    while True:
        params = {
            "lang": "CN",
            "reference": "1,2,3",
            "bondType": code,
            "startDate": chunk_start.strftime("%Y-%m-%d"),
            "endDate": chunk_end.strftime("%Y-%m-%d"),
            "termId": "1",
            "pageNum": str(page_num),
            "pageSize": str(_PAGE_SIZE),
        }
        try:
            r = session.get(_HIS_URL, params=params, timeout=30)
            payload = r.json()
        except (requests.RequestException, ValueError):
            if registered:
                raise
            _register_session(session)
            registered = True
            continue
        records = payload.get("records") or []
        out.extend(records)
        data = payload.get("data") or {}
        try:
            page_total = int(data.get("pageTotal") or 0)
        except (TypeError, ValueError):
            page_total = 0
        if not records or page_num >= page_total:
            break
        page_num += 1
        time.sleep(0.3)
    return out


def fetch_chinamoney_bond_yield(
    req: FetchRequest,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a single China bond yield series (one curve + one tenor).

    Returns a dataframe with columns ``date`` and ``close`` (到期收益率, %).

    Note: the ChinaMoney history endpoint only serves a rolling recent window,
    so deep history is not available; older chunks come back empty and iteration
    stops once the window boundary is reached.
    """
    meta: dict[str, Any] = {
        "provider": "chinamoney_bond",
        "symbol": req.symbol_label,
        "tenor_years": req.tenor_years,
    }
    start = _parse_yyyymmdd(req.start_date)
    end = _parse_yyyymmdd(req.end_date)
    if end < start:
        return pd.DataFrame(), {**meta, "error": "end_before_start"}

    session = _new_session()
    code = _resolve_curve_code(session, req.symbol_label)
    if not code:
        return pd.DataFrame(), {**meta, "error": "unknown_curve_label"}
    meta["code"] = code

    rows: list[dict[str, Any]] = []
    empty_chunks = 0
    for chunk_start, chunk_end in _iter_chunks(start, end):
        try:
            records = _fetch_curve_chunk(
                session, code=code, chunk_start=chunk_start, chunk_end=chunk_end
            )
        except requests.RequestException as e:
            logger.warning("chinamoney fetch failed code=%s err=%s", code, e)
            return pd.DataFrame(), {**meta, "error": str(e)}

        chunk_rows = 0
        for rec in records:
            term = pd.to_numeric(rec.get("yearTermStr"), errors="coerce")
            if pd.isna(term) or abs(float(term) - float(req.tenor_years)) > 1e-9:
                continue
            d = pd.to_datetime(rec.get("newDateValueCN"), errors="coerce")
            if pd.isna(d):
                continue
            val = pd.to_numeric(rec.get("maturityYieldStr"), errors="coerce")
            rows.append(
                {
                    "date": d.date(),
                    "close": float(val) if pd.notna(val) else None,
                }
            )
            chunk_rows += 1

        # The endpoint serves a contiguous recent window; stop scanning further
        # back once we hit empty chunks (avoids thousands of empty requests when
        # start is far in the past).
        if not records:
            empty_chunks += 1
            if empty_chunks >= 2:
                break
        else:
            empty_chunks = 0

    if not rows:
        return pd.DataFrame(), {**meta, "error": "empty_fetch"}

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {**meta, "error": "empty_in_range"}
    return df[["date", "close"]], meta
