from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Iterable

# pylint: disable=import-error
# Allow running from repo root without editable install.
sys.path.insert(0, "src")

from etf_momentum.db.futures_repo import list_futures_prices  # noqa: E402
from etf_momentum.db.session import make_engine, make_session_factory  # noqa: E402
from etf_momentum.db.repo import list_prices  # noqa: E402
from etf_momentum.settings import get_settings  # noqa: E402


def _parse_date_yyyymmdd(value: str) -> dt.date:
    return dt.datetime.strptime(str(value), "%Y%m%d").date()


def _parse_codes(raw_codes: str) -> list[str]:
    out = [str(x).strip().upper() for x in str(raw_codes).split(",") if str(x).strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for code in out:
        if code in seen:
            continue
        seen.add(code)
        deduped.append(code)
    if not deduped:
        raise ValueError("codes must not be empty")
    return deduped


def _dates_and_closes(rows: Iterable[object]) -> tuple[list[dt.date], list[float]]:
    dates: list[dt.date] = []
    closes: list[float] = []
    for row in rows:
        d = getattr(row, "trade_date", None)
        c = getattr(row, "close", None)
        if d is None or c is None:
            continue
        dates.append(d)
        closes.append(float(c))
    return dates, closes


def _max_abs_return(closes: list[float]) -> float | None:
    if len(closes) < 2:
        return None
    mx = 0.0
    prev = closes[0]
    for cur in closes[1:]:
        if prev == 0:
            prev = cur
            continue
        r = abs((cur / prev) - 1.0)
        if r > mx:
            mx = r
        prev = cur
    return mx


def _format_date(value: dt.date | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y%m%d")


def _run_checks(
    *,
    asset_domain: str,
    codes: list[str],
    start: dt.date,
    end: dt.date,
    min_points: int,
    max_missing_ratio: float,
    abs_return_threshold: float,
    limit: int,
) -> dict:
    settings = get_settings()
    engine = make_engine(db_url=settings.db_url)
    sf = make_session_factory(engine)

    errors: list[str] = []
    warnings: list[str] = []
    reports: list[dict] = []
    union_dates: set[dt.date] = set()
    by_code_alignment_dates: dict[str, set[dt.date]] = {}

    with sf() as db:
        for code in codes:
            if asset_domain == "etf":
                adjusts = ["none", "qfq", "hfq"]
            else:
                adjusts = ["none"]

            series: dict[str, dict] = {}
            alignment_dates: set[dt.date] = set()
            for adj in adjusts:
                if asset_domain == "etf":
                    rows = list_prices(
                        db,
                        code=code,
                        adjust=adj,
                        start_date=start,
                        end_date=end,
                        limit=limit,
                    )
                else:
                    rows = list_futures_prices(
                        db,
                        code=code,
                        adjust=adj,
                        start_date=start,
                        end_date=end,
                        limit=limit,
                    )
                dates, closes = _dates_and_closes(rows)
                points = len(dates)
                max_abs_ret = _max_abs_return(closes)
                item = {
                    "points": points,
                    "start": _format_date(dates[0] if dates else None),
                    "end": _format_date(dates[-1] if dates else None),
                    "max_abs_return": max_abs_ret,
                }
                series[adj] = item

                if points < min_points:
                    errors.append(
                        f"{asset_domain}:{code}:{adj} points={points} < min_points={min_points}"
                    )
                if max_abs_ret is not None and max_abs_ret > abs_return_threshold:
                    warnings.append(
                        f"{asset_domain}:{code}:{adj} max_abs_return={max_abs_ret:.4f} > threshold={abs_return_threshold:.4f}"
                    )
                if adj == "none":
                    alignment_dates = set(dates)

            if asset_domain == "etf":
                for required_adj in ("none", "qfq", "hfq"):
                    if int(series.get(required_adj, {}).get("points", 0)) <= 0:
                        errors.append(
                            f"etf:{code} missing required adjust={required_adj}"
                        )

            by_code_alignment_dates[code] = alignment_dates
            union_dates.update(alignment_dates)
            reports.append({"code": code, "series": series})

    union_n = len(union_dates)
    if union_n == 0:
        errors.append("no data in requested window for any symbol")

    for report in reports:
        code = str(report["code"])
        dset = by_code_alignment_dates.get(code, set())
        present_n = len(dset)
        missing_ratio = 0.0 if union_n <= 0 else (1.0 - (present_n / union_n))
        report["alignment"] = {
            "union_points": union_n,
            "present_points": present_n,
            "missing_ratio": missing_ratio,
        }
        if missing_ratio > max_missing_ratio:
            errors.append(
                f"{asset_domain}:{code} missing_ratio={missing_ratio:.4f} > max_missing_ratio={max_missing_ratio:.4f}"
            )

    return {
        "ok": len(errors) == 0,
        "meta": {
            "asset_domain": asset_domain,
            "codes": codes,
            "start": _format_date(start),
            "end": _format_date(end),
            "min_points": min_points,
            "max_missing_ratio": max_missing_ratio,
            "abs_return_threshold": abs_return_threshold,
            "union_points": union_n,
        },
        "reports": reports,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-check trend backtest readiness for configured symbol groups."
    )
    ap.add_argument("--asset-domain", choices=["etf", "futures"], default="etf")
    ap.add_argument("--codes", required=True, help="Comma-separated symbol list")
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    ap.add_argument("--min-points", type=int, default=252)
    ap.add_argument("--max-missing-ratio", type=float, default=0.10)
    ap.add_argument("--abs-return-threshold", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=30000)
    args = ap.parse_args(argv)

    codes = _parse_codes(args.codes)
    start = _parse_date_yyyymmdd(args.start)
    end = _parse_date_yyyymmdd(args.end)
    if start > end:
        raise SystemExit("start must be <= end")
    if args.min_points < 2:
        raise SystemExit("min-points must be >= 2")
    if args.max_missing_ratio < 0 or args.max_missing_ratio > 1:
        raise SystemExit("max-missing-ratio must be in [0,1]")
    if args.abs_return_threshold <= 0:
        raise SystemExit("abs-return-threshold must be > 0")

    out = _run_checks(
        asset_domain=args.asset_domain,
        codes=codes,
        start=start,
        end=end,
        min_points=int(args.min_points),
        max_missing_ratio=float(args.max_missing_ratio),
        abs_return_threshold=float(args.abs_return_threshold),
        limit=int(args.limit),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if bool(out.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
