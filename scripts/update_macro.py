from __future__ import annotations

import argparse
import datetime as dt
import sys

# pylint: disable=import-error
# Allow running from repo root without editable install.
sys.path.insert(0, "src")

from etf_momentum.data.macro_ingestion import MACRO_SERIES, ingest_macro_series  # noqa: E402
from etf_momentum.db.init_db import init_db  # noqa: E402
from etf_momentum.db.session import make_engine, make_session_factory  # noqa: E402
from etf_momentum.settings import get_settings  # noqa: E402


def _today_yyyymmdd() -> str:
    return dt.datetime.now().strftime("%Y%m%d")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Update macro series for Step 1 (global gold / US yields / DXY).")
    ap.add_argument("--start", default="1985101", help="YYYYMMDD (default 1985101)")
    ap.add_argument("--end", default=_today_yyyymmdd(), help="YYYYMMDD (default today)")
    ap.add_argument(
        "--series",
        action="append",
        default=[],
        help="Optional series_id(s) to update. Can be repeated or comma-separated, e.g. --series DGS10 --series DINIW,GC_FUT",
    )
    args = ap.parse_args(argv)

    settings = get_settings()
    engine = make_engine(db_url=settings.db_url)
    init_db(engine)
    sf = make_session_factory(engine)

    # series filter (default: all)
    raw = ",".join([str(x) for x in (args.series or [])])
    chosen = [s.strip().upper() for s in raw.split(",") if s.strip()] if raw.strip() else []
    if chosen:
        by_id = {s.series_id.strip().upper(): s for s in MACRO_SERIES}
        missing = [sid for sid in chosen if sid not in by_id]
        if missing:
            raise SystemExit(f"Unknown --series: {missing}. Known: {sorted(by_id)}")
        series_specs = [by_id[sid] for sid in chosen]
    else:
        series_specs = list(MACRO_SERIES)

    ok = True
    with sf() as db:
        for spec in series_specs:
            res = ingest_macro_series(db, spec=spec, start=str(args.start), end=str(args.end))
            print(res)
            ok = ok and bool(res.get("ok"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

