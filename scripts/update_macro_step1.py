from __future__ import annotations

import argparse
import datetime as dt
import sys

# pylint: disable=import-error
# Allow running from repo root without editable install.
sys.path.insert(0, "src")

from etf_momentum.data.macro_ingestion import DEFAULT_STEP1_SERIES, ingest_macro_series  # noqa: E402
from etf_momentum.db.init_db import init_db  # noqa: E402
from etf_momentum.db.session import make_engine, make_session_factory  # noqa: E402
from etf_momentum.settings import get_settings  # noqa: E402


def _today_yyyymmdd() -> str:
    return dt.datetime.now().strftime("%Y%m%d")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Update macro series for Step 1 (global gold / US yields / DXY).")
    ap.add_argument("--start", default="1985101", help="YYYYMMDD (default 1985101)")
    ap.add_argument("--end", default=_today_yyyymmdd(), help="YYYYMMDD (default today)")
    args = ap.parse_args(argv)

    settings = get_settings()
    engine = make_engine(db_url=settings.db_url)
    init_db(engine)
    sf = make_session_factory(engine)

    ok = True
    with sf() as db:
        for spec in DEFAULT_STEP1_SERIES:
            res = ingest_macro_series(db, spec=spec, start=str(args.start), end=str(args.end))
            print(res)
            ok = ok and bool(res.get("ok"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

