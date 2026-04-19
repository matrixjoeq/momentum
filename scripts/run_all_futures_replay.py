from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from etf_momentum.db.session import make_engine
from etf_momentum.settings import get_settings


def main() -> int:
    engine = make_engine(db_url=get_settings().db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT code, MIN(trade_date) AS start_date, MAX(trade_date) AS end_date
                FROM futures_prices
                WHERE adjust='none'
                GROUP BY code
                ORDER BY code
                """
            )
        ).mappings()
        all_rows = list(rows)

    base = Path("outputs/futures_replay_all")
    base.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for r in all_rows:
        code = str(r["code"])
        if not code.endswith("0"):
            summary.append(
                {"code": code, "status": "skipped", "reason": "not_main0_symbol"}
            )
            continue

        underlying = code[:-1]
        start = str(r["start_date"]).replace("-", "")
        end = str(r["end_date"]).replace("-", "")
        outdir = base / code
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "etf_momentum.scripts.futures_continuous_replay",
            "--underlying",
            underlying,
            "--main-symbol",
            code,
            "--start-date",
            start,
            "--end-date",
            end,
            "--output-dir",
            str(outdir),
        ]

        print(f"[RUN] {code} {start}-{end}", flush=True)
        try:
            cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
            (outdir / "run.stdout.log").write_text(cp.stdout or "", encoding="utf-8")
            (outdir / "run.stderr.log").write_text(cp.stderr or "", encoding="utf-8")
            summary.append({"code": code, "status": "ok", "start": start, "end": end})
            print(f"[OK] {code}", flush=True)
        except subprocess.CalledProcessError as e:
            (outdir / "run.stdout.log").write_text(e.stdout or "", encoding="utf-8")
            (outdir / "run.stderr.log").write_text(e.stderr or "", encoding="utf-8")
            summary.append(
                {
                    "code": code,
                    "status": "failed",
                    "start": start,
                    "end": end,
                    "returncode": e.returncode,
                }
            )
            print(f"[FAIL] {code} rc={e.returncode}", flush=True)

    (base / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ok = sum(1 for x in summary if x.get("status") == "ok")
    fail = sum(1 for x in summary if x.get("status") == "failed")
    print(f"[DONE] total={len(summary)} ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
