from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# pylint: disable=import-error
# Allow running from repo root without editable install.
sys.path.insert(0, "src")

from etf_momentum.db.base import Base  # noqa: E402
from etf_momentum.db.init_db import init_db  # noqa: E402
from etf_momentum.settings import get_settings  # noqa: E402


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _rowcount(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar_one()
        return int(n or 0)


def _sqlite_engine(sqlite_path: str) -> Engine:
    # file-based sqlite for migration input
    return create_engine(f"sqlite+pysqlite:///{sqlite_path}", future=True)


def _mysql_engine() -> Engine:
    settings = get_settings()
    if not settings.db_url:
        raise ValueError("missing MySQL db_url (check data/.env.local or MOMENTUM_DB_URL)")
    # Use SQLAlchemy engine settings consistent with runtime
    from etf_momentum.db.session import make_engine  # noqa: E402

    return make_engine(db_url=settings.db_url)


def _wipe_mysql_tables(mysql: Engine, tables_in_order: list[str]) -> None:
    with mysql.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in reversed(tables_in_order):
            conn.execute(text(f"DELETE FROM `{t}`"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))


def _reset_autoinc(mysql: Engine, tables: list[str]) -> None:
    insp = inspect(mysql)
    with mysql.begin() as conn:
        for t in tables:
            cols = {c["name"]: c for c in insp.get_columns(t)}
            if "id" not in cols:
                continue
            if not bool(cols["id"].get("autoincrement", True)):
                # best-effort; MySQL reflects this inconsistently
                pass
            max_id = conn.execute(text(f"SELECT MAX(id) FROM `{t}`")).scalar_one()
            nxt = int(max_id or 0) + 1
            conn.execute(text(f"ALTER TABLE `{t}` AUTO_INCREMENT = {nxt}"))


def _varchar_max_len(col_type: Any) -> int | None:
    """
    Best-effort detect VARCHAR length from SQLAlchemy reflected type.
    """
    try:
        ln = getattr(col_type, "length", None)
        return int(ln) if ln is not None else None
    except Exception:
        return None


def _clip_str(x: Any, max_len: int) -> Any:
    if x is None:
        return None
    if not isinstance(x, str):
        return x
    if len(x) <= max_len:
        return x
    suffix = "...(truncated)"
    keep = max(0, max_len - len(suffix))
    return (x[:keep] + suffix)[:max_len]


def _prepare_row(row: dict[str, Any], *, col_maxlens: dict[str, int]) -> dict[str, Any]:
    """
    Prepare row values for MySQL insert:
    - truncate overlong VARCHAR columns to avoid DataError(1406)
    """
    if not col_maxlens:
        return row
    out: dict[str, Any] = dict(row)
    for k, max_len in col_maxlens.items():
        if k in out:
            out[k] = _clip_str(out[k], max_len)
    return out


def migrate_sqlite_to_mysql(*, sqlite_path: str, force: bool, chunk_size: int) -> None:
    print(f"[{_utc_now()}] source_sqlite={sqlite_path}")
    mysql = _mysql_engine()
    sqlite = _sqlite_engine(sqlite_path)

    # Ensure target schema exists
    init_db(mysql)

    # Determine table order by SQLAlchemy metadata (respects FK dependencies)
    tables = [t.name for t in Base.metadata.sorted_tables]

    # Safety checks
    missing = []
    sinsp = inspect(sqlite)
    sqlite_tables = set(sinsp.get_table_names())
    for t in tables:
        if t not in sqlite_tables:
            missing.append(t)
    if missing:
        print(f"[{_utc_now()}] WARNING: sqlite missing tables: {missing} (will skip)")

    if not force:
        nonempty = [t for t in tables if _rowcount(mysql, t) > 0]
        if nonempty:
            raise SystemExit(
                "target MySQL is not empty. Re-run with --force to wipe target tables first. "
                f"nonempty_tables={nonempty}"
            )
    else:
        print(f"[{_utc_now()}] wiping target MySQL tables ...")
        _wipe_mysql_tables(mysql, tables)

    # Copy data table by table
    minsp = inspect(mysql)
    for t in tables:
        if t not in sqlite_tables:
            continue
        mysql_cols_info = minsp.get_columns(t)
        mysql_cols = [c["name"] for c in mysql_cols_info]
        col_maxlens: dict[str, int] = {}
        for c in mysql_cols_info:
            name = c.get("name")
            if not name:
                continue
            ml = _varchar_max_len(c.get("type"))
            if ml is not None and ml > 0:
                col_maxlens[str(name)] = int(ml)
        sqlite_cols = [c["name"] for c in sinsp.get_columns(t)]
        use_cols = [c for c in mysql_cols if c in sqlite_cols]
        if not use_cols:
            print(f"[{_utc_now()}] skip table={t} (no common cols)")
            continue

        src_n = _rowcount(sqlite, t)
        if src_n == 0:
            print(f"[{_utc_now()}] table={t} rows=0 (skip)")
            continue

        print(f"[{_utc_now()}] copying table={t} rows={src_n} cols={len(use_cols)} ...")

        # Stream from sqlite, insert into mysql
        inserted = 0
        with sqlite.connect() as src, mysql.begin() as dst:
            res = src.execute(text(f"SELECT {', '.join(use_cols)} FROM `{t}`"))
            batch: list[dict[str, Any]] = []
            for row in res.mappings():
                raw = {k: row.get(k) for k in use_cols}
                batch.append(_prepare_row(raw, col_maxlens=col_maxlens))
                if len(batch) >= chunk_size:
                    dst.execute(text(_insert_sql(t, use_cols)), batch)
                    inserted += len(batch)
                    batch.clear()
            if batch:
                dst.execute(text(_insert_sql(t, use_cols)), batch)
                inserted += len(batch)

        dst_n = _rowcount(mysql, t)
        if dst_n != src_n:
            raise SystemExit(f"rowcount mismatch table={t} sqlite={src_n} mysql={dst_n}")
        print(f"[{_utc_now()}] table={t} ok (inserted={inserted})")

    print(f"[{_utc_now()}] resetting AUTO_INCREMENT ...")
    _reset_autoinc(mysql, tables)
    print(f"[{_utc_now()}] migration done.")


def _insert_sql(table: str, cols: list[str]) -> str:
    col_sql = ", ".join(f"`{c}`" for c in cols)
    val_sql = ", ".join(f":{c}" for c in cols)
    return f"INSERT INTO `{table}` ({col_sql}) VALUES ({val_sql})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate data/etf_momentum.sqlite3 into local MySQL (same schema).")
    ap.add_argument("--sqlite", default="data/etf_momentum.sqlite3", help="Path to source sqlite file")
    ap.add_argument("--force", action="store_true", help="Wipe target MySQL tables before migration")
    ap.add_argument("--chunk-size", type=int, default=2000, help="Insert batch size")
    args = ap.parse_args(argv)

    migrate_sqlite_to_mysql(sqlite_path=str(args.sqlite), force=bool(args.force), chunk_size=int(args.chunk_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

