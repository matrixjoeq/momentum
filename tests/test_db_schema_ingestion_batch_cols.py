from __future__ import annotations

from sqlalchemy import text

from etf_momentum.db.schema import ensure_sqlite_schema
from etf_momentum.db.session import make_engine


def test_schema_adds_ingestion_batch_validation_columns(tmp_path) -> None:
    sqlite_path = tmp_path / "legacy.sqlite3"
    engine = make_engine(str(sqlite_path))

    # Create legacy tables required by ensure_sqlite_schema
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE etf_pool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(32) NOT NULL UNIQUE,
                    name VARCHAR(128) NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE ingestion_batch (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(32) NOT NULL,
                    source VARCHAR(32) NOT NULL,
                    adjust VARCHAR(8) NOT NULL,
                    start_date VARCHAR(8) NOT NULL,
                    end_date VARCHAR(8) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    message VARCHAR(512),
                    snapshot_path VARCHAR(512),
                    pre_fingerprint VARCHAR(128),
                    post_fingerprint VARCHAR(128),
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )

    ensure_sqlite_schema(engine)

    with engine.connect() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(ingestion_batch)")).fetchall()]
    assert "val_max_abs_return" in cols
    assert "val_max_hl_spread" in cols
    assert "val_max_gap_days" in cols

