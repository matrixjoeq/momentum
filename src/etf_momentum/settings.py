from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOMENTUM_", extra="ignore")

    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/etf_momentum.sqlite3")
    # If provided, overrides sqlite_path and uses this SQLAlchemy database URL.
    # Example (MySQL 8): mysql+pymysql://user:pass@host:3306/dbname?charset=utf8mb4
    db_url: str | None = None

    default_start_date: str = "20100101"  # YYYYMMDD
    default_end_date: str = "20991231"  # YYYYMMDD

    log_level: str = "INFO"

    # --- Market data sync (Cloud trigger preferred) ---
    # Recommended: use WeChat Cloud scheduled trigger to call /api/admin/sync/fixed-pool.
    # If you still want in-process scheduler, you can enable it explicitly.
    auto_sync_enabled: bool = False
    auto_sync_tz: str = "Asia/Shanghai"
    auto_sync_calendar: str = "XSHG"
    auto_sync_hour: int = 15
    auto_sync_minute: int = 15  # after close (15:00) with some buffer
    # For data consistency/completeness, refresh full history every run (qfq/hfq/none).
    # This is heavier than incremental mode but guarantees no stale history.
    auto_sync_full_refresh: bool = True

    # If set, /api/admin/sync/fixed-pool requires this token via header X-Momentum-Token or body.token
    sync_token: str | None = None


def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    # ensure sqlite parent exists for local dev (only when sqlite is used)
    if not s.db_url:
        s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return s

