from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOMENTUM_", extra="ignore")

    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/etf_momentum.sqlite3")

    default_start_date: str = "20100101"  # YYYYMMDD
    default_end_date: str = "20991231"  # YYYYMMDD

    log_level: str = "INFO"

    # --- Background scheduler: auto-sync market data after close ---
    # NOTE: In tests (pytest), the scheduler is automatically disabled.
    auto_sync_enabled: bool = True
    auto_sync_tz: str = "Asia/Shanghai"
    auto_sync_calendar: str = "XSHG"
    auto_sync_hour: int = 15
    auto_sync_minute: int = 10  # after close (15:00) with some buffer
    # For data consistency/completeness, refresh full history every run (qfq/hfq/none).
    # This is heavier than incremental mode but guarantees no stale history.
    auto_sync_full_refresh: bool = True


def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    # ensure parent exists
    s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return s

