from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationPolicyOut(BaseModel):
    id: int
    name: str
    description: str | None = None
    max_abs_return: float
    max_hl_spread: float
    max_gap_days: int


class EtfPoolUpsert(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")
    validation_policy_id: int | None = None
    max_abs_return_override: float | None = None


class EtfPoolOut(BaseModel):
    code: str
    name: str
    start_date: str | None
    end_date: str | None
    validation_policy: ValidationPolicyOut | None = None
    max_abs_return_effective: float | None = None
    max_abs_return_override: float | None = None
    last_fetch_status: str | None = None
    last_fetch_message: str | None = None
    last_data_start_date: str | None = None  # YYYYMMDD
    last_data_end_date: str | None = None  # YYYYMMDD


class FetchResult(BaseModel):
    code: str
    inserted_or_updated: int
    status: str
    message: str | None = None


class FetchSelectedRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="ETF codes to fetch")
    adjust: str = Field(default="hfq", description="qfq/hfq/none (global)")


class PriceOut(BaseModel):
    code: str
    trade_date: str  # YYYY-MM-DD
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    amount: float | None
    source: str
    adjust: str


class IngestionBatchOut(BaseModel):
    id: int
    code: str
    start_date: str
    end_date: str
    source: str
    adjust: str
    status: str
    message: str | None = None
    snapshot_path: str | None = None
    pre_fingerprint: str | None = None
    post_fingerprint: str | None = None
    val_max_abs_return: float | None = None
    val_max_hl_spread: float | None = None
    val_max_gap_days: int | None = None


class BaselineAnalysisRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    benchmark_code: str | None = None
    adjust: str = Field(default="hfq", description="qfq/hfq/none (global)")
    rebalance: str = Field(default="weekly", description="daily/weekly/monthly/quarterly/yearly/none")
    risk_free_rate: float = Field(
        default=0.025,
        description="Annualized risk-free rate for Sharpe/Sortino (decimal). Default 0.025 ~= 2.5% (CN 0-1y gov).",
    )
    rolling_weeks: list[int] = Field(default_factory=lambda: [4, 12, 52])
    rolling_months: list[int] = Field(default_factory=lambda: [3, 6, 12])
    rolling_years: list[int] = Field(default_factory=lambda: [1, 3])


class RotationBacktestRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    rebalance: str = Field(default="weekly", description="daily/weekly/monthly/quarterly/yearly")
    top_k: int = Field(default=1, ge=1)
    lookback_days: int = Field(default=20, ge=1)
    skip_days: int = Field(default=0, ge=0)
    risk_off: bool = False
    defensive_code: str | None = None
    momentum_floor: float = 0.0
    score_method: str = Field(
        default="raw_mom",
        description="Ranking score: raw_mom | sharpe_mom | sortino_mom | return_over_vol | mom_minus_lambda_vol | mom_over_vol_power",
    )
    score_lambda: float = Field(default=0.0, description="Used by mom_minus_lambda_vol: score = mom - lambda * vol")
    score_vol_power: float = Field(default=1.0, description="Used by mom_over_vol_power: score = mom / vol**power")
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    cost_bps: float = Field(default=0.0, ge=0.0)
    # Pre-trade risk controls (all optional; defaults keep previous behavior)
    trend_filter: bool = Field(default=False, description="Enable trend filter gating (pre-trade)")
    trend_mode: str = Field(default="each", description="each|universe")
    trend_sma_window: int = Field(default=200, ge=1, description="SMA window for trend filter (trading days)")
    rsi_filter: bool = Field(default=False, description="Enable RSI filter gating (pre-trade)")
    rsi_window: int = Field(default=14, ge=1, description="RSI window (trading days)")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0)
    rsi_block_overbought: bool = Field(default=True, description="If true, exclude assets with RSI > overbought")
    rsi_block_oversold: bool = Field(default=False, description="If true, exclude assets with RSI < oversold")
    vol_monitor: bool = Field(default=False, description="Enable volatility-based position sizing (pre-trade)")
    vol_window: int = Field(default=20, ge=1, description="Realized vol window (trading days)")
    vol_target_ann: float = Field(default=0.20, gt=0.0, description="Annualized target vol for sizing")
    vol_max_ann: float = Field(default=0.60, gt=0.0, description="Annualized hard stop vol; above -> no risk position")


class MonteCarloRequest(BaseModel):
    n_sims: int = Field(default=10000, ge=100, le=50000, description="Number of Monte Carlo simulations")
    block_size: int = Field(default=5, ge=1, le=252, description="Circular block size in trading days")
    seed: int | None = Field(default=None, description="Optional RNG seed for reproducibility")
    sample_window_days: int | None = Field(
        default=None,
        ge=2,
        le=20000,
        description="Optional rolling window length (trading days) used as sampling pool; None means full backtest range.",
    )


class BaselineMonteCarloRequest(BaselineAnalysisRequest, MonteCarloRequest):
    pass


class RotationMonteCarloRequest(RotationBacktestRequest, MonteCarloRequest):
    pass

