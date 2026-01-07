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
    fft_windows: list[int] = Field(
        default_factory=lambda: [252, 126],
        description="FFT rolling windows in trading days, e.g. [252,126] -> last_252 and last_126 summaries",
    )
    fft_roll: bool = Field(default=True, description="If true, compute rolling FFT time series for EW (downsampled by fft_roll_step)")
    fft_roll_step: int = Field(default=5, ge=1, description="Compute rolling FFT features every N trading days to reduce runtime")


class BaselineCalendarEffectRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none (global)")
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    rebalance: str = Field(default="weekly", description="weekly/monthly/quarterly/yearly (calendar-effect study)")
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    # Backward compatibility: old field name used by weekly-only UI/tests.
    weekdays: list[int] | None = Field(default=None, description="[deprecated] same as anchors when rebalance=weekly")
    exec_prices: list[str] = Field(default_factory=lambda: ["open", "close", "oc2"], description="Execution price list: open|close|oc2 (OC average)")


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
    trend_sma_window: int = Field(default=20, ge=1, description="SMA window for trend filter (trading days)")
    rsi_filter: bool = Field(default=False, description="Enable RSI filter gating (pre-trade)")
    rsi_window: int = Field(default=20, ge=1, description="RSI window (trading days)")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0)
    rsi_block_overbought: bool = Field(default=True, description="If true, exclude assets with RSI > overbought")
    rsi_block_oversold: bool = Field(default=False, description="If true, exclude assets with RSI < oversold")
    vol_monitor: bool = Field(default=False, description="Enable volatility-based position sizing (pre-trade)")
    vol_window: int = Field(default=20, ge=1, description="Realized vol window (trading days)")
    vol_target_ann: float = Field(default=0.20, gt=0.0, description="Annualized target vol for sizing")
    vol_max_ann: float = Field(default=0.60, gt=0.0, description="Annualized hard stop vol; above -> no risk position")
    chop_filter: bool = Field(default=False, description="Enable choppiness filter (ER/ADX)")
    chop_mode: str = Field(default="er", description="Choppiness mode: er|adx")
    chop_window: int = Field(default=20, ge=2, description="Efficiency Ratio window (trading days)")
    chop_er_threshold: float = Field(default=0.25, gt=0.0, description="ER < threshold => choppy => exclude")
    chop_adx_window: int = Field(default=20, ge=2, description="ADX window (trading days)")
    chop_adx_threshold: float = Field(default=20.0, gt=0.0, description="ADX < threshold => choppy => exclude")
    # Take-profit / stop-loss (qfq)
    tp_sl_mode: str = Field(
        default="none",
        description="Take-profit/stop-loss mode: none | prev_week_low_stop (qfq low-based stop-loss).",
    )
    # ATR chandelier (qfq close-based)
    atr_window: int | None = Field(
        default=None,
        ge=2,
        description="ATR lookback window (trading days) computed from qfq close. None -> defaults to lookback_days.",
    )
    atr_mult: float = Field(default=2.0, gt=0.0, description="ATR stop multiple (e.g. 2.0).")
    atr_step: float = Field(default=0.5, gt=0.0, description="Progressive mode step in ATR units (e.g. 0.5).")
    atr_min_mult: float = Field(default=0.5, gt=0.0, description="Progressive mode minimum distance multiple (e.g. 0.5).")
    # Correlation filter (hfq)
    corr_filter: bool = Field(default=False, description="Enable correlation gate between new pick and current holding (hfq).")
    corr_window: int | None = Field(
        default=None,
        ge=2,
        description="Correlation lookback window (trading days) computed from hfq closes. None -> defaults to lookback_days.",
    )
    corr_threshold: float = Field(default=0.5, ge=-1.0, le=1.0, description="Block rebalance if corr > threshold.")
    # Rolling-return based position sizing (strategy trailing return)
    rr_sizing: bool = Field(default=False, description="Enable rolling-return based exposure sizing at rebalance.")
    rr_years: float = Field(default=3.0, gt=0.0, description="Trailing window length in years (approx 252*years trading days).")
    rr_thresholds: list[float] | None = Field(
        default=None,
        description="Optional return thresholds (decimal). Max 5. If null, backend uses defaults when rr_sizing=true.",
    )
    rr_weights: list[float] | None = Field(
        default=None,
        description="Optional exposure levels. If null, backend uses defaults when rr_sizing=true.",
    )
    # Drawdown control (strategy NAV)
    dd_control: bool = Field(default=False, description="Enable drawdown control (based on strategy NAV drawdown).")
    dd_threshold: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="Trigger when drawdown >= threshold (decimal). Default 0.10 (=10%).",
    )
    dd_reduce: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Reduce position by this fraction on trigger (0..1). Default 1.0 (=reduce 100% -> cash).",
    )
    dd_sleep_days: int = Field(default=20, ge=1, description="Sleep days after trigger (trading days). Default 20.")


class RotationCalendarEffectRequest(RotationBacktestRequest):
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    weekdays: list[int] | None = Field(default=None, description="[deprecated] same as anchors when rebalance=weekly")
    exec_prices: list[str] = Field(default_factory=lambda: ["open", "close", "oc2"], description="Execution price list: open|close|oc2 (OC average)")


class TrendBacktestRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32, description="Single ETF code for trend-following backtest")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(
        default="hfq",
        description="[deprecated] Trend research uses mixed basis like rotation: signal=qfq, nav=none with hfq fallback, benchmark=hfq.",
    )
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    cost_bps: float = Field(default=0.0, ge=0.0, description="Round-trip transaction cost in bps per turnover")
    strategy: str = Field(default="ma_filter", description="ma_filter|ema_filter|ma_cross|donchian|tsmom|linreg_slope|bias (long/cash)")
    # parameters (some are strategy-specific)
    sma_window: int = Field(default=200, ge=2, description="MA filter window (trading days)")
    fast_window: int = Field(default=50, ge=2, description="Fast MA window (trading days)")
    slow_window: int = Field(default=200, ge=2, description="Slow MA window (trading days)")
    donchian_entry: int = Field(default=20, ge=2, description="Donchian entry window (trading days)")
    donchian_exit: int = Field(default=10, ge=2, description="Donchian exit window (trading days)")
    mom_lookback: int = Field(default=252, ge=2, description="TS momentum lookback (trading days)")
    # BIAS strategy params
    bias_ma_window: int = Field(default=20, ge=2, description="EMA window N in BIAS=(LN(C)-LN(EMA(C,N)))*100 (trading days)")
    bias_entry: float = Field(default=2.0, description="Enter when BIAS > entry (percent)")
    bias_hot: float = Field(default=10.0, description="Take-profit exit when BIAS >= hot (percent)")
    bias_cold: float = Field(default=-2.0, description="Stop-loss exit when BIAS <= cold (percent)")
    bias_pos_mode: str = Field(default="binary", description="Position mode for BIAS strategy: binary|continuous")


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

