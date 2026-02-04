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


class SyncFixedPoolRequest(BaseModel):
    """
    Cloud-triggerable market data sync for the fixed 4-ETF pool.
    Intended to be invoked by WeChat Cloud scheduled trigger (HTTP).
    """

    token: str | None = Field(default=None, description="Optional sync token. If MOMENTUM_SYNC_TOKEN is set, token is required.")
    date: str | None = Field(default=None, description="YYYYMMDD; default=server today (Asia/Shanghai)")
    adjusts: list[str] = Field(default_factory=lambda: ["qfq", "hfq", "none"], description="Adjust list: subset of qfq/hfq/none")
    full_refresh: bool | None = Field(default=None, description="If true, refresh full history every run; if null, use server default.")
    force_new: bool = Field(
        default=False,
        description="If true, always create a new job id (retry suffix) even if an identical job is already queued/running.",
    )


class SyncFixedPoolResponse(BaseModel):
    ok: bool
    skipped: bool = False
    reason: str | None = None
    date: str
    full_refresh: bool
    adjusts: list[str]
    codes: dict


class SyncJobTriggerResponse(BaseModel):
    job_id: int
    status: str
    dedupe_key: str


class SyncJobOut(BaseModel):
    id: int
    status: str
    job_type: str
    dedupe_key: str
    run_date: str | None = None  # YYYY-MM-DD
    full_refresh: bool
    adjusts: list[str]
    progress: dict | None = None
    result: dict | None = None
    error_message: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None

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


class LeadLagAnalysisRequest(BaseModel):
    """
    Lead/lag and causality study between an ETF and a volatility index (VIX/GVZ).
    """

    etf_code: str = Field(min_length=1, description="ETF code (db mode) or a label for the asset (external mode), e.g. 518880")
    asset_provider: str = Field(default="db", description="db|stooq|yahoo|auto (asset side)")
    asset_symbol: str | None = Field(
        default=None,
        description="When asset_provider != db, the provider symbol to fetch as the asset close series, e.g. qqq.us or ^ndx",
    )
    index_symbol: str = Field(
        min_length=1,
        description="Index/series symbol. Examples: VIX/GVZ (Cboe), ^VIX/^GVZ (Yahoo), DGS2/DGS5/DGS10/DGS30 (FRED), DINIW (Sina), XAUUSD (Stooq), GC.F (Stooq), GC=F (Yahoo).",
    )
    index_provider: str = Field(default="cboe", description="cboe|yahoo|fred|stooq|sina|auto")
    index_align: str = Field(default="cn_next_trading_day", description="none|cn_next_trading_day")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")
    max_lag: int = Field(default=20, ge=0, le=252, description="Cross-correlation lag window (+/- trading days)")
    granger_max_lag: int = Field(default=10, ge=1, le=60, description="Max lag order for Granger causality tests")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Significance level")
    # Trading usefulness evaluation
    trade_cost_bps: float = Field(default=0.0, ge=0.0, description="Per-switch cost (bps) for the toy strategy")
    rolling_window: int = Field(default=252, ge=20, le=2520, description="Rolling window for stability charts (trading days)")
    enable_threshold: bool = Field(default=True, description="If true, add threshold-gated signal evaluation")
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0, description="Quantile on |index_ret| to trigger signals")
    walk_forward: bool = Field(default=True, description="If true, run walk-forward (train->test) parameter selection")
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9, description="Train split ratio for walk-forward")
    walk_objective: str = Field(default="sharpe", description="Walk-forward objective: sharpe|cagr")

    # Volatility-timing strategy (level-based, tiered exposure), e.g. GVZ high -> reduce exposure
    vol_timing: bool = Field(default=False, description="If true, backtest tiered exposure based on index close level quantiles")
    vol_level_quantiles: list[float] = Field(
        default_factory=lambda: [0.8],
        description="Quantile cut points on index close level, ascending. Example: [0.7,0.85,0.95]",
    )
    vol_level_exposures: list[float] = Field(
        default_factory=lambda: [1.0, 0.5],
        description="Tier exposures, length = len(vol_level_quantiles)+1. Example: [1.0,0.7,0.4,0.1]",
    )

    vol_level_window: str = Field(
        default="all",
        description="Quantile window for level-based vol timing: all(expanding,no-lookahead)|static_all(full-sample,lookahead)|1y|3y|5y|10y",
    )

    # Volatility-timing strategy (level-based, tiered exposure), e.g. GVZ high -> reduce exposure
    vol_timing: bool = Field(default=False, description="If true, backtest tiered exposure based on index close level quantiles")
    vol_level_quantiles: list[float] = Field(
        default_factory=lambda: [0.8],
        description="Quantile cut points on index close level, ascending. Example: [0.7,0.85,0.95]",
    )
    vol_level_exposures: list[float] = Field(
        default_factory=lambda: [1.0, 0.5],
        description="Tier exposures, length = len(vol_level_quantiles)+1. Example: [1.0,0.7,0.4,0.1]",
    )


class LeadLagAnalysisResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    corr: dict | None = None
    granger: dict | None = None
    trade: dict | None = None
    error: str | None = None


class MacroPairLeadLagRequest(BaseModel):
    a_series_id: str = Field(min_length=1, description="Series A (target) id, e.g. XAUUSD")
    b_series_id: str = Field(min_length=1, description="Series B (indicator) id, e.g. DGS10 or DINIW")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    index_align: str = Field(default="none", description="none|cn_next_trading_day")
    max_lag: int = Field(default=20, ge=0, le=252)
    granger_max_lag: int = Field(default=10, ge=1, le=60)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    trade_cost_bps: float = Field(default=0.0, ge=0.0)
    rolling_window: int = Field(default=252, ge=20, le=2520)
    enable_threshold: bool = Field(default=True)
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0)
    walk_forward: bool = Field(default=True)
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9)
    walk_objective: str = Field(default="sharpe", description="sharpe|cagr")


class MacroPairLeadLagResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    corr: dict | None = None
    granger: dict | None = None
    trade: dict | None = None
    error: str | None = None


class MacroStep1Request(BaseModel):
    """
    Step 1: global gold (spot/fut), US yields, DXY relations.
    This endpoint reads from local DB only; it does NOT trigger updates.
    """

    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")

    gold_spot_series_id: str = Field(default="XAUUSD", min_length=1)
    gold_fut_series_id: str | None = Field(default="GC_FUT", description="Optional, e.g. GC_FUT")
    dxy_series_id: str = Field(default="DINIW", min_length=1)
    yield_series_id: str = Field(default="DGS10", min_length=1, description="One tenor to focus on, e.g. DGS10")

    index_align: str = Field(default="none", description="none|cn_next_trading_day")
    max_lag: int = Field(default=20, ge=0, le=252)
    granger_max_lag: int = Field(default=10, ge=1, le=60)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    rolling_window: int = Field(default=252, ge=20, le=2520)
    trade_cost_bps: float = Field(default=0.0, ge=0.0)
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0)
    walk_forward: bool = Field(default=True)
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9)
    walk_objective: str = Field(default="sharpe", description="sharpe|cagr")


class MacroStep1Response(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    pairs: dict | None = None
    error: str | None = None


class MacroStep2Request(BaseModel):
    """
    Step 2: CN gold (spot/fut), CNH, CN yields relations.
    Series ids are user-defined (depends on your ingestion naming).
    """

    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")

    cn_spot_series_id: str = Field(default="SGE_AU9999", min_length=1)
    cn_fut_series_id: str | None = Field(default="SHFE_AU", description="Optional")
    cnh_series_id: str = Field(default="USDCNH", min_length=1)
    yield_series_id: str = Field(default="CN10Y", min_length=1)

    index_align: str = Field(default="none", description="none|cn_next_trading_day")
    max_lag: int = Field(default=20, ge=0, le=252)
    granger_max_lag: int = Field(default=10, ge=1, le=60)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    rolling_window: int = Field(default=252, ge=20, le=2520)
    trade_cost_bps: float = Field(default=0.0, ge=0.0)
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0)
    walk_forward: bool = Field(default=True)
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9)
    walk_objective: str = Field(default="sharpe", description="sharpe|cagr")


class MacroStep2Response(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    pairs: dict | None = None
    error: str | None = None


class MacroStep3Request(BaseModel):
    """
    Step 3: CN gold vs global gold (optionally via FX conversion).
    """

    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")

    cn_gold_series_id: str = Field(default="SGE_AU9999", min_length=1)
    global_gold_series_id: str = Field(default="XAUUSD", min_length=1)
    fx_series_id: str = Field(default="USDCNH", min_length=1, description="FX to convert global->CNY")

    index_align: str = Field(default="none", description="none|cn_next_trading_day")
    max_lag: int = Field(default=20, ge=0, le=252)
    granger_max_lag: int = Field(default=10, ge=1, le=60)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    rolling_window: int = Field(default=252, ge=20, le=2520)
    trade_cost_bps: float = Field(default=0.0, ge=0.0)
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0)
    walk_forward: bool = Field(default=True)
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9)
    walk_objective: str = Field(default="sharpe", description="sharpe|cagr")


class MacroStep3Response(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    pairs: dict | None = None
    error: str | None = None


class MacroStep4Request(BaseModel):
    """
    Step 4: CN gold ETF vs CN spot gold.
    ETF comes from etf_prices; spot comes from macro_prices.
    """

    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")

    etf_code: str = Field(default="518880", min_length=1, description="ETF code in etf_pool/etf_prices")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")
    cn_spot_series_id: str = Field(default="SGE_AU9999", min_length=1)

    index_align: str = Field(default="none", description="none|cn_next_trading_day")
    max_lag: int = Field(default=20, ge=0, le=252)
    granger_max_lag: int = Field(default=10, ge=1, le=60)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    rolling_window: int = Field(default=252, ge=20, le=2520)
    trade_cost_bps: float = Field(default=0.0, ge=0.0)
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0)
    walk_forward: bool = Field(default=True)
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9)
    walk_objective: str = Field(default="sharpe", description="sharpe|cagr")


class MacroStep4Response(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    pair: dict | None = None
    error: str | None = None

class VixNextActionRequest(BaseModel):
    """
    Live-tradable next-day instruction for CN ETF using VIX/GVZ (Cboe).
    """

    etf_code: str = Field(default="513100", description="A-share ETF code (default: Nasdaq ETF)")
    index: str = Field(default="VIX", description="Vol index: VIX|GVZ")
    index_align: str = Field(default="cn_next_trading_day", description="none|cn_next_trading_day")
    calendar: str = Field(default="XSHG", description="Exchange calendar for CN trading days")
    current_position: str = Field(default="unknown", description="long|cash|unknown")
    lookback_window: int = Field(default=252, ge=20, le=2520, description="Lookback window for threshold estimation")
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0, description="Quantile on |index_ret| (past window) to trigger trades")
    min_abs_ret: float = Field(default=0.0, ge=0.0, description="Hard minimum abs(log-ret) threshold")
    mode: str = Field(
        default="next_cn_day",
        description="next_cn_day|latest_available. next_cn_day: return action for next CN trading day; "
        "if signal not ready, return error.",
    )
    target_cn_trade_date: str | None = Field(default=None, description="Optional CN trade date YYYYMMDD; if null use latest available mapped date")


class VixNextActionResponse(BaseModel):
    ok: bool
    action_date: str | None = None  # YYYY-MM-DD
    action: str | None = None  # BUY/SELL/HOLD
    target_position: str | None = None  # long/cash/unknown
    current_position: str | None = None
    reason: str | None = None
    index: str | None = None
    index_align: str | None = None
    calendar: str | None = None
    signal: dict | None = None
    error: str | None = None


class VixSignalBacktestRequest(BaseModel):
    etf_code: str = Field(default="513100", description="A-share ETF code (default: Nasdaq ETF)")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")
    index: str = Field(default="VIX", description="Vol index: VIX|GVZ")
    index_align: str = Field(default="cn_next_trading_day", description="none|cn_next_trading_day")
    calendar: str = Field(default="XSHG", description="Exchange calendar for CN trading days")
    lookback_window: int = Field(default=252, ge=20, le=2520, description="Lookback window for threshold estimation")
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0, description="Quantile on |index_log_ret| to trigger trades")
    min_abs_ret: float = Field(default=0.0, ge=0.0, description="Hard minimum abs(log-ret) threshold")
    trade_cost_bps: float = Field(default=10.0, ge=0.0, description="Per-switch cost (bps) when position changes")
    initial_position: str = Field(default="long", description="long|cash starting position at start date")
    initial_nav: float = Field(default=1.0, gt=0.0, description="Initial NAV")


class VixSignalBacktestResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    metrics: dict | None = None
    trades: list[dict] | None = None
    error: str | None = None


class IndexDistributionRequest(BaseModel):
    symbol: str = Field(description="Cboe symbol: GVZ|VXN|VIX")
    window: str = Field(default="all", description="1y|3y|5y|10y|all")
    bins: int = Field(default=60, ge=10, le=200)


class IndexDistributionResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    close: dict | None = None
    ret_log: dict | None = None
    error: str | None = None


class VolProxyMethod(BaseModel):
    """
    Volatility proxy computed from the ETF OHLC series.

    Output level is annualized volatility (decimal), intended for level-based tier timing.
    """

    name: str = Field(min_length=1, description="Unique method name in response, e.g. rv20, yz20, har252")
    kind: str = Field(
        description="rv_close|ewma_close|parkinson|garman_klass|rogers_satchell|yang_zhang|har_rv"
    )
    window: int = Field(default=20, ge=2, le=2520, description="Rolling window (trading days)")
    ann: int = Field(default=252, ge=50, le=400, description="Annualization factor")

    ewma_lambda: float = Field(default=0.94, gt=0.0, lt=1.0, description="EWMA decay (ewma_close only)")
    har_train_window: int = Field(default=252, ge=30, le=2520, description="Rolling train window (har_rv only)")
    har_horizons: list[int] = Field(default_factory=lambda: [1, 5, 22], description="HAR horizons (har_rv only)")


class VolProxyTimingRequest(BaseModel):
    etf_code: str = Field(min_length=1, description="ETF code, e.g. 518880 / 513100")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")

    methods: list[VolProxyMethod] = Field(min_length=1, description="Vol proxy variants to compute and backtest")

    # Tiering config (same semantics as leadlag vol_timing)
    level_quantiles: list[float] = Field(default_factory=lambda: [0.8, 0.9], description="Quantiles on level (train)")
    level_exposures: list[float] = Field(default_factory=lambda: [1.0, 0.5, 0.2], description="Tier exposures, len=quantiles+1")
    level_window: str = Field(
        default="all",
        description="Quantile window for levels: all(expanding,no-lookahead)|static_all(full-sample,lookahead)|1y|3y|5y|10y",
    )
    trade_cost_bps: float = Field(default=0.0, ge=0.0, description="Per-switch cost in bps")

    walk_forward: bool = Field(default=True, description="If true, split train/test and apply train thresholds to test")
    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9, description="Train ratio for walk-forward")


class VolProxyTimingResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    methods: dict | None = None
    error: str | None = None


class BaselineCalendarEffectRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none (global)")
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    rebalance: str = Field(default="weekly", description="weekly/monthly/quarterly/yearly (calendar-effect study)")
    rebalance_shift: str = Field(
        default="prev",
        description="If anchor falls on non-trading day: prev -> shift to previous trading day (default); next -> shift to next trading day.",
    )
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    # Backward compatibility: old field name used by weekly-only UI/tests.
    weekdays: list[int] | None = Field(default=None, description="[deprecated] same as anchors when rebalance=weekly")
    exec_prices: list[str] = Field(default_factory=lambda: ["open", "close", "oc2"], description="Execution price list: open|close|oc2 (OC average)")


class AssetRiskControlRule(BaseModel):
    """
    Per-asset risk-control rule applied to weights daily:
    - signal is computed from the asset's own hfq close-based NAV proxy
    - when triggered, scale that asset's weight by (1 - reduce_pct)
    """

    code: str = Field(min_length=1, description="ETF code")
    sig_type: str = Field(description="return|volatility|downside_vol|drawdown")
    k: int = Field(ge=2, le=2520, description="Signal lookback window in trading days")
    p_in: float = Field(gt=0.0, lt=100.0, description="Trigger percentile (0-100). For return, low-tail is used (100-p_in).")
    reduce_pct: float = Field(ge=0.0, le=100.0, description="Reduce percent on trigger: exposure = 1 - reduce_pct/100")
    recovery_mode: str = Field(default="immediate", description="immediate|hysteresis|cooldown")
    p_out: float | None = Field(default=None, gt=0.0, lt=100.0, description="Recovery percentile for hysteresis mode")
    cooldown_days: int = Field(default=0, ge=0, le=2520, description="Minimum days to keep reduced exposure (cooldown mode)")


class AssetVolIndexTimingRule(BaseModel):
    """
    Per-asset volatility-index timing rule applied daily to weights:
    - signal is the volatility index LEVEL (e.g. VIX/GVZ), expected aligned to CN next trading day
    - thresholds are computed from rolling/expanding quantiles and shifted by 1 day (no lookahead)
    - when triggered (higher-vol bucket), scale that asset's weight by the tier exposure (cash remainder)
    """

    code: str = Field(min_length=1, description="ETF code")
    index: str = Field(description="Vol index code: VIX|GVZ (Cboe)")
    level_window: str = Field(
        default="all",
        description="Quantile lookback window: 30d|90d|180d|1y|3y|5y|10y|all (all=expanding; non-static to avoid lookahead).",
    )
    level_quantiles: list[float] = Field(
        default_factory=lambda: [0.8],
        description="Ascending quantiles in (0,1). Example [0.8] means 'top 20%' bucket.",
    )
    level_exposures: list[float] = Field(
        default_factory=lambda: [1.0, 0.5],
        description="Tier exposures in [0,1], length must be len(level_quantiles)+1. Example [1.0,0.5].",
    )
    min_periods: int = Field(default=20, ge=2, le=2520, description="Minimum observations before thresholds become active.")


class RotationBacktestRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    rebalance: str = Field(default="weekly", description="daily/weekly/monthly/quarterly/yearly")
    rebalance_shift: str = Field(
        default="prev",
        description="If anchor falls on non-trading day: prev -> shift to previous trading day (default); next -> shift to next trading day. Only used when rebalance_anchor is set (e.g. calendar-effect study).",
    )
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
    # Inertia / dampening (avoid frequent rebalances)
    inertia: bool = Field(default=False, description="Enable inertia (dampening) to avoid frequent rebalances.")
    inertia_min_hold_periods: int = Field(default=0, ge=0, description="Minimum decision periods between holding changes (0 disables).")
    inertia_score_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Only for top_k=1: require new_score - cur_score >= gap to switch (0 disables).",
    )
    inertia_min_turnover: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Skip rebalance if expected turnover < threshold (0 disables).",
    )
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
    # Timing (strategy NAV RSI gate; uses shadow NAV that ignores this timing gate for RSI signal)
    timing_rsi_gate: bool = Field(default=False, description="Enable timing: sleep when strategy NAV RSI < 50, reactivate when >= 50 (signal from shadow NAV)")
    timing_rsi_window: int = Field(default=24, ge=2, description="RSI window (trading days) for timing gate; typical 6/12/24; default=24")
    # Per-asset risk control rules (optional; applied daily to weights as exposure scaling)
    asset_rc_rules: list[AssetRiskControlRule] | None = Field(
        default=None,
        description="Optional per-asset risk-control rules (signal on hfq close-based NAV; scales down weights when triggered).",
    )
    asset_vol_index_rules: list[AssetVolIndexTimingRule] | None = Field(
        default=None,
        description="Optional per-asset vol-index timing rules (e.g. 518880->GVZ, 513100->VIX). Thresholds use rolling/expanding quantiles with shift(1) to avoid lookahead; scales weights daily (cash remainder).",
    )


class RotationCalendarEffectRequest(RotationBacktestRequest):
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    weekdays: list[int] | None = Field(default=None, description="[deprecated] same as anchors when rebalance=weekly")
    exec_prices: list[str] = Field(default_factory=lambda: ["open", "close", "oc2"], description="Execution price list: open|close|oc2 (OC average)")


class RotationWeekly5OpenSimRequest(BaseModel):
    """
    Simplified weekly simulation used by the mini-program:
    - weekly rebalance
    - anchors fixed to Mon..Fri (0..4) in one call
    - exec_price fixed to open
    - all risk controls off
    - cost_bps fixed to 0
    """

    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    anchor_weekday: int | None = Field(
        default=None,
        ge=0,
        le=4,
        description="Optional: if set, compute only one anchor weekday (0=Mon..4=Fri) to reduce payload/runtime.",
    )
    asset_rc_rules: list[AssetRiskControlRule] | None = Field(
        default=None,
        description="Optional per-asset risk-control rules to apply (same as rotation backtest).",
    )
    asset_vol_index_rules: list[AssetVolIndexTimingRule] | None = Field(
        default=None,
        description="Optional per-asset vol-index timing rules to apply (same as rotation backtest).",
    )


class RotationNextPlanRequest(BaseModel):
    """
    Next rebalance plan for the fixed 4-ETF mini-program strategy (weekly, top1, lookback20, open execution).
    Used by the mini-program to show "tomorrow plan" when tomorrow is a rebalance effective day.
    """

    anchor_weekday: int = Field(ge=0, le=4, description="0=Mon..4=Fri")
    asof: str = Field(description="YYYYMMDD (usually the latest available trading day in backtest range)")


class SimPortfolioCreateRequest(BaseModel):
    name: str = Field(default="默认账户", description="Portfolio name")
    initial_cash: float = Field(default=1_000_000.0, gt=0.0, description="Initial cash (base_ccy units)")


class SimPortfolioOut(BaseModel):
    id: int
    name: str
    base_ccy: str
    initial_cash: float
    created_at: str


class SimInitFixedStrategyResponse(BaseModel):
    portfolio_id: int
    config_id: int
    variant_ids: list[int]


class SimDecisionGenerateRequest(BaseModel):
    portfolio_id: int = Field(ge=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")


class SimTradePreviewRequest(BaseModel):
    variant_id: int = Field(ge=1)
    decision_id: int = Field(ge=1)


class SimTradeConfirmRequest(BaseModel):
    variant_id: int = Field(ge=1)
    decision_id: int = Field(ge=1)


class BaselineWeekly5EWDashboardRequest(BaseModel):
    """
    Equal-weight benchmark dashboard for the fixed 4-ETF pool, for 5 weekly anchor weekdays (MON..FRI).
    Uses hfq close and rebalances at close on decision_date (effective next trading day).
    """

    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    rebalance_shift: str = Field(default="prev", description="prev|next when anchor falls on non-trading day")
    anchor_weekday: int | None = Field(
        default=None,
        ge=0,
        le=4,
        description="Optional: if set, compute only one anchor weekday (0=Mon..4=Fri) to reduce payload/runtime.",
    )


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

