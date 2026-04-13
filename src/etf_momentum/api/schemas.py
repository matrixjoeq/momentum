from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class FetchAllRequest(BaseModel):
    """Body for POST /fetch-all: multi-symbol fetch mode (per-symbol still ingests qfq/hfq/none serially)."""

    fetch_mode: Literal["serial", "parallel"] = Field(
        default="serial",
        description='Batch fetch for all pool symbols: "serial" or "parallel" across symbols.',
    )
    parallel_symbol_workers: int = Field(
        default=2,
        ge=2,
        le=5,
        description="When fetch_mode=parallel: max concurrent symbols (2–5). Ignored when serial.",
    )


class FetchSelectedRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="ETF codes to fetch")
    adjust: str = Field(default="hfq", description="qfq/hfq/none (global)")
    fetch_mode: Literal["serial", "parallel"] = Field(
        default="serial",
        description='When multiple codes: "serial" or "parallel" across symbols.',
    )
    parallel_symbol_workers: int = Field(
        default=2,
        ge=2,
        le=5,
        description="When fetch_mode=parallel: max concurrent symbols (2–5). Ignored when serial.",
    )


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


class OffFundPoolUpsert(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")


class OffFundPoolOut(BaseModel):
    code: str
    name: str
    start_date: str | None
    end_date: str | None
    last_fetch_status: str | None = None
    last_fetch_message: str | None = None
    last_data_start_date: str | None = None
    last_data_end_date: str | None = None


class OffFundFetchResult(BaseModel):
    code: str
    inserted_or_updated: int
    status: str
    message: str | None = None


class OffFundFetchSelectedRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="off-fund codes to fetch")


class OffFundNavOut(BaseModel):
    code: str
    trade_date: str  # YYYY-MM-DD
    nav: float | None
    accum_nav: float | None
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
    rp_window_days: int = Field(
        default=60,
        ge=2,
        le=2520,
        description="ERC / inverse-vol rolling window (trading days): sample covariance for RP (ERC) and vol for IVOL",
    )
    holding_mode: str = Field(default="EW", description="Holding strategy mode: EW|RP (ERC)|IVOL (inverse-vol)|CUSTOM")
    custom_weights: dict[str, float] | None = Field(
        default=None,
        description="Custom target weights by code in decimal (e.g. {'510300':0.4,'518880':0.3}); leftover to cash.",
    )
    dynamic_universe: bool = Field(
        default=False,
        description="If true, use dynamic universe over union interval; otherwise legacy common-interval (intersection).",
    )
    corr_min_obs: int = Field(
        default=20,
        ge=3,
        description="Minimum pairwise observations for correlation; below threshold returns null ('-').",
    )
    exec_price: str = Field(
        default="close",
        description="等权组合成交价口径: open=执行日开盘价, close=执行日收盘价, oc2=OC均价",
    )


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
    trade_cost_bps: float = Field(default=2.0, ge=0.0, description="Per-switch cost (bps) for the toy strategy")
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
    trade_cost_bps: float = Field(default=2.0, ge=0.0)
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
    trade_cost_bps: float = Field(default=2.0, ge=0.0)
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


class MacroSeriesBatchRequest(BaseModel):
    # Optional: if omitted/null/empty, backend will auto-use each series' full stored date range.
    start: str | None = Field(default=None, description="YYYYMMDD (optional)")
    end: str | None = Field(default=None, description="YYYYMMDD (optional)")
    series_ids: list[str] = Field(min_length=1, description="macro series_id list to fetch from macro_prices")


class MacroSeriesBatchResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    error: str | None = None


class SimGbmPhase1Request(BaseModel):
    start: str = Field(default="19900101", description="YYYYMMDD")
    end: str | None = Field(default=None, description="YYYYMMDD; default=last business day")
    n_assets: int = Field(default=4, ge=2, le=20)
    vol_low: float = Field(default=0.05, gt=0.0, lt=2.0)
    vol_high: float = Field(default=0.30, gt=0.0, lt=2.0)
    corr_low: float | None = Field(default=None, ge=-0.99, lt=0.99, description="Optional pairwise correlation lower bound; unset means uncorrelated")
    corr_high: float | None = Field(default=None, ge=-0.99, lt=0.99, description="Optional pairwise correlation upper bound; unset means uncorrelated")
    mu_low: float | None = Field(default=None, ge=-1.0, le=3.0, description="Optional annual drift lower bound; unset uses random default range")
    mu_high: float | None = Field(default=None, ge=-1.0, le=3.0, description="Optional annual drift upper bound; unset uses random default range")
    seed: int | None = Field(default=None)


class SimGbmHoldingStrategyParams(BaseModel):
    rebalance: str = Field(default="weekly", description="daily|weekly|monthly|quarterly|yearly|none")
    cost_bps: float = Field(default=2.0, ge=0.0, description="Round-trip transaction cost in bps per turnover")
    rp_vol_window: int = Field(default=20, ge=2, le=2520, description="Rolling vol window for risk-parity allocation")


class SimGbmPhase2Request(SimGbmPhase1Request):
    lookback_days: int = Field(default=20, ge=2, le=2520)
    strategy_a: dict | None = Field(default=None, description="Optional rotation strategy params (same semantics as A/B strategy A)")
    strategy_b: dict | None = Field(default=None, description="Optional rotation strategy params for B variant")
    target_a: str | None = Field(default=None, description="Compare target A: cash|equal_weight|risk_parity|rotation_a|rotation_b")
    target_b: str | None = Field(default=None, description="Compare target B: cash|equal_weight|risk_parity|rotation_a|rotation_b")
    holding_strategy: SimGbmHoldingStrategyParams = Field(default_factory=SimGbmHoldingStrategyParams)
    holding_strategy_a: SimGbmHoldingStrategyParams | None = Field(default=None)
    holding_strategy_b: SimGbmHoldingStrategyParams | None = Field(default=None)
    phase1_base: dict | None = Field(default=None, description="Optional phase1 payload to reuse generated GBM world directly")


class SimGbmPhase3Request(SimGbmPhase2Request):
    n_sims: int = Field(default=10000, ge=8, le=50000)
    chunk_size: int = Field(default=200, ge=1, le=2000)
    n_jobs: int = Field(default=0, ge=0, le=64, description="Parallel workers for Monte Carlo runs; 0=auto")


class SimGbmPhase4Request(SimGbmPhase3Request):
    initial_cash: float = Field(default=1_000_000.0, gt=0.0)
    position_pct: float = Field(default=0.10, ge=0.0, le=10.0)


class SimGbmAbStrategyParams(BaseModel):
    rebalance: str = Field(default="weekly", description="daily/weekly/monthly/quarterly/yearly")
    rebalance_anchor: int | None = Field(default=None)
    rebalance_shift: str = Field(default="prev")
    exec_price: str = Field(default="open", description="open|close|oc2")
    top_k: int = Field(
        default=1,
        description="Non-zero: top-K by momentum if positive, bottom-K (inverse) if negative; effective=min(|K|, pool).",
    )
    position_mode: str = Field(default="adaptive", description="adaptive|fixed")
    entry_backfill: bool = Field(default=False)
    entry_match_n: int = Field(default=0, ge=0)
    exit_match_n: int = Field(default=0, ge=0)
    lookback_days: int = Field(default=20, ge=1)
    skip_days: int = Field(default=0, ge=0)
    score_method: str = Field(default="raw_mom")
    risk_free_rate: float = Field(default=0.025)
    cost_bps: float = Field(default=2.0, ge=0.0)
    trend_filter: bool = Field(default=False)
    trend_exit_filter: bool = Field(default=False)
    trend_sma_window: int = Field(default=20, ge=1)
    trend_ma_type: str = Field(default="sma", description="sma|ema|vma(variable/adaptive)")
    bias_filter: bool = Field(default=False)
    bias_exit_filter: bool = Field(default=False)
    bias_type: str = Field(default="bias", description="BIAS signal type: bias|bias_v")
    bias_ma_window: int = Field(default=20, ge=2)
    bias_level_window: str = Field(default="all")
    bias_threshold_type: str = Field(default="quantile", description="quantile|fixed")
    bias_quantile: float = Field(default=95.0, gt=0.0, lt=100.0)
    bias_fixed_value: float = Field(default=10.0, ge=0.0)
    bias_min_periods: int = Field(default=20, ge=2, le=2520)
    group_enforce: bool = Field(default=False)
    group_pick_policy: str = Field(default="strongest_score")
    asset_groups: dict[str, str] | None = Field(default=None)
    dynamic_universe: bool = Field(default=False)
    asset_momentum_floor_rules: list[AssetMomentumFloorRule] | None = Field(default=None)
    asset_trend_rules: list[AssetTrendRule] | None = Field(default=None)
    asset_bias_rules: list[AssetBiasRule] | None = Field(default=None)
    asset_vol_index_rules: list[AssetVolIndexTimingRule] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_sim_ab_top_k(self) -> SimGbmAbStrategyParams:
        if int(self.top_k) == 0:
            raise ValueError("top_k must be non-zero")
        return self


class SimGbmAbSignificanceRequest(BaseModel):
    start: str = Field(default="19900101", description="YYYYMMDD")
    end: str | None = Field(default=None, description="YYYYMMDD; default=last business day")
    n_worlds: int = Field(default=3000, ge=2, le=20000)
    n_assets: int = Field(default=4, ge=2, le=20)
    vol_low: float = Field(default=0.05, gt=0.0, lt=2.0)
    vol_high: float = Field(default=0.30, gt=0.0, lt=2.0)
    corr_low: float | None = Field(default=None, ge=-0.99, lt=0.99, description="Optional pairwise correlation lower bound; unset means uncorrelated")
    corr_high: float | None = Field(default=None, ge=-0.99, lt=0.99, description="Optional pairwise correlation upper bound; unset means uncorrelated")
    mu_low: float | None = Field(default=None, ge=-1.0, le=3.0, description="Optional annual drift lower bound; unset uses random default range")
    mu_high: float | None = Field(default=None, ge=-1.0, le=3.0, description="Optional annual drift upper bound; unset uses random default range")
    seed: int | None = Field(default=None)
    n_perm: int = Field(default=5000, ge=200, le=20000, description="Permutations; UI default is high; tests may use fewer")
    n_boot: int = Field(default=3000, ge=200, le=20000, description="Bootstrap resamples; UI default is high; tests may use fewer")
    n_jobs: int = Field(default=1, ge=0, le=64, description="Parallel workers for world evaluation; 0=auto")
    stability_repeats: int = Field(default=0, ge=0, le=30, description="Seed stability repeats; 0 disables")
    stability_worlds: int = Field(default=100, ge=2, le=2000, description="Worlds per seed stability repeat")
    target_a: str | None = Field(
        default=None,
        description="A target: cash|equal_weight|risk_parity|rotation_a|rotation_b",
    )
    target_b: str | None = Field(
        default=None,
        description="B target: cash|equal_weight|risk_parity|rotation_a|rotation_b",
    )
    comparison_mode: str = Field(
        default="custom_ab",
        description="Deprecated compatibility mode: custom_ab|rotation_vs_equal_weight|risk_parity_vs_equal_weight|rotation_vs_risk_parity|equal_weight_vs_cash",
    )
    strategy_a: SimGbmAbStrategyParams = Field(default_factory=SimGbmAbStrategyParams)
    strategy_b: SimGbmAbStrategyParams = Field(default_factory=SimGbmAbStrategyParams)
    holding_strategy_a: SimGbmHoldingStrategyParams = Field(default_factory=SimGbmHoldingStrategyParams)
    holding_strategy_b: SimGbmHoldingStrategyParams = Field(default_factory=SimGbmHoldingStrategyParams)


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
    trade_cost_bps: float = Field(default=2.0, ge=0.0)
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
    trade_cost_bps: float = Field(default=2.0, ge=0.0)
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
    trade_cost_bps: float = Field(default=2.0, ge=0.0)
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
    exec_model: str = Field(default="open_open", description="Execution/return model: open_open|close_close")
    lookback_window: int = Field(default=252, ge=20, le=2520, description="Lookback window for threshold estimation")
    threshold_quantile: float = Field(default=0.80, gt=0.0, lt=1.0, description="Quantile on |index_log_ret| to trigger trades")
    min_abs_ret: float = Field(default=0.0, ge=0.0, description="Hard minimum abs(log-ret) threshold")
    trade_cost_bps: float = Field(default=2.0, ge=0.0, description="Per-switch cost (bps) when position changes")
    initial_position: str = Field(default="long", description="long|cash starting position at start date")
    initial_nav: float = Field(default=1.0, gt=0.0, description="Initial NAV")


class VixSignalBacktestResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    metrics: dict | None = None
    period_returns: dict | None = None
    distributions: dict | None = None
    trades: list[dict] | None = None
    error: str | None = None


class IndexDistributionRequest(BaseModel):
    symbol: str = Field(description="Cboe symbol: GVZ|VXN|VIX|OVX")
    window: str = Field(default="all", description="1y|3y|5y|10y|all")
    bins: int = Field(default=60, ge=10, le=200)
    mode: str = Field(default="raw", description="distribution mode: raw|log")


class IndexDistributionResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
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
    trade_cost_bps: float = Field(default=2.0, ge=0.0, description="Per-switch cost in bps")

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
    exec_prices: list[str] = Field(default_factory=lambda: ["open", "close", "oc2"], description="Execution price list: open|close|oc2 (OC average)")


class CalendarTimingStrategyRequest(BaseModel):
    mode: str = Field(default="portfolio", description="portfolio|single")
    code: str | None = Field(default=None, description="Single-asset mode code")
    codes: list[str] | None = Field(default=None, description="Portfolio mode candidate codes")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="none", description="qfq/hfq/none for execution price series")
    decision_day: int = Field(
        default=1,
        description="Monthly natural decision day in [-28,28] excluding 0. Negative means from month-end.",
    )
    hold_days: int = Field(default=1, ge=1, le=252, description="Holding days from execution day")
    position_mode: str = Field(default="equal", description="equal|fixed_ratio|risk_budget")
    fixed_pos_ratio: float = Field(default=1.0, ge=0.0, le=1.0, description="Exposure when position_mode=fixed_ratio")
    risk_budget_atr_window: int = Field(default=20, ge=2, description="ATR window when position_mode=risk_budget")
    risk_budget_pct: float = Field(default=0.01, ge=0.001, le=0.02, description="Per-asset NAV risk budget for 1 ATR move (0.01 = 1%)")
    dynamic_universe: bool = Field(default=False, description="If true, allow dynamic candidate coverage over union interval")
    exec_price: str = Field(default="open", description="open|close")
    cost_bps: float = Field(default=2.0, ge=0.0, description="Two-way transaction cost in bps")
    slippage_rate: float = Field(default=0.001, ge=0.0, description="One-way adverse slippage spread (absolute price diff)")
    rebalance_shift: str = Field(
        default="prev",
        description="If decision day is non-trading: prev|next|skip",
    )
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    calendar: str = Field(default="XSHG", description="Trading calendar used by decision-day shift")


class AssetRiskControlRule(BaseModel):
    """
    Per-asset risk-control rule applied to weights daily:
    - signal is computed from the asset's own qfq close-based NAV proxy
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
    index: str = Field(description="Vol index code: VIX|GVZ (Cboe) | WAVOL (asset weekly rolling ann vol)")
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


class AssetTrendRule(BaseModel):
    """
    Per-asset trend filter rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    op: str = Field(default=">", description="Comparison operator between close and MA: = | != | > | < | >= | <=")
    stage: str = Field(default="entry", description="Rule stage: entry | exit | both")
    trend_sma_window: int = Field(default=20, ge=1, description="MA window (trading days, qfq close-based)")
    trend_ma_type: str = Field(default="sma", description="MA type: sma|ema|vma(variable/adaptive) (self close vs self MA)")


class AssetBiasRule(BaseModel):
    """
    Per-asset BIAS filter rule (qfq close-based):
    BIAS = close / MA(window) - 1.
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    op: str = Field(default=">", description="Comparison operator: = | != | > | < | >= | <=")
    stage: str = Field(default="entry", description="Rule stage: entry | exit | both")
    bias_type: str = Field(default="bias", description="BIAS signal type: bias|bias_v")
    bias_ma_window: int = Field(default=20, ge=2, description="MA window (trading days)")
    level_window: str = Field(default="all", description="Threshold lookback window: 30d|90d|180d|1y|3y|5y|10y|all(expanding)")
    threshold_type: str = Field(default="quantile", description="quantile|fixed")
    quantile: float = Field(default=95.0, gt=0.0, lt=100.0, description="Percentile value when threshold_type=quantile, e.g. 95")
    fixed_value: float = Field(default=10.0, ge=0.0, description="Fixed threshold (%) when threshold_type=fixed, e.g. 10 means 10%")
    min_periods: int = Field(default=20, ge=2, le=2520, description="Minimum observations before quantile threshold becomes active")


class AssetRsiRule(BaseModel):
    """
    Per-asset RSI filter rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    rsi_window: int = Field(default=14, ge=1, description="RSI window (trading days, fixed to 14)")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0)
    rsi_block_overbought: bool = Field(default=True, description="If true, exclude assets with RSI > overbought")
    rsi_block_oversold: bool = Field(default=False, description="If true, exclude assets with RSI < oversold")


class AssetChopRule(BaseModel):
    """
    Per-asset choppiness filter rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    chop_mode: str = Field(default="er", description="er|adx")
    chop_window: int = Field(default=20, ge=2, description="Efficiency Ratio window (trading days)")
    chop_er_threshold: float = Field(default=0.25, gt=0.0, description="ER < threshold => choppy => exclude")
    chop_adx_window: int = Field(default=20, ge=2, description="ADX window (trading days)")
    chop_adx_threshold: float = Field(default=20.0, gt=0.0, description="ADX < threshold => choppy => exclude")


class AssetVolMonitorRule(BaseModel):
    """
    Per-asset volatility monitor (position sizing) rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    vol_window: int = Field(default=20, ge=1, description="Realized vol window (trading days)")
    vol_target_ann: float = Field(default=0.20, gt=0.0, description="Annualized target vol for sizing")
    vol_max_ann: float = Field(default=0.60, gt=0.0, description="Annualized hard stop vol; above -> no risk position")


class AssetMomentumFloorRule(BaseModel):
    """
    Per-asset momentum condition rule (shared by entry/exit):
    - compares momentum score to threshold with a configurable operator
    - stage controls where the rule applies: entry | exit | both
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    momentum_floor: float | None = Field(
        default=None,
        description="[legacy] momentum threshold in raw score units; used when threshold is not set.",
    )
    threshold: float | None = Field(
        default=None,
        description="Momentum threshold value (raw or pct based on threshold_unit).",
    )
    threshold_unit: str = Field(
        default="raw",
        description="Threshold unit: raw | pct. pct means threshold is percent (1 => 1% => 0.01).",
    )
    op: str = Field(
        default=">",
        description="Comparison operator: = | != | > | < | >= | <= .",
    )
    stage: str = Field(
        default="entry",
        description="Rule stage: entry | exit | both",
    )


class RotationBacktestRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    rebalance: str = Field(default="weekly", description="daily/weekly/monthly/quarterly/yearly")
    rebalance_anchor: int | None = Field(
        default=None,
        description=(
            "Rebalance anchor by frequency: weekly=1..5 (Mon..Fri), "
            "monthly=1..28 (calendar day), quarterly=1..90 (day of quarter), "
            "yearly=1..365 (day of year). daily ignores this."
        ),
    )
    rebalance_shift: str = Field(
        default="prev",
        description=(
            "If anchor falls on non-trading day: prev -> previous trading day (default), "
            "next -> next trading day, skip -> skip this rebalance."
        ),
    )
    exec_price: str = Field(
        default="open",
        description="Execution price for rebalance trading: open|close|oc2 (open/close average).",
    )
    benchmark_mode: str = Field(
        default="EW_REBAL",
        description="Benchmark mode for rotation comparison: EW_REBAL|RP_REBAL|IVOL_REBAL. Default EW_REBAL (compute RP/IVOL only when selected).",
    )
    top_k_mode: str = Field(
        default="fixed",
        description="Top-K selection mode: fixed|floating.",
    )
    floating_benchmark_code: str | None = Field(
        default=None,
        description="Benchmark code for floating top-k mode. Must be one of codes when top_k_mode=floating.",
    )
    top_k: int = Field(
        default=1,
        description="Non-zero integer: top-K by score if positive, bottom-K (inverse) if negative; effective=min(|K|, pool).",
    )
    position_mode: str = Field(
        default="adaptive",
        description="Base position sizing among selected assets: adaptive(equal among selected) | fixed(each uses 1/|top_k|) | risk_budget(ATR risk budget).",
    )
    risk_budget_atr_window: int = Field(default=20, ge=2, description="ATR window for risk-budget sizing")
    risk_budget_pct: float = Field(default=0.01, ge=0.001, le=0.02, description="Per-asset NAV risk budget for 1 ATR move (0.01 = 1%)")
    entry_backfill: bool = Field(
        default=False,
        description="If true, refill from lower-ranked candidates when top_k assets are excluded by entry filters.",
    )
    entry_match_n: int = Field(
        default=0,
        ge=0,
        description=(
            "Entry controls threshold: require at least N enabled entry filters to pass "
            "(0 means default AND of all enabled filters)."
        ),
    )
    exit_match_n: int = Field(
        default=0,
        ge=0,
        description=(
            "Exit controls threshold (reserved): N-of-M setting for exit-control aggregation; "
            "0 keeps current behavior."
        ),
    )
    lookback_days: int = Field(default=20, ge=1)
    skip_days: int = Field(default=0, ge=0)
    score_method: str = Field(
        default="raw_mom",
        description="Ranking score: raw_mom | sharpe_mom | sortino_mom",
    )
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    cost_bps: float = Field(default=2.0, ge=0.0)
    slippage_rate: float = Field(default=0.001, ge=0.0, description="One-way adverse slippage spread (absolute price diff)")
    atr_stop_mode: str = Field(default="none", description="Universal ATR stop mode: none|static|trailing|tightening")
    atr_stop_atr_basis: str = Field(default="latest", description="ATR basis for dynamic modes: entry|latest")
    atr_stop_reentry_mode: str = Field(default="reenter", description="Re-entry after ATR stop: reenter|wait_next_entry")
    atr_stop_window: int = Field(default=14, ge=2, description="ATR window for universal stop")
    atr_stop_n: float = Field(default=2.0, gt=0.0, description="ATR stop distance multiplier n")
    atr_stop_m: float = Field(default=0.5, gt=0.0, description="ATR tightening step m (used by tightening mode)")
    # Pre-trade risk controls (all optional; defaults keep previous behavior)
    trend_filter: bool = Field(default=False, description="Enable trend filter gating (pre-trade)")
    trend_exit_filter: bool = Field(default=False, description="Enable trend-based daily exit gating (post-entry; next-day execution)")
    trend_sma_window: int = Field(default=20, ge=1, description="MA window for trend filter (trading days, qfq close-based)")
    trend_ma_type: str = Field(default="sma", description="Trend MA type: sma|ema|vma(variable/adaptive) (self close vs self MA)")
    bias_filter: bool = Field(default=False, description="Enable BIAS filter gating (pre-trade)")
    bias_exit_filter: bool = Field(default=False, description="Enable BIAS-based daily exit gating (post-entry; next-day execution)")
    bias_type: str = Field(default="bias", description="BIAS signal type: bias|bias_v")
    bias_ma_window: int = Field(default=20, ge=2, description="BIAS MA window (trading days)")
    bias_level_window: str = Field(default="all", description="BIAS threshold lookback window: 30d|90d|180d|1y|3y|5y|10y|all")
    bias_threshold_type: str = Field(default="quantile", description="BIAS threshold type: quantile|fixed")
    bias_quantile: float = Field(default=95.0, gt=0.0, lt=100.0, description="BIAS percentile threshold (0,100), e.g. 95")
    bias_fixed_value: float = Field(default=10.0, ge=0.0, description="BIAS fixed threshold in percent, e.g. 10 means 10%")
    bias_min_periods: int = Field(default=20, ge=2, le=2520, description="Minimum observations for BIAS quantile threshold")
    group_enforce: bool = Field(
        default=False,
        description="Enable hard group constraint: at most one selected asset per group.",
    )
    group_pick_policy: str = Field(
        default="strongest_score",
        description="Group winner policy: strongest_score | earliest_entry | lowest_vol",
    )
    asset_groups: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping: asset code -> group_id. Missing codes default to independent groups.",
    )
    dynamic_universe: bool = Field(
        default=False,
        description="If true, allow dynamic candidate pool by period over union interval.",
    )
    # Phase-1: per-asset parameter rules (optional; if provided, override the corresponding global params)
    asset_momentum_floor_rules: list[AssetMomentumFloorRule] | None = Field(
        default=None,
        description="Optional per-asset momentum-floor rules. Use code='*' as default rule.",
    )
    asset_trend_rules: list[AssetTrendRule] | None = Field(
        default=None,
        description="Optional per-asset trend filter rules (qfq close-based). Use code='*' as default rule.",
    )
    asset_bias_rules: list[AssetBiasRule] | None = Field(
        default=None,
        description="Optional per-asset BIAS filter rules (qfq close-based). Use code='*' as default rule.",
    )
    # Per-asset risk control rules (optional; applied daily to weights as exposure scaling)
    asset_rc_rules: list[AssetRiskControlRule] | None = Field(
        default=None,
        description="Optional per-asset risk-control rules (signal on qfq close-based NAV; scales down weights when triggered).",
    )
    asset_vol_index_rules: list[AssetVolIndexTimingRule] | None = Field(
        default=None,
        description="Optional per-asset vol-index timing rules (e.g. 518880->GVZ, 513100->VIX). Thresholds use rolling/expanding quantiles with shift(1) to avoid lookahead; scales weights daily (cash remainder).",
    )

    @model_validator(mode="after")
    def _validate_rotation_top_k(self) -> RotationBacktestRequest:
        tkm = str(getattr(self, "top_k_mode", "fixed") or "fixed").strip().lower()
        if tkm not in {"fixed", "floating"}:
            raise ValueError("top_k_mode must be one of: fixed|floating")
        if tkm == "fixed":
            if int(self.top_k) == 0:
                raise ValueError("top_k must be non-zero")
        else:
            bm_code = str(getattr(self, "floating_benchmark_code", "") or "").strip()
            if not bm_code:
                raise ValueError("floating_benchmark_code is required when top_k_mode=floating")
            if bm_code not in [str(x) for x in (self.codes or [])]:
                raise ValueError("floating_benchmark_code must be in codes when top_k_mode=floating")
        bm = str(getattr(self, "benchmark_mode", "EW_REBAL") or "EW_REBAL").strip().upper()
        if bm not in {"EW_REBAL", "RP_REBAL", "IVOL_REBAL", "ALL"}:
            raise ValueError("benchmark_mode must be one of: EW_REBAL|RP_REBAL|IVOL_REBAL|ALL")
        return self


class RotationCalendarEffectRequest(RotationBacktestRequest):
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    exec_prices: list[str] = Field(default_factory=lambda: ["open", "close", "oc2"], description="Execution price list: open|close|oc2 (OC average)")


class RotationWeekly5OpenSimRequest(RotationBacktestRequest):
    """
    Mini-program friendly weekly5-open simulation request.

    Notes:
    - Universe is fixed to 4 ETFs (159915/511010/513100/518880). The API will ignore provided codes.
    - Execution is on open, and rebalance_shift is effectively forced to 'prev' due to open-exec semantics.
    - All retained rotation parameters（基础参数 + 动量/趋势/乖离 + 恐慌择时）均可用。
    """

    # Keep backward compatibility: allow omitting codes in clients.
    codes: list[str] = Field(
        default_factory=lambda: ["159915", "511010", "513100", "518880"],
        description="(Ignored by server) Fixed universe. Included for schema compatibility.",
        min_length=1,
    )
    anchor_weekday: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optional: if set, compute only one execution weekday (1=Mon..5=Fri) to reduce payload/runtime.",
    )


class RotationNextPlanRequest(BaseModel):
    """
    Next rebalance plan for the fixed 4-ETF mini-program strategy (weekly, top1, lookback20, open execution).
    Used by the mini-program to show "tomorrow plan" when tomorrow is a rebalance effective day.
    """

    anchor_weekday: int = Field(ge=1, le=5, description="1=Mon..5=Fri")
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
        ge=1,
        le=5,
        description="Optional: if set, compute only one anchor weekday (1=Mon..5=Fri) to reduce payload/runtime.",
    )


class RTakeProfitTier(BaseModel):
    r_multiple: float = Field(gt=0.0, description="Activate drawdown take-profit when peak floating profit reaches this R multiple")
    retrace_ratio: float = Field(gt=0.0, lt=1.0, description="Allowed pullback ratio from peak floating profit once activated")


class TrendBacktestRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32, description="Single ETF code for trend-following backtest")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(
        default="hfq",
        description="[deprecated] Trend research uses mixed basis like rotation: signal=qfq, nav=none with hfq fallback, benchmark=hfq.",
    )
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    cost_bps: float = Field(default=2.0, ge=0.0, description="Round-trip transaction cost in bps per turnover")
    slippage_rate: float = Field(default=0.001, ge=0.0, description="One-way adverse slippage spread (absolute price diff)")
    exec_price: str = Field(default="open", description="open|close|oc2")
    strategy: str = Field(
        default="ma_filter",
        description="ma_filter|ma_cross|donchian|tsmom|linreg_slope|bias|macd_cross|macd_zero_filter|macd_v|random_entry (long/cash); ma_filter uses ma_type sma|ema|kama",
    )
    position_sizing: str = Field(default="equal", description="equal|vol_target|fixed_ratio|risk_budget")
    vol_window: int = Field(default=20, ge=2, description="Rolling vol window for vol-target sizing")
    vol_target_ann: float = Field(default=0.20, gt=0.0, description="Annualized target vol for portfolio scaling")
    fixed_pos_ratio: float = Field(default=0.04, gt=0.0, description="Fixed position ratio when position_sizing=fixed_ratio")
    fixed_overcap_policy: str = Field(default="skip", description="Over-cap policy placeholder: skip|extend")
    fixed_max_holdings: int = Field(default=10, ge=1, description="Max holdings placeholder for unified payload schema")
    risk_budget_atr_window: int = Field(default=20, ge=2, description="ATR window when position_sizing=risk_budget")
    risk_budget_pct: float = Field(default=0.01, ge=0.001, le=0.02, description="NAV risk budget for 1 ATR move (0.01 = 1%)")
    risk_budget_overcap_policy: str = Field(default="scale", description="When risk-budget new entry exceeds total 100% exposure: scale|skip_entry|replace_entry|leverage_entry")
    risk_budget_max_leverage_multiple: float = Field(default=2.0, ge=1.0, le=10.0, description="Max leverage multiple when risk_budget_overcap_policy=leverage_entry")
    vol_regime_risk_mgmt_enabled: bool = Field(default=False, description="Enable post-entry volatility-regime risk management in risk_budget sizing")
    vol_ratio_fast_atr_window: int = Field(default=5, ge=2, description="Fast ATR window for volatility ratio ATR(fast)/ATR(slow)")
    vol_ratio_slow_atr_window: int = Field(default=50, ge=2, description="Slow ATR window for volatility ratio ATR(fast)/ATR(slow)")
    vol_ratio_expand_threshold: float = Field(default=1.45, gt=0.0, description="If ATR ratio > threshold, volatility is expanded and de-risk is triggered")
    vol_ratio_contract_threshold: float = Field(default=0.65, gt=0.0, description="If ATR ratio < threshold, volatility is contracted and add-risk is triggered")
    vol_ratio_normal_threshold: float = Field(default=1.05, gt=0.0, description="Normal-zone recovery threshold used after expanded/contracted states")
    # parameters (some are strategy-specific)
    sma_window: int = Field(default=200, ge=2, description="MA filter window (trading days)")
    fast_window: int = Field(default=50, ge=2, description="Fast MA window (trading days)")
    slow_window: int = Field(default=200, ge=2, description="Slow MA window (trading days)")
    ma_type: str = Field(default="sma", description="MA type: ma_filter supports sma|ema|kama; ma_cross supports sma|ema")
    kama_er_window: int = Field(default=10, ge=2, description="KAMA ER lookback window (used when ma_type=kama)")
    kama_fast_window: int = Field(default=2, ge=1, description="KAMA fast smoothing window (used when ma_type=kama)")
    kama_slow_window: int = Field(default=30, ge=2, description="KAMA slow smoothing window (used when ma_type=kama)")
    kama_std_window: int = Field(default=20, ge=2, description="KAMA std window for hysteresis band: KAMA ± coef*std(KAMA)")
    kama_std_coef: float = Field(default=1.0, ge=0.0, le=3.0, description="KAMA std filter coefficient in [0,3]")
    donchian_entry: int = Field(default=20, ge=2, description="Donchian entry window (trading days)")
    donchian_exit: int = Field(default=10, ge=2, description="Donchian exit window (trading days)")
    mom_lookback: int = Field(default=252, ge=2, description="TS momentum lookback (trading days)")
    tsmom_entry_threshold: float = Field(default=0.0, description="TSMOM entry threshold on momentum score; enter when score > threshold")
    tsmom_exit_threshold: float = Field(default=0.0, description="TSMOM exit threshold on momentum score; exit when score <= threshold")
    atr_stop_mode: str = Field(default="none", description="Universal ATR stop mode: none|static|trailing|tightening")
    atr_stop_atr_basis: str = Field(default="latest", description="ATR basis for dynamic modes: entry|latest")
    atr_stop_reentry_mode: str = Field(default="reenter", description="Re-entry after ATR stop: reenter|wait_next_entry")
    atr_stop_window: int = Field(default=14, ge=2, description="ATR window for universal stop")
    atr_stop_n: float = Field(default=2.0, gt=0.0, description="ATR stop distance multiplier n")
    atr_stop_m: float = Field(default=0.5, gt=0.0, description="ATR tightening step m (used by tightening mode)")
    r_take_profit_enabled: bool = Field(default=False, description="Enable universal R-multiple drawdown take-profit overlay")
    r_take_profit_reentry_mode: str = Field(default="reenter", description="Re-entry after R take-profit: reenter|wait_next_entry")
    r_take_profit_tiers: list[RTakeProfitTier] | None = Field(
        default=None,
        description="Tiered config: peak>=R multiple activates pullback-exit threshold, e.g. [{r_multiple:2,retrace_ratio:0.5}]",
    )
    bias_v_take_profit_enabled: bool = Field(default=False, description="Enable universal BIAS-V take-profit overlay")
    bias_v_take_profit_reentry_mode: str = Field(default="reenter", description="Re-entry after BIAS-V take-profit: reenter|wait_next_entry")
    bias_v_ma_window: int = Field(default=20, ge=2, description="MA window in BIAS-V=(close-MA)/ATR")
    bias_v_atr_window: int = Field(default=20, ge=2, description="ATR window in BIAS-V=(close-MA)/ATR")
    bias_v_take_profit_threshold: float = Field(default=5.0, gt=0.0, description="Trigger BIAS-V take-profit when BIAS-V >= threshold")
    monthly_risk_budget_enabled: bool = Field(default=False, description="Enable account-level monthly max-loss risk budget gate before new entries")
    monthly_risk_budget_pct: float = Field(default=0.06, ge=0.01, le=0.06, description="Monthly max-loss budget on account NAV (0.06 = 6%)")
    monthly_risk_budget_include_new_trade_risk: bool = Field(default=False, description="If true, include candidate new-trade risk in monthly budget check")
    # BIAS strategy params
    bias_ma_window: int = Field(default=20, ge=2, description="EMA window N in BIAS=(LN(C)-LN(EMA(C,N)))*100 (trading days)")
    bias_entry: float = Field(default=2.0, description="Enter when BIAS > entry (percent)")
    bias_hot: float = Field(default=10.0, description="Take-profit exit when BIAS >= hot (percent)")
    bias_cold: float = Field(default=-2.0, description="Stop-loss exit when BIAS <= cold (percent)")
    bias_pos_mode: str = Field(default="binary", description="Position mode for BIAS strategy: binary|continuous")
    macd_fast: int = Field(default=12, ge=2, description="MACD fast EMA window")
    macd_slow: int = Field(default=26, ge=2, description="MACD slow EMA window")
    macd_signal: int = Field(default=9, ge=2, description="MACD signal EMA window")
    macd_v_atr_window: int = Field(default=26, ge=2, description="ATR window used by MACD-V normalization")
    macd_v_scale: float = Field(default=100.0, gt=0.0, description="Scale factor for MACD-V")
    er_filter: bool = Field(default=False, description="Universal ER entry filter switch (when true, allow entry only if ER >= threshold)")
    er_window: int = Field(default=10, ge=2, description="ER lookback window (trading days)")
    er_threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="ER entry threshold in [0,1]")
    impulse_entry_filter: bool = Field(default=False, description="Universal Impulse entry filter switch (Elder Impulse System)")
    impulse_allow_bull: bool = Field(default=True, description="Allow new long entries in BULL impulse state")
    impulse_allow_bear: bool = Field(default=False, description="Allow new long entries in BEAR impulse state")
    impulse_allow_neutral: bool = Field(default=False, description="Allow new long entries in NEUTRAL impulse state")
    er_exit_filter: bool = Field(default=False, description="Universal ER exit filter switch (when true, exit if ER >= threshold)")
    er_exit_window: int = Field(default=10, ge=2, description="ER exit filter lookback window (trading days)")
    er_exit_threshold: float = Field(default=0.88, ge=0.0, le=1.0, description="ER exit threshold in [0,1]")
    random_hold_days: int = Field(default=20, ge=1, description="Random-entry strategy base exit: hold N trading days after entry")
    random_seed: int | None = Field(default=42, description="Random-entry strategy seed for reproducible coin-toss signals; null means system random seed")


class TrendPortfolioBacktestRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="Portfolio candidate codes")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    risk_free_rate: float = Field(default=0.025, description="Annualized rf (decimal)")
    cost_bps: float = Field(default=2.0, ge=0.0, description="Round-trip transaction cost in bps per turnover")
    slippage_rate: float = Field(default=0.001, ge=0.0, description="One-way adverse slippage spread (absolute price diff)")
    exec_price: str = Field(default="open", description="open|close|oc2")
    strategy: str = Field(
        default="ma_filter",
        description="ma_filter|ma_cross|donchian|tsmom|linreg_slope|bias|macd_cross|macd_zero_filter|macd_v|random_entry; ma_filter uses ma_type sma|ema|kama",
    )
    position_sizing: str = Field(default="equal", description="equal|vol_target|fixed_ratio|risk_budget")
    vol_window: int = Field(default=20, ge=2, description="Rolling vol window for vol-target sizing")
    vol_target_ann: float = Field(default=0.20, gt=0.0, description="Annualized target vol for portfolio scaling")
    fixed_pos_ratio: float = Field(default=0.04, gt=0.0, description="Fixed position ratio per active asset when position_sizing=fixed_ratio")
    fixed_overcap_policy: str = Field(default="skip", description="When fixed-ratio entry exceeds constraints: skip|extend")
    fixed_max_holdings: int = Field(default=10, ge=1, description="Max number of held assets when position_sizing=fixed_ratio")
    risk_budget_atr_window: int = Field(default=20, ge=2, description="ATR window when position_sizing=risk_budget")
    risk_budget_pct: float = Field(default=0.01, ge=0.001, le=0.02, description="Per-asset NAV risk budget for 1 ATR move (0.01 = 1%)")
    risk_budget_overcap_policy: str = Field(default="scale", description="When risk-budget new entry exceeds total 100% exposure: scale|skip_entry|replace_entry|leverage_entry")
    risk_budget_max_leverage_multiple: float = Field(default=2.0, ge=1.0, le=10.0, description="Max leverage multiple when risk_budget_overcap_policy=leverage_entry")
    vol_regime_risk_mgmt_enabled: bool = Field(default=False, description="Enable post-entry volatility-regime risk management in risk_budget sizing")
    vol_ratio_fast_atr_window: int = Field(default=5, ge=2, description="Fast ATR window for volatility ratio ATR(fast)/ATR(slow)")
    vol_ratio_slow_atr_window: int = Field(default=50, ge=2, description="Slow ATR window for volatility ratio ATR(fast)/ATR(slow)")
    vol_ratio_expand_threshold: float = Field(default=1.45, gt=0.0, description="If ATR ratio > threshold, volatility is expanded and de-risk is triggered")
    vol_ratio_contract_threshold: float = Field(default=0.65, gt=0.0, description="If ATR ratio < threshold, volatility is contracted and add-risk is triggered")
    vol_ratio_normal_threshold: float = Field(default=1.05, gt=0.0, description="Normal-zone recovery threshold used after expanded/contracted states")
    dynamic_universe: bool = Field(default=False, description="If true, allow dynamic candidate pool by period over union interval")
    sma_window: int = Field(default=200, ge=2)
    fast_window: int = Field(default=50, ge=2)
    slow_window: int = Field(default=200, ge=2)
    ma_type: str = Field(default="sma", description="MA type: ma_filter supports sma|ema|kama; ma_cross supports sma|ema")
    kama_er_window: int = Field(default=10, ge=2, description="KAMA ER lookback window (used when ma_type=kama)")
    kama_fast_window: int = Field(default=2, ge=1, description="KAMA fast smoothing window (used when ma_type=kama)")
    kama_slow_window: int = Field(default=30, ge=2, description="KAMA slow smoothing window (used when ma_type=kama)")
    kama_std_window: int = Field(default=20, ge=2, description="KAMA std window for hysteresis band: KAMA ± coef*std(KAMA)")
    kama_std_coef: float = Field(default=1.0, ge=0.0, le=3.0, description="KAMA std filter coefficient in [0,3]")
    donchian_entry: int = Field(default=20, ge=2)
    donchian_exit: int = Field(default=10, ge=2)
    mom_lookback: int = Field(default=252, ge=2)
    tsmom_entry_threshold: float = Field(default=0.0)
    tsmom_exit_threshold: float = Field(default=0.0)
    atr_stop_mode: str = Field(default="none", description="none|static|trailing|tightening")
    atr_stop_atr_basis: str = Field(default="latest", description="entry|latest")
    atr_stop_reentry_mode: str = Field(default="reenter", description="reenter|wait_next_entry")
    atr_stop_window: int = Field(default=14, ge=2)
    atr_stop_n: float = Field(default=2.0, gt=0.0)
    atr_stop_m: float = Field(default=0.5, gt=0.0)
    r_take_profit_enabled: bool = Field(default=False, description="Enable universal R-multiple drawdown take-profit overlay")
    r_take_profit_reentry_mode: str = Field(default="reenter", description="reenter|wait_next_entry")
    r_take_profit_tiers: list[RTakeProfitTier] | None = Field(default=None)
    bias_v_take_profit_enabled: bool = Field(default=False, description="Enable universal BIAS-V take-profit overlay")
    bias_v_take_profit_reentry_mode: str = Field(default="reenter", description="reenter|wait_next_entry")
    bias_v_ma_window: int = Field(default=20, ge=2, description="MA window in BIAS-V=(close-MA)/ATR")
    bias_v_atr_window: int = Field(default=20, ge=2, description="ATR window in BIAS-V=(close-MA)/ATR")
    bias_v_take_profit_threshold: float = Field(default=5.0, gt=0.0, description="Trigger BIAS-V take-profit when BIAS-V >= threshold")
    monthly_risk_budget_enabled: bool = Field(default=False, description="Enable account-level monthly max-loss risk budget gate before new entries")
    monthly_risk_budget_pct: float = Field(default=0.06, ge=0.01, le=0.06, description="Monthly max-loss budget on account NAV (0.06 = 6%)")
    monthly_risk_budget_include_new_trade_risk: bool = Field(default=False, description="If true, include candidate new-trade risk in monthly budget check")
    bias_ma_window: int = Field(default=20, ge=2)
    bias_entry: float = Field(default=2.0)
    bias_hot: float = Field(default=10.0)
    bias_cold: float = Field(default=-2.0)
    bias_pos_mode: str = Field(default="binary", description="binary|continuous")
    macd_fast: int = Field(default=12, ge=2)
    macd_slow: int = Field(default=26, ge=2)
    macd_signal: int = Field(default=9, ge=2)
    macd_v_atr_window: int = Field(default=26, ge=2)
    macd_v_scale: float = Field(default=100.0, gt=0.0)
    er_filter: bool = Field(default=False, description="Universal ER entry filter switch")
    er_window: int = Field(default=10, ge=2)
    er_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    impulse_entry_filter: bool = Field(default=False, description="Universal Impulse entry filter switch (Elder Impulse System)")
    impulse_allow_bull: bool = Field(default=True, description="Allow new long entries in BULL impulse state")
    impulse_allow_bear: bool = Field(default=False, description="Allow new long entries in BEAR impulse state")
    impulse_allow_neutral: bool = Field(default=False, description="Allow new long entries in NEUTRAL impulse state")
    er_exit_filter: bool = Field(default=False, description="Universal ER exit filter switch")
    er_exit_window: int = Field(default=10, ge=2)
    er_exit_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    random_hold_days: int = Field(default=20, ge=1)
    random_seed: int | None = Field(default=42)
    group_enforce: bool = Field(
        default=False,
        description="Enable trend group constraint for portfolio mode.",
    )
    group_pick_policy: str = Field(
        default="highest_sharpe",
        description="Trend group winner policy: earliest_entry | highest_sharpe",
    )
    group_max_holdings: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Max number of selected assets per group when group_enforce=true.",
    )
    asset_groups: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping: asset code -> group_id. Missing codes default to independent groups.",
    )


class AssetGroupSuggestRequest(BaseModel):
    codes: list[str] = Field(min_length=2, description="Candidate codes for auto grouping")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="Price adjust basis for correlation clustering")
    lookback_days: int = Field(default=252, ge=20, description="Rolling lookback days for correlation matrix")
    corr_threshold: float = Field(default=0.75, ge=0.0, le=0.99, description="Absolute correlation threshold for linking two assets")


class RotationCandidateScreenRequest(BaseModel):
    codes: list[str] = Field(min_length=2, description="Candidate codes from preset pool")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="Price adjust basis")
    lookback_days: int = Field(default=252, ge=20, le=2520, description="Lookback window for scoring/correlation")
    top_n: int = Field(default=12, ge=2, le=200, description="Max number of selected assets")
    min_n: int = Field(default=4, ge=1, le=200, description="Minimum selected assets (fallback fill by score)")
    max_pair_corr: float = Field(default=0.75, ge=0.0, le=0.99, description="Max absolute pairwise correlation among selected assets")
    signif_horizon_days: int = Field(default=20, ge=5, le=252, description="Forward horizon for momentum significance test")
    factor_weights: dict[str, float] | None = Field(
        default=None,
        description="Optional factor weights, keys: mom_63,mom_126,sharpe,win_rate,liquidity,mdd",
    )
    category_quotas: dict[str, int] | None = Field(
        default=None,
        description="Optional minimum quota by inferred category, e.g. {'CN_EQ':2,'US_EQ':2}",
    )


class MonteCarloRequest(BaseModel):
    n_sims: int = Field(default=10000, ge=50, le=50000, description="Number of Monte Carlo simulations")
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


class RotationOosBootstrapRequest(BaseModel):
    """Request for out-of-sample bootstrap parameter optimisation (Carver-style)."""

    codes: list[str] = Field(min_length=1, description="Universe codes")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    oos_ratio: float = Field(default=0.3, gt=0.0, lt=1.0, description="Fraction of period for OOS (at end)")
    n_bootstrap: int = Field(default=50, ge=5, le=500, description="Number of bootstrap resamples")
    block_size: int = Field(default=21, ge=1, description="Block size for circular block bootstrap (trading days)")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    cost_bps: float = Field(default=3.0, ge=0.0)
    param_grid: dict[str, list[Any]] | None = Field(
        default=None,
        description=(
            "Optional param grid, e.g. {'lookback_days': [60,90,120], 'top_k': [1,2]}. "
            "If omitted, a default grid is used."
        ),
    )


class TrendOosBootstrapRequest(BaseModel):
    """Request for out-of-sample bootstrap parameter optimisation for trend (portfolio) strategies."""

    codes: list[str] = Field(min_length=1, description="Portfolio codes")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    oos_ratio: float = Field(default=0.3, gt=0.0, lt=1.0, description="Fraction of period for OOS (at end)")
    n_bootstrap: int = Field(default=50, ge=5, le=500, description="Number of bootstrap resamples")
    block_size: int = Field(default=21, ge=1, description="Block size for circular block bootstrap (trading days)")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    strategy: str = Field(
        default="ma_filter",
        description="ma_filter|ma_cross|donchian|tsmom|linreg_slope|bias|macd_cross|macd_zero_filter|macd_v|random_entry",
    )
    cost_bps: float = Field(default=2.0, ge=0.0)
    risk_free_rate: float = Field(default=0.025)
    exec_price: str = Field(default="open", description="open|close|oc2")
    param_grid: dict[str, list[Any]] | None = Field(
        default=None,
        description="Optional param grid per strategy; if omitted, a default grid is used.",
    )

