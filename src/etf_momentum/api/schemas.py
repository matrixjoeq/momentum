from __future__ import annotations

import math

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


class EtfResearchGroupUpsert(BaseModel):
    name: str = Field(min_length=1, max_length=128, description="Group name")
    codes: list[str] = Field(default_factory=list, description="ETF symbols in group")
    set_active: bool = Field(
        default=True, description="If true, set this group as current active group"
    )


class EtfResearchGroupOut(BaseModel):
    name: str
    codes: list[str]
    is_active: bool


class EtfResearchGroupsImportRequest(BaseModel):
    groups: dict[str, list[str]] = Field(
        description="Mapping: group name -> symbol list"
    )
    active_group: str | None = Field(
        default=None, description="Optional active group name"
    )
    replace_all: bool = Field(
        default=False,
        description="If true, remove DB groups not present in incoming payload first",
    )


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


class GlobalBenchmarkPoolUpsert(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    series_kind: str | None = Field(default=None, max_length=32)
    code_format: str | None = Field(default=None, max_length=32)
    provider_hint: str | None = Field(default=None, max_length=32)
    provider_symbol: str | None = Field(default=None, max_length=64)
    source_locked: bool | None = None
    fallback_sources: list[dict[str, str]] | None = None
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")


class GlobalBenchmarkSeriesOut(BaseModel):
    series_kind: str
    code_format: str | None = None
    provider_hint: str | None = None
    provider_symbol: str | None = None
    source_locked: bool = False
    start_date: str | None
    end_date: str | None
    last_fetch_status: str | None = None
    last_fetch_message: str | None = None
    last_data_start_date: str | None = None
    last_data_end_date: str | None = None


class GlobalBenchmarkPoolOut(BaseModel):
    code: str
    name: str
    series: list[GlobalBenchmarkSeriesOut] = Field(default_factory=list)


class GlobalBenchmarkPriceOut(BaseModel):
    code: str
    series_kind: str
    trade_date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    amount: float | None
    source: str
    adjust: str


class GlobalBenchmarkFetchResult(BaseModel):
    code: str
    series_kind: str
    inserted_or_updated: int
    status: str
    message: str | None = None
    code_format: str | None = None
    final_provider: str | None = None
    final_symbol: str | None = None
    provider_attempts: list[dict[str, Any]] = Field(default_factory=list)


class GlobalBenchmarkFetchSelectedRequest(BaseModel):
    codes: list[str] = Field(
        min_length=1, description="Global benchmark codes to fetch"
    )


class GlobalBenchmarkDefaultInstallRequest(BaseModel):
    overwrite_existing: bool = False


class GlobalBenchmarkDefaultInstallItem(BaseModel):
    code: str
    name: str
    series_kind: str
    action: str


class GlobalBenchmarkDefaultInstallResponse(BaseModel):
    ok: bool
    total: int
    inserted: int
    updated: int
    skipped: int
    items: list[GlobalBenchmarkDefaultInstallItem] = Field(default_factory=list)


class GlobalBenchmarkDefaultAcceptanceRequest(BaseModel):
    codes: list[str] | None = Field(default=None, description="Subset of default codes")
    fetch: bool = True
    continue_on_error: bool = True


class GlobalBenchmarkDefaultAcceptanceSeriesItem(BaseModel):
    series_kind: str
    status: str
    message: str | None = None
    final_provider: str | None = None
    final_symbol: str | None = None
    sample_days: int = 0
    data_start_date: str | None = None
    data_end_date: str | None = None


class GlobalBenchmarkDefaultAcceptanceItem(BaseModel):
    code: str
    name: str
    status: str
    failure_reason: str | None = None
    series: list[GlobalBenchmarkDefaultAcceptanceSeriesItem] = Field(
        default_factory=list
    )


class GlobalBenchmarkDefaultAcceptanceResponse(BaseModel):
    ok: bool
    total: int
    succeeded: int
    failed: int
    skipped: int
    items: list[GlobalBenchmarkDefaultAcceptanceItem] = Field(default_factory=list)


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


class OffFundResearchStateUpdate(BaseModel):
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")
    adjust: Literal["hfq", "qfq", "none"] = Field(default="hfq")
    rf: float = Field(default=0.025, ge=-1.0, le=1.0)
    inner_mode: Literal["risk_parity_cov", "equal", "custom"] = Field(
        default="risk_parity_cov"
    )
    rp_window: int = Field(default=60, ge=20, le=2000)
    rebalance_cycle: Literal[
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
        "none",
    ] = Field(default="daily")
    drift_rebalance_enabled: bool = Field(default=True)
    drift_abs_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    drift_rel_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    pair_chart_prefs_json: str | None = Field(default=None)


class OffFundResearchStateMeta(BaseModel):
    contract_version: str = "pair_contract_v1"
    warnings: list[str] = Field(default_factory=list)


class OffFundResearchStateOut(BaseModel):
    start_date: str | None = "20110210"
    end_date: str | None = None
    adjust: str = "hfq"
    rf: float = 0.025
    inner_mode: str = "risk_parity_cov"
    rp_window: int = 60
    rebalance_cycle: str = "daily"
    drift_rebalance_enabled: bool = True
    drift_abs_threshold: float = 0.05
    drift_rel_threshold: float = 0.25
    pair_chart_prefs_json: str | None = None
    meta: OffFundResearchStateMeta = Field(default_factory=OffFundResearchStateMeta)


class OffFundRegressionFactorRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    label: str | None = Field(default=None, max_length=128)
    aliases: list[str] = Field(
        min_length=1, description="Candidate benchmark codes in priority order"
    )


class OffFundRegressionFactorConfigUpsert(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    set_active: bool = Field(default=True)
    benchmark_profile: str = Field(
        default="cn_stock_core",
        description='Built-in profile. current: "cn_stock_core"',
    )
    benchmark_factors: list[OffFundRegressionFactorRequest] | None = Field(
        default=None,
        description="Optional custom factors; when set, overrides benchmark_profile",
    )


class OffFundRegressionFactorConfigOut(BaseModel):
    name: str
    is_active: bool
    benchmark_profile: str
    benchmark_factors: list[OffFundRegressionFactorRequest] = Field(
        default_factory=list
    )
    effective_benchmark_factors: list[OffFundRegressionFactorRequest] = Field(
        default_factory=list,
        description=(
            "Resolved factor list used by backend: custom factors if provided; "
            "otherwise factors expanded from benchmark_profile."
        ),
    )


class OffFundRegressionClassifyRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="Off-fund codes to classify")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    fund_adjust: str = Field(default="hfq", description="Off-fund nav adjust basis")
    benchmark_adjust: str = Field(
        default="hfq", description="Benchmark close adjust basis"
    )
    benchmark_profile: str = Field(
        default="cn_stock_core",
        description='Built-in profile. current: "cn_stock_core"',
    )
    benchmark_factors: list[OffFundRegressionFactorRequest] | None = Field(
        default=None,
        description="Optional custom factors; when set, overrides benchmark_profile",
    )
    rolling_window: int = Field(default=252, ge=40, le=2000)
    min_samples: int = Field(default=120, ge=40, le=2000)
    dominance_gap: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="If top1-top2 exposure < gap, classify as balanced style",
    )
    include_exposure_series: bool = Field(default=False)
    max_series_points: int = Field(default=260, ge=0, le=2000)


class OffFundRegressionFactorAvailabilityRequest(BaseModel):
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    benchmark_adjust: str = Field(
        default="hfq", description="Benchmark close adjust basis"
    )
    benchmark_profile: str = Field(
        default="cn_stock_core",
        description='Built-in profile. current: "cn_stock_core"',
    )
    benchmark_factors: list[OffFundRegressionFactorRequest] | None = Field(
        default=None,
        description="Optional custom factors; when set, overrides benchmark_profile",
    )
    rolling_window: int = Field(default=252, ge=40, le=2000)
    min_samples: int = Field(default=120, ge=40, le=2000)


class OffFundRegressionSeriesPoint(BaseModel):
    trade_date: str
    r2: float | None = None
    exposures: dict[str, float] = Field(default_factory=dict)


class OffFundRegressionClassifyItem(BaseModel):
    code: str
    name: str | None = None
    status: str
    sample_days: int
    effective_windows: int | None = None
    avg_r2: float | None = None
    latest_r2: float | None = None
    label: str
    confidence: str
    primary_asset_class: str
    avg_exposures: dict[str, float] = Field(default_factory=dict)
    latest_exposures: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    exposure_series: list[OffFundRegressionSeriesPoint] = Field(default_factory=list)


class OffFundRegressionClassifyResponse(BaseModel):
    ok: bool
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    factors: list[dict[str, Any]] = Field(default_factory=list)
    items: list[OffFundRegressionClassifyItem] = Field(default_factory=list)


class OffFundRegressionFactorAvailabilityItem(BaseModel):
    key: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    selected_code: str | None = None
    sample_days: int
    required_days: int
    enough: bool
    status: str
    alias_samples: dict[str, int] = Field(default_factory=dict)


class OffFundRegressionFactorAvailabilityResponse(BaseModel):
    ok: bool
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    items: list[OffFundRegressionFactorAvailabilityItem] = Field(default_factory=list)


class FuturesPoolUpsert(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")
    min_margin_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum margin ratio, decimal in [0,1]",
    )
    contract_multiplier: float | None = Field(
        default=None, gt=0.0, description="Contract multiplier (trading unit)"
    )
    price_unit: str | None = Field(
        default=None, max_length=64, description="Quoted price unit, e.g. 元/吨"
    )
    min_price_tick: float | None = Field(
        default=None, gt=0.0, description="Minimum tick size"
    )
    tags: list[str] | None = Field(
        default=None, description="Optional tags; empty means auto category tag"
    )
    contract_extend_calendar_days: int | None = Field(
        default=None,
        ge=1,
        le=5000,
        description="Calendar days to extend beyond local main continuous end when fetching deliverable contracts",
    )
    contract_parallel: int | None = Field(
        default=None,
        ge=1,
        le=1,
        description="Reserved for compatibility; deliverable-month fetch is fixed to serial (1) for AkShare",
    )


class FuturesPoolOut(BaseModel):
    code: str
    name: str
    start_date: str | None
    end_date: str | None
    min_margin_ratio: float | None = None
    contract_multiplier: float | None = None
    price_unit: str | None = None
    min_price_tick: float | None = None
    tags: list[str] = Field(default_factory=list)
    contract_extend_calendar_days: int = 366
    contract_parallel: int = 1
    last_fetch_status: str | None = None
    last_fetch_message: str | None = None
    last_data_start_date: str | None = None
    last_data_end_date: str | None = None
    last_contract_fetch_status: str | None = None
    last_contract_fetch_message: str | None = None


class FuturesContractFetchStatusOut(BaseModel):
    contract_code: str
    last_fetch_status: str | None = None
    last_fetch_message: str | None = None
    rows_upserted: int | None = None
    last_data_end_date: str | None = None


class FuturesFetchResult(BaseModel):
    code: str
    inserted_or_updated: int
    status: str
    message: str | None = None


class FuturesFetchRequest(BaseModel):
    fetch_type: Literal["incremental", "full"] = Field(
        default="incremental",
        description='Fetch mode: "incremental" (fallback to full when no local data) or "full".',
    )


class FuturesFetchAllRequest(BaseModel):
    fetch_type: Literal["incremental", "full"] = Field(
        default="incremental",
        description='Batch fetch mode: "incremental" or "full".',
    )


class FuturesFetchSelectedRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="futures symbols to fetch")
    fetch_type: Literal["incremental", "full"] = Field(
        default="incremental",
        description='Fetch mode for selected symbols: "incremental" or "full".',
    )


class FuturesSynthesisValidationItemOut(BaseModel):
    code: str
    status: Literal["passed", "failed", "skipped"]
    conclusion: str
    details: dict[str, Any] = Field(default_factory=dict)


class FuturesSynthesisValidationAllOut(BaseModel):
    total: int
    passed: int
    failed: int
    skipped: int
    items: list[FuturesSynthesisValidationItemOut] = Field(default_factory=list)


class FuturesSynthesisValidationRequest(BaseModel):
    rel_mean_max: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Maximum mean absolute percentage error (MAPE) for usability pass",
    )
    rel_p95_max: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Maximum 95th percentile absolute percentage error (P95 APE) for usability pass",
    )
    auto_correct: bool = Field(
        default=True,
        description="Auto-correct synthesized 88 field points exceeding P95_APE threshold using main0 values, then rebuild 888/889.",
    )


class FuturesPriceOut(BaseModel):
    code: str
    trade_date: str  # YYYY-MM-DD
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    settle: float | None
    volume: float | None
    amount: float | None
    hold: float | None
    dominant_contract_suffix: str | None = None
    source: str
    adjust: str


class FuturesResearchGroupUpsert(BaseModel):
    name: str = Field(min_length=1, max_length=128, description="Group name")
    codes: list[str] = Field(
        default_factory=list, description="Futures symbols in group"
    )
    set_active: bool = Field(
        default=True, description="If true, set this group as current active group"
    )


class FuturesResearchGroupOut(BaseModel):
    name: str
    codes: list[str]
    is_active: bool


class FuturesResearchGroupsImportRequest(BaseModel):
    groups: dict[str, list[str]] = Field(
        description="Mapping: group name -> symbol list"
    )
    active_group: str | None = Field(
        default=None, description="Optional active group name"
    )


class FuturesResearchStateUpdate(BaseModel):
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")
    dynamic_universe: bool = Field(default=True)
    quick_range_key: str = Field(default="all", description="1m|3m|6m|1y|3y|5y|10y|all")


class FuturesResearchStateOut(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    dynamic_universe: bool = True
    quick_range_key: str = "all"
    active_group: str | None = None


class FuturesCorrelationRequest(BaseModel):
    group_name: str | None = Field(
        default=None, description="If null, use active group"
    )
    range_key: str = Field(default="all", description="1m|3m|6m|1y|3y|5y|10y|all")
    start_date: str | None = Field(
        default=None, description="Optional explicit YYYYMMDD"
    )
    end_date: str | None = Field(default=None, description="Optional explicit YYYYMMDD")
    dynamic_universe: bool | None = Field(
        default=None, description="If null, use saved global setting"
    )
    min_obs: int = Field(default=20, ge=2, le=2520)


class FuturesCoverageSummaryRequest(BaseModel):
    group_name: str | None = Field(
        default=None, description="If null, use active group"
    )
    range_key: str = Field(default="all", description="1m|3m|6m|1y|3y|5y|10y|all")
    start_date: str | None = Field(
        default=None, description="Optional explicit YYYYMMDD"
    )
    end_date: str | None = Field(default=None, description="Optional explicit YYYYMMDD")
    dynamic_universe: bool | None = Field(
        default=None, description="If null, use saved global setting"
    )


class FuturesCorrelationSelectRequest(BaseModel):
    group_name: str | None = Field(
        default=None, description="If null, use active group"
    )
    range_key: str = Field(default="all", description="1m|3m|6m|1y|3y|5y|10y|all")
    start_date: str | None = Field(
        default=None, description="Optional explicit YYYYMMDD"
    )
    end_date: str | None = Field(default=None, description="Optional explicit YYYYMMDD")
    dynamic_universe: bool | None = Field(
        default=None, description="If null, use saved global setting"
    )
    min_obs: int = Field(default=2, ge=2, le=2520)
    mode: str = Field(default="lowest", description="lowest|highest")
    score_basis: str = Field(default="mean", description="mean|mean_abs")
    n: int = Field(default=5, ge=1, le=500)


class FuturesTrendBacktestRequest(BaseModel):
    group_name: str | None = Field(
        default=None, description="If null, use active group"
    )
    range_key: str = Field(default="all", description="1m|3m|6m|1y|3y|5y|10y|all")
    start_date: str | None = Field(
        default=None, description="Optional explicit YYYYMMDD"
    )
    end_date: str | None = Field(default=None, description="Optional explicit YYYYMMDD")
    dynamic_universe: bool | None = Field(
        default=None, description="If null, use saved global setting"
    )
    backtest_mode: str = Field(
        default="portfolio",
        description="portfolio|single — single-asset uses full allocation; "
        "portfolio enables sizing",
    )
    single_code: str | None = Field(
        default=None,
        description="Required when backtest_mode=single; must be in the group",
    )
    position_sizing: str = Field(
        default="equal",
        description="equal|risk_budget — portfolio only; ignored for single",
    )
    risk_budget_atr_window: int = Field(
        default=20,
        ge=2,
        le=500,
        description="ATR window for risk-budget sizing (portfolio + risk_budget)",
    )
    risk_budget_pct: float = Field(
        default=0.01,
        ge=0.001,
        le=0.03,
        description="Per-asset risk budget as fraction of NAV (1% => 0.01)",
    )
    risk_budget_overcap_policy: str = Field(
        default="scale",
        description="scale|skip_entry|replace_entry|leverage_entry",
    )
    risk_budget_max_leverage_multiple: float = Field(
        default=2.0,
        gt=1.0,
        le=10.0,
        description="Cap when policy=leverage_entry; above cap, scale to this gross",
    )
    monthly_risk_budget_enabled: bool = Field(
        default=False,
        description="Portfolio only: ETF-aligned monthly max-loss gate on new entries",
    )
    monthly_risk_budget_pct: float = Field(
        default=0.06,
        ge=0.01,
        le=0.06,
        description="Monthly budget as fraction of NAV (6% => 0.06), same range as ETF trend portfolio",
    )
    monthly_risk_budget_include_new_trade_risk: bool = Field(
        default=False,
        description="If true, count candidate new-trade risk against monthly budget headroom",
    )
    atr_stop_mode: str = Field(
        default="none",
        description="Universal ATR stop (aligned with ETF trend): none|static|trailing|tightening. "
        "Independent of monthly_risk_budget_enabled; futures engine applies these only when the monthly gate runs.",
    )
    atr_stop_atr_basis: str = Field(
        default="latest",
        description="entry|latest — ATR reference for trailing/tightening (ETF-aligned).",
    )
    atr_stop_reentry_mode: str = Field(
        default="reenter",
        description="reenter|wait_next_entry — aligned with ETF trend UI; "
        "monthly gate ignores reentry (same as ETF portfolio monthly gate).",
    )
    atr_stop_window: int = Field(
        default=14,
        ge=2,
        description="ATR window for universal params (distinct from risk_budget_atr_window); "
        "used by monthly gate when that gate is enabled.",
    )
    atr_stop_n: float = Field(default=2.0, gt=0.0)
    atr_stop_m: float = Field(default=0.5, gt=0.0)
    exec_price: str = Field(default="close", description="open|close")
    trend_strategy: str = Field(
        default="ma_cross",
        description="Trend signal family: ma_cross | ma_filter",
    )
    trade_direction: str = Field(
        default="long_only",
        description="Ma_cross: long_only | short_only | both "
        "(risk_budget sizing allows long_only only)",
    )
    ma_type: str = Field(
        default="sma",
        description="ma_cross: sma|ema|wma; ma_filter: kama only",
    )
    fast_ma: int = Field(default=20, ge=2, le=500)
    slow_ma: int = Field(default=60, ge=3, le=800)
    kama_er_window: int = Field(
        default=10,
        ge=2,
        description="KAMA ER lookback window (ma_filter only)",
    )
    kama_fast_window: int = Field(
        default=2,
        ge=1,
        description="KAMA fast smoothing window (ma_filter only)",
    )
    kama_slow_window: int = Field(
        default=30,
        ge=2,
        description="KAMA slow smoothing window (ma_filter only; must be > fast)",
    )
    kama_std_window: int = Field(
        default=20,
        ge=2,
        description="KAMA std window (ma_filter only)",
    )
    kama_std_coef: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="KAMA std coefficient in [0,3] (ma_filter only)",
    )
    entry_filter_enabled: bool = Field(
        default=False,
        description="Universal entry filter switch (independent from trend_strategy)",
    )
    long_entry_filter_ma: int = Field(
        default=200,
        ge=2,
        le=2000,
        description="Universal long-entry filter MA window: trigger-day close must be above this MA",
    )
    short_entry_filter_ma: int = Field(
        default=200,
        ge=2,
        le=2000,
        description="Universal short-entry filter MA window: trigger-day close must be below this MA",
    )
    position_size_pct: float = Field(default=1.0, gt=0.0, le=1.0)
    min_points: int = Field(default=120, ge=2, le=100000)
    cost_bps: float = Field(
        default=4.0,
        ge=0.0,
        le=2000.0,
        description="Commission bps; with fee_side=one_way each open/close fill pays full bps (default).",
    )
    fee_side: str = Field(
        default="one_way",
        description="one_way: bps per fill (open+close each pay cost_bps); "
        "two_way: cost_bps is round-trip total, halved per fill",
    )
    slippage_type: str = Field(
        default="tick_multiple",
        description="percent|price_spread|tick_multiple — tick_multiple uses pool min_price_tick",
    )
    slippage_value: float = Field(
        default=1.0,
        ge=0.0,
        description="percent ratio, absolute price spread, or integer tick multiple (tick_multiple)",
    )
    slippage_side: str = Field(
        default="one_way",
        description="one_way|two_way — same semantics as fee_side for spread ratio per fill",
    )
    account_capital_wan: float = Field(
        default=500.0,
        gt=0.0,
        description="Initial account size in 万 RMB (×10000 → CNY for the lot engine)",
    )
    backtest_margin_rate_pct: float = Field(
        default=15.0,
        gt=0.0,
        le=100.0,
        description="Backtest-only margin rate % on settle×multiplier per lot (not pool min margin)",
    )
    reserve_margin_ratio: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Min equity fraction kept unencumbered; max margin use = equity×(1−reserve)",
    )


class FuturesRotationBacktestRequest(BaseModel):
    group_name: str | None = Field(
        default=None, description="If null, use active group"
    )
    range_key: str = Field(default="all", description="1m|3m|6m|1y|3y|5y|10y|all")
    start_date: str | None = Field(
        default=None, description="Optional explicit YYYYMMDD"
    )
    end_date: str | None = Field(default=None, description="Optional explicit YYYYMMDD")
    dynamic_universe: bool | None = Field(
        default=None, description="If null, use saved global setting"
    )
    rebalance: Literal["daily", "weekly", "monthly"] = Field(
        default="weekly",
        description="Rebalance frequency; weekly/monthly use period-end anchors",
    )
    lookback_days: int = Field(default=20, ge=2, le=2520)
    top_k: int = Field(default=1, ge=1, le=500)
    trade_direction: Literal["long_only", "short_only"] = Field(default="long_only")
    position_mode: Literal["equal", "inverse_vol"] = Field(default="equal")
    inverse_vol_window: int = Field(
        default=20,
        ge=2,
        le=500,
        description="Used when position_mode=inverse_vol",
    )
    exec_price: Literal["open", "close"] = Field(default="close")
    position_size_pct: float = Field(default=1.0, gt=0.0, le=1.0)
    min_points: int = Field(default=120, ge=2, le=100000)
    cost_bps: float = Field(
        default=4.0,
        ge=0.0,
        le=2000.0,
        description="Commission bps; per-fill semantics controlled by fee_side",
    )
    fee_side: Literal["one_way", "two_way"] = Field(default="one_way")
    slippage_type: Literal["percent", "price_spread", "tick_multiple"] = Field(
        default="tick_multiple"
    )
    slippage_value: float = Field(
        default=1.0,
        ge=0.0,
        description="percent ratio, absolute spread, or integer tick multiple",
    )
    slippage_side: Literal["one_way", "two_way"] = Field(default="one_way")
    account_capital_wan: float = Field(
        default=500.0,
        gt=0.0,
        description="Initial account size in 万 RMB (×10000 => CNY)",
    )
    backtest_margin_rate_pct: float = Field(
        default=15.0,
        gt=0.0,
        le=100.0,
        description="Backtest-only margin rate % on settle×multiplier per lot",
    )
    reserve_margin_ratio: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Min equity fraction kept unencumbered; max margin use = equity×(1−reserve)",
    )


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

    token: str | None = Field(
        default=None,
        description="Optional sync token. If MOMENTUM_SYNC_TOKEN is set, token is required.",
    )
    date: str | None = Field(
        default=None, description="YYYYMMDD; default=server today (Asia/Shanghai)"
    )
    adjusts: list[str] = Field(
        default_factory=lambda: ["qfq", "hfq", "none"],
        description="Adjust list: subset of qfq/hfq/none",
    )
    full_refresh: bool | None = Field(
        default=None,
        description="If true, refresh full history every run; if null, use server default.",
    )
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
    rebalance: str = Field(
        default="weekly", description="daily/weekly/monthly/quarterly/yearly/none"
    )
    rolling_weeks: list[int] = Field(default_factory=lambda: [4, 12, 52])
    rolling_months: list[int] = Field(default_factory=lambda: [3, 6, 12])
    rolling_years: list[int] = Field(default_factory=lambda: [1, 3])
    fft_windows: list[int] = Field(
        default_factory=lambda: [252, 126],
        description="FFT rolling windows in trading days, e.g. [252,126] -> last_252 and last_126 summaries",
    )
    fft_roll: bool = Field(
        default=True,
        description="If true, compute rolling FFT time series for EW (downsampled by fft_roll_step)",
    )
    fft_roll_step: int = Field(
        default=5,
        ge=1,
        description="Compute rolling FFT features every N trading days to reduce runtime",
    )
    rp_window_days: int = Field(
        default=60,
        ge=2,
        le=2520,
        description="ERC / inverse-vol rolling window (trading days): sample covariance for RP (ERC) and vol for IVOL",
    )
    holding_mode: str = Field(
        default="EW",
        description="Holding strategy mode: EW|RP (ERC)|IVOL (inverse-vol)|CUSTOM",
    )
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
    dca_enabled: bool = Field(
        default=False,
        description="Enable fixed-amount DCA mode for holding strategy returns.",
    )
    dca_base_amount: float = Field(
        default=100000.0,
        gt=0.0,
        description="Base position amount invested on the first trading day when DCA is enabled.",
    )
    dca_periodic_amount: float = Field(
        default=10000.0,
        ge=0.0,
        description="Periodic DCA amount invested on each scheduled contribution date.",
    )
    dca_frequency: Literal[
        "none", "daily", "weekly", "monthly", "quarterly", "yearly"
    ] = Field(
        default="monthly",
        description="Periodic DCA frequency. 'none' means only the first-day base amount.",
    )
    lppl_enabled: bool = Field(
        default=False,
        description="Enable LPPL crash prediction block in period_distributions.daily_lppl",
    )
    lppl_lookback_days: int = Field(
        default=504,
        ge=60,
        le=5000,
        description="Maximum lookback window for LPPL fit (trading days)",
    )
    lppl_min_points: int = Field(
        default=120,
        ge=30,
        le=2000,
        description="Minimum valid points required for LPPL fitting",
    )
    lppl_horizon_days: int = Field(
        default=120,
        ge=5,
        le=1260,
        description="Accepted future horizon for tc (calendar days)",
    )
    lppl_multistart: int = Field(
        default=25,
        ge=5,
        le=200,
        description="Max random searches passed to lppls fit(max_searches)",
    )
    lppl_start_mode: str = Field(
        default="auto_lagrange",
        description="LPPL fit window start mode: auto_lagrange|fixed_lookback",
    )
    lppl_start_min_window: int = Field(
        default=120,
        ge=30,
        le=5000,
        description="Min window for lagrange start detection",
    )
    lppl_start_max_window: int = Field(
        default=504,
        ge=60,
        le=5000,
        description="Max window for lagrange start detection",
    )
    lppl_start_step: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Step size for lagrange start detection window scan",
    )
    lppl_bootstrap_on: bool = Field(
        default=True,
        description="Enable moving-block bootstrap for tc distribution",
    )
    lppl_bootstrap_reps: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Bootstrap repetitions for tc distribution",
    )
    lppl_bootstrap_block_size: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Moving-block bootstrap block length (calendar days)",
    )
    lppl_bootstrap_seed: int | None = Field(
        default=None,
        description="Optional random seed for LPPL bootstrap reproducibility",
    )
    lppl_c_rel_min: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum |C|/|B| threshold for oscillation significance",
    )


class LeadLagAnalysisRequest(BaseModel):
    """
    Lead/lag and causality study between an ETF and a volatility index (VIX/GVZ).
    """

    etf_code: str = Field(
        min_length=1,
        description="ETF code (db mode) or a label for the asset (external mode), e.g. 518880",
    )
    asset_provider: str = Field(
        default="db", description="db|stooq|yahoo|auto (asset side)"
    )
    asset_symbol: str | None = Field(
        default=None,
        description="When asset_provider != db, the provider symbol to fetch as the asset close series, e.g. qqq.us or ^ndx",
    )
    index_symbol: str = Field(
        min_length=1,
        description="Index/series symbol. Examples: VIX/GVZ (Cboe), ^VIX/^GVZ (Yahoo), DGS2/DGS5/DGS10/DGS30 (FRED), DINIW (Sina), XAUUSD/XAGUSD (Sina global spot), GC.F (Stooq), GC=F (Yahoo).",
    )
    index_provider: str = Field(
        default="cboe", description="cboe|yahoo|fred|stooq|sina|sina_global|auto"
    )
    index_align: str = Field(
        default="cn_next_trading_day", description="none|cn_next_trading_day"
    )
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")
    max_lag: int = Field(
        default=20,
        ge=0,
        le=252,
        description="Cross-correlation lag window (+/- trading days)",
    )
    granger_max_lag: int = Field(
        default=10, ge=1, le=60, description="Max lag order for Granger causality tests"
    )
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Significance level")
    # Trading usefulness evaluation
    trade_cost_bps: float = Field(
        default=2.0, ge=0.0, description="Per-switch cost (bps) for the toy strategy"
    )
    rolling_window: int = Field(
        default=252,
        ge=20,
        le=2520,
        description="Rolling window for stability charts (trading days)",
    )
    enable_threshold: bool = Field(
        default=True, description="If true, add threshold-gated signal evaluation"
    )
    threshold_quantile: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Quantile on |index_ret| to trigger signals",
    )
    walk_forward: bool = Field(
        default=True,
        description="If true, run walk-forward (train->test) parameter selection",
    )
    train_ratio: float = Field(
        default=0.60, gt=0.1, lt=0.9, description="Train split ratio for walk-forward"
    )
    walk_objective: str = Field(
        default="sharpe", description="Walk-forward objective: sharpe|cagr"
    )

    # Volatility-timing strategy (level-based, tiered exposure), e.g. GVZ high -> reduce exposure
    vol_timing: bool = Field(
        default=False,
        description="If true, backtest tiered exposure based on index close level quantiles",
    )
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
    vol_timing: bool = Field(
        default=False,
        description="If true, backtest tiered exposure based on index close level quantiles",
    )
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
    a_series_id: str = Field(
        min_length=1, description="Series A (target) id, e.g. XAUUSD"
    )
    b_series_id: str = Field(
        min_length=1, description="Series B (indicator) id, e.g. DGS10 or DINIW"
    )
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
    gold_fut_series_id: str | None = Field(
        default="GC_FUT", description="Optional, e.g. GC_FUT"
    )
    dxy_series_id: str = Field(default="DINIW", min_length=1)
    yield_series_id: str = Field(
        default="DGS10", min_length=1, description="One tenor to focus on, e.g. DGS10"
    )

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
    series_ids: list[str] = Field(
        min_length=1, description="macro series_id list to fetch from macro_prices"
    )


class MacroSeriesBatchResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    error: str | None = None


class SimGbmPhase1Request(BaseModel):
    start: str = Field(default="19900101", description="YYYYMMDD")
    end: str | None = Field(
        default=None, description="YYYYMMDD; default=last business day"
    )
    n_assets: int = Field(default=4, ge=2, le=20)
    vol_low: float = Field(default=0.05, gt=0.0, lt=2.0)
    vol_high: float = Field(default=0.30, gt=0.0, lt=2.0)
    corr_low: float | None = Field(
        default=None,
        ge=-0.99,
        lt=0.99,
        description="Optional pairwise correlation lower bound; unset means uncorrelated",
    )
    corr_high: float | None = Field(
        default=None,
        ge=-0.99,
        lt=0.99,
        description="Optional pairwise correlation upper bound; unset means uncorrelated",
    )
    mu_low: float | None = Field(
        default=None,
        ge=-1.0,
        le=3.0,
        description="Optional annual drift lower bound; unset uses random default range",
    )
    mu_high: float | None = Field(
        default=None,
        ge=-1.0,
        le=3.0,
        description="Optional annual drift upper bound; unset uses random default range",
    )
    seed: int | None = Field(default=None)


class SimGbmHoldingStrategyParams(BaseModel):
    rebalance: str = Field(
        default="weekly", description="daily|weekly|monthly|quarterly|yearly|none"
    )
    cost_bps: float = Field(
        default=2.0,
        ge=0.0,
        description="Round-trip transaction cost in bps per turnover",
    )
    rp_vol_window: int = Field(
        default=20,
        ge=2,
        le=2520,
        description="Rolling vol window for risk-parity allocation",
    )


class SimGbmPhase2Request(SimGbmPhase1Request):
    lookback_days: int = Field(default=20, ge=2, le=2520)
    strategy_a: dict | None = Field(
        default=None,
        description="Optional rotation strategy params (same semantics as A/B strategy A)",
    )
    strategy_b: dict | None = Field(
        default=None, description="Optional rotation strategy params for B variant"
    )
    target_a: str | None = Field(
        default=None,
        description="Compare target A: cash|equal_weight|risk_parity|rotation_a|rotation_b",
    )
    target_b: str | None = Field(
        default=None,
        description="Compare target B: cash|equal_weight|risk_parity|rotation_a|rotation_b",
    )
    holding_strategy: SimGbmHoldingStrategyParams = Field(
        default_factory=SimGbmHoldingStrategyParams
    )
    holding_strategy_a: SimGbmHoldingStrategyParams | None = Field(default=None)
    holding_strategy_b: SimGbmHoldingStrategyParams | None = Field(default=None)
    phase1_base: dict | None = Field(
        default=None,
        description="Optional phase1 payload to reuse generated GBM world directly",
    )


class SimGbmPhase3Request(SimGbmPhase2Request):
    n_sims: int = Field(default=10000, ge=8, le=50000)
    chunk_size: int = Field(default=200, ge=1, le=2000)
    n_jobs: int = Field(
        default=0,
        ge=0,
        le=64,
        description="Parallel workers for Monte Carlo runs; 0=auto",
    )


class SimGbmPhase4Request(SimGbmPhase3Request):
    initial_cash: float = Field(default=1_000_000.0, gt=0.0)
    position_pct: float = Field(default=0.10, ge=0.0, le=10.0)


class SimGbmAbStrategyParams(BaseModel):
    rebalance: str = Field(
        default="weekly", description="daily/weekly/monthly/quarterly/yearly"
    )
    rebalance_anchor: int | None = Field(default=None)
    rebalance_shift: str = Field(default="prev")
    exec_price: str = Field(
        default="open",
        description="open|close; oc2 is deprecated and computed as open/close arithmetic average",
    )
    top_k: int = Field(
        default=1,
        description="Non-zero: top-K by momentum if positive, bottom-K (inverse) if negative; effective=min(|K|, pool).",
    )
    position_mode: str = Field(
        default="adaptive", description="adaptive|fixed|inverse_vol"
    )
    entry_backfill: bool = Field(default=False)
    entry_match_n: int = Field(default=0, ge=0)
    exit_match_n: int = Field(default=0, ge=0)
    lookback_days: int = Field(default=20, ge=1)
    skip_days: int = Field(default=0, ge=0)
    score_method: str = Field(default="raw_mom")
    cost_bps: float = Field(default=2.0, ge=0.0)
    trend_filter: bool = Field(default=False)
    trend_exit_filter: bool = Field(default=False)
    trend_sma_window: int = Field(default=20, ge=1)
    trend_ma_type: str = Field(
        default="sma", description="sma|ema|vma(variable/adaptive)"
    )
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
    asset_momentum_floor_rules: list[AssetMomentumFloorRule] | None = Field(
        default=None
    )
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
    end: str | None = Field(
        default=None, description="YYYYMMDD; default=last business day"
    )
    n_worlds: int = Field(default=3000, ge=2, le=20000)
    n_assets: int = Field(default=4, ge=2, le=20)
    vol_low: float = Field(default=0.05, gt=0.0, lt=2.0)
    vol_high: float = Field(default=0.30, gt=0.0, lt=2.0)
    corr_low: float | None = Field(
        default=None,
        ge=-0.99,
        lt=0.99,
        description="Optional pairwise correlation lower bound; unset means uncorrelated",
    )
    corr_high: float | None = Field(
        default=None,
        ge=-0.99,
        lt=0.99,
        description="Optional pairwise correlation upper bound; unset means uncorrelated",
    )
    mu_low: float | None = Field(
        default=None,
        ge=-1.0,
        le=3.0,
        description="Optional annual drift lower bound; unset uses random default range",
    )
    mu_high: float | None = Field(
        default=None,
        ge=-1.0,
        le=3.0,
        description="Optional annual drift upper bound; unset uses random default range",
    )
    seed: int | None = Field(default=None)
    n_perm: int = Field(
        default=5000,
        ge=200,
        le=20000,
        description="Permutations; UI default is high; tests may use fewer",
    )
    n_boot: int = Field(
        default=3000,
        ge=200,
        le=20000,
        description="Bootstrap resamples; UI default is high; tests may use fewer",
    )
    n_jobs: int = Field(
        default=1,
        ge=0,
        le=64,
        description="Parallel workers for world evaluation; 0=auto",
    )
    stability_repeats: int = Field(
        default=0, ge=0, le=30, description="Seed stability repeats; 0 disables"
    )
    stability_worlds: int = Field(
        default=100, ge=2, le=2000, description="Worlds per seed stability repeat"
    )
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
    holding_strategy_a: SimGbmHoldingStrategyParams = Field(
        default_factory=SimGbmHoldingStrategyParams
    )
    holding_strategy_b: SimGbmHoldingStrategyParams = Field(
        default_factory=SimGbmHoldingStrategyParams
    )


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
    fx_series_id: str = Field(
        default="USDCNH", min_length=1, description="FX to convert global->CNY"
    )

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

    etf_code: str = Field(
        default="518880", min_length=1, description="ETF code in etf_pool/etf_prices"
    )
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

    etf_code: str = Field(
        default="513100", description="A-share ETF code (default: Nasdaq ETF)"
    )
    index: str = Field(default="VIX", description="Vol index: VIX|GVZ")
    index_align: str = Field(
        default="cn_next_trading_day", description="none|cn_next_trading_day"
    )
    calendar: str = Field(
        default="XSHG", description="Exchange calendar for CN trading days"
    )
    current_position: str = Field(default="unknown", description="long|cash|unknown")
    lookback_window: int = Field(
        default=252,
        ge=20,
        le=2520,
        description="Lookback window for threshold estimation",
    )
    threshold_quantile: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Quantile on |index_ret| (past window) to trigger trades",
    )
    min_abs_ret: float = Field(
        default=0.0, ge=0.0, description="Hard minimum abs(log-ret) threshold"
    )
    mode: str = Field(
        default="next_cn_day",
        description="next_cn_day|latest_available. next_cn_day: return action for next CN trading day; "
        "if signal not ready, return error.",
    )
    target_cn_trade_date: str | None = Field(
        default=None,
        description="Optional CN trade date YYYYMMDD; if null use latest available mapped date",
    )


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
    etf_code: str = Field(
        default="513100", description="A-share ETF code (default: Nasdaq ETF)"
    )
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")
    index: str = Field(default="VIX", description="Vol index: VIX|GVZ")
    index_align: str = Field(
        default="cn_next_trading_day", description="none|cn_next_trading_day"
    )
    calendar: str = Field(
        default="XSHG", description="Exchange calendar for CN trading days"
    )
    exec_model: str = Field(
        default="open_open", description="Execution/return model: open_open|close_close"
    )
    lookback_window: int = Field(
        default=252,
        ge=20,
        le=2520,
        description="Lookback window for threshold estimation",
    )
    threshold_quantile: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Quantile on |index_log_ret| to trigger trades",
    )
    min_abs_ret: float = Field(
        default=0.0, ge=0.0, description="Hard minimum abs(log-ret) threshold"
    )
    trade_cost_bps: float = Field(
        default=2.0, ge=0.0, description="Per-switch cost (bps) when position changes"
    )
    initial_position: str = Field(
        default="long", description="long|cash starting position at start date"
    )
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


class BaselineGarchVolatilityRequest(BaseModel):
    etf_code: str = Field(min_length=1, description="ETF code, e.g. 510300")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: Literal["hfq", "qfq", "none"] = Field(
        default="hfq", description="qfq/hfq/none for ETF prices"
    )
    max_points: int = Field(
        default=0,
        ge=0,
        le=6000,
        description="Max return points used for fitting (0 means no cap)",
    )
    min_samples: int = Field(
        default=120, ge=60, le=3000, description="Minimum cleaned return samples"
    )
    arch_lags: int = Field(
        default=10, ge=1, le=40, description="ARCH-LM lag count for diagnostics"
    )
    ann_factor: int = Field(
        default=252, ge=200, le=400, description="Annualization factor"
    )
    return_scale: float = Field(
        default=100.0,
        gt=0.0,
        le=10000.0,
        description="Scale applied before fitting, e.g. 100 means percent returns",
    )


class BaselineGarchArchLmOut(BaseModel):
    ok: bool
    lags: int
    n_obs: int
    stat: float | None = None
    pvalue: float | None = None
    significant: bool | None = None


class BaselineGarchParamsOut(BaseModel):
    mu: float | None = None
    omega: float | None = None
    alpha1: float | None = None
    gamma1: float | None = None
    beta1: float | None = None
    nu: float | None = None
    persistence: float | None = None
    unconditional_var_daily: float | None = None
    unconditional_vol_daily: float | None = None
    unconditional_vol_annualized: float | None = None


class BaselineGarchDiagnosticsOut(BaseModel):
    converged: bool
    convergence_flag: int
    n_obs_raw: int
    n_obs_price: int
    n_obs_returns: int
    dropped_obs: int
    ann_factor: int
    return_scale: float
    loglikelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    std_resid_mean: float | None = None
    std_resid_std: float | None = None
    std_resid_skew: float | None = None
    std_resid_kurtosis_excess: float | None = None
    arch_lm_pre: BaselineGarchArchLmOut
    arch_lm_post: BaselineGarchArchLmOut


class BaselineGarchInterpretationOut(BaseModel):
    model_value: Literal["high", "medium", "low"]
    value_score: float = Field(ge=0.0, le=1.0)
    summary: str
    reasons: list[str] = Field(default_factory=list)


class BaselineGarchSeriesOut(BaseModel):
    price_dates: list[str] = Field(default_factory=list)
    price_close: list[float | None] = Field(default_factory=list)
    vol_dates: list[str] = Field(default_factory=list)
    cond_vol_daily: list[float | None] = Field(default_factory=list)
    cond_vol_annualized: list[float | None] = Field(default_factory=list)
    log_returns: list[float | None] = Field(default_factory=list)


class BaselineGarchVolatilityResponse(BaseModel):
    ok: bool
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    params: BaselineGarchParamsOut | None = None
    diagnostics: BaselineGarchDiagnosticsOut | None = None
    interpretation: BaselineGarchInterpretationOut | None = None
    series: BaselineGarchSeriesOut | None = None


class VolProxyMethod(BaseModel):
    """
    Volatility proxy computed from the ETF OHLC series.

    Output level is annualized volatility (decimal), intended for level-based tier timing.
    """

    name: str = Field(
        min_length=1,
        description="Unique method name in response, e.g. rv20, yz20, har252",
    )
    kind: str = Field(
        description="rv_close|ewma_close|parkinson|garman_klass|rogers_satchell|yang_zhang|har_rv"
    )
    window: int = Field(
        default=20, ge=2, le=2520, description="Rolling window (trading days)"
    )
    ann: int = Field(default=252, ge=50, le=400, description="Annualization factor")

    ewma_lambda: float = Field(
        default=0.94, gt=0.0, lt=1.0, description="EWMA decay (ewma_close only)"
    )
    har_train_window: int = Field(
        default=252, ge=30, le=2520, description="Rolling train window (har_rv only)"
    )
    har_horizons: list[int] = Field(
        default_factory=lambda: [1, 5, 22], description="HAR horizons (har_rv only)"
    )


class VolProxyTimingRequest(BaseModel):
    etf_code: str = Field(min_length=1, description="ETF code, e.g. 518880 / 513100")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none for ETF prices")

    methods: list[VolProxyMethod] = Field(
        min_length=1, description="Vol proxy variants to compute and backtest"
    )

    # Tiering config (same semantics as leadlag vol_timing)
    level_quantiles: list[float] = Field(
        default_factory=lambda: [0.8, 0.9], description="Quantiles on level (train)"
    )
    level_exposures: list[float] = Field(
        default_factory=lambda: [1.0, 0.5, 0.2],
        description="Tier exposures, len=quantiles+1",
    )
    level_window: str = Field(
        default="all",
        description="Quantile window for levels: all(expanding,no-lookahead)|static_all(full-sample,lookahead)|1y|3y|5y|10y",
    )
    trade_cost_bps: float = Field(
        default=2.0, ge=0.0, description="Per-switch cost in bps"
    )

    walk_forward: bool = Field(
        default=True,
        description="If true, split train/test and apply train thresholds to test",
    )
    train_ratio: float = Field(
        default=0.60, gt=0.1, lt=0.9, description="Train ratio for walk-forward"
    )


class VolProxyTimingResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    methods: dict | None = None
    error: str | None = None


class RangeStateMonitorRequest(BaseModel):
    etf_code: str = Field(min_length=1, description="ETF code, e.g. 510300")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="qfq", description="qfq/hfq/none for ETF prices")
    mode: str = Field(default="adx", description="adx|adxr|er")
    window: int = Field(default=14, ge=2, le=252)
    enter_threshold: float = Field(default=20.0)
    exit_threshold: float = Field(default=25.0)

    @model_validator(mode="after")
    def _validate_range_state_monitor(self) -> RangeStateMonitorRequest:
        mode = str(self.mode or "adx").strip().lower()
        if mode not in {"adx", "adxr", "er"}:
            raise ValueError("mode must be one of: adx|adxr|er")
        if mode in {"adx", "adxr"}:
            if not (0.0 <= float(self.enter_threshold) <= 100.0):
                raise ValueError("enter_threshold must be in [0,100] for mode=adx|adxr")
            if not (0.0 <= float(self.exit_threshold) <= 100.0):
                raise ValueError("exit_threshold must be in [0,100] for mode=adx|adxr")
        else:
            if not (0.0 <= float(self.enter_threshold) <= 1.0):
                raise ValueError("enter_threshold must be in [0,1] for mode=er")
            if not (0.0 <= float(self.exit_threshold) <= 1.0):
                raise ValueError("exit_threshold must be in [0,1] for mode=er")
        if float(self.exit_threshold) < float(self.enter_threshold):
            raise ValueError("exit_threshold must be >= enter_threshold")
        return self


class RangeStateMonitorResponse(BaseModel):
    ok: bool
    meta: dict | None = None
    series: dict | None = None
    latest: dict | None = None
    summary: dict | None = None
    error: str | None = None


class BaselineCalendarEffectRequest(BaseModel):
    codes: list[str] = Field(min_length=1)
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="qfq/hfq/none (global)")
    rebalance: str = Field(
        default="weekly",
        description="weekly/monthly/quarterly/yearly (calendar-effect study)",
    )
    rebalance_shift: str = Field(
        default="prev",
        description="If anchor falls on non-trading day: prev -> shift to previous trading day (default); next -> shift to next trading day.",
    )
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    exec_prices: list[str] = Field(
        default_factory=lambda: ["open", "close", "oc2"],
        description="Execution price list: open|close|oc2 (OC average)",
    )


class CalendarTimingStrategyRequest(BaseModel):
    mode: str = Field(default="portfolio", description="portfolio|single")
    code: str | None = Field(default=None, description="Single-asset mode code")
    codes: list[str] | None = Field(
        default=None, description="Portfolio mode candidate codes"
    )
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(
        default="none", description="qfq/hfq/none for execution price series"
    )
    decision_day: int = Field(
        default=1,
        description="Monthly natural decision day in [-28,28] excluding 0. Negative means from month-end.",
    )
    hold_days: int = Field(
        default=1, ge=1, le=252, description="Holding days from execution day"
    )
    position_mode: str = Field(
        default="equal", description="equal|fixed_ratio|risk_budget"
    )
    fixed_pos_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Exposure when position_mode=fixed_ratio",
    )
    risk_budget_atr_window: int = Field(
        default=20, ge=2, description="ATR window when position_mode=risk_budget"
    )
    risk_budget_pct: float = Field(
        default=0.01,
        ge=0.001,
        le=0.03,
        description="Per-asset NAV risk budget for 1 ATR move (0.01 = 1%)",
    )
    dynamic_universe: bool = Field(
        default=False,
        description="If true, allow dynamic candidate coverage over union interval",
    )
    exec_price: str = Field(default="open", description="open|close")
    cost_bps: float = Field(
        default=2.0, ge=0.0, description="Two-way transaction cost in bps"
    )
    slippage_rate: float = Field(
        default=0.001,
        ge=0.0,
        description="One-way adverse slippage spread (absolute price diff)",
    )
    rebalance_shift: str = Field(
        default="prev",
        description="If decision day is non-trading: prev|next|skip",
    )
    calendar: str = Field(
        default="XSHG", description="Trading calendar used by decision-day shift"
    )


class AssetRiskControlRule(BaseModel):
    """
    Per-asset risk-control rule applied to weights daily:
    - signal is computed from the asset's own qfq close-based NAV proxy
    - when triggered, scale that asset's weight by (1 - reduce_pct)
    """

    code: str = Field(min_length=1, description="ETF code")
    sig_type: str = Field(description="return|volatility|downside_vol|drawdown")
    k: int = Field(ge=2, le=2520, description="Signal lookback window in trading days")
    p_in: float = Field(
        gt=0.0,
        lt=100.0,
        description="Trigger percentile (0-100). For return, low-tail is used (100-p_in).",
    )
    reduce_pct: float = Field(
        ge=0.0,
        le=100.0,
        description="Reduce percent on trigger: exposure = 1 - reduce_pct/100",
    )
    recovery_mode: str = Field(
        default="immediate", description="immediate|hysteresis|cooldown"
    )
    p_out: float | None = Field(
        default=None,
        gt=0.0,
        lt=100.0,
        description="Recovery percentile for hysteresis mode",
    )
    cooldown_days: int = Field(
        default=0,
        ge=0,
        le=2520,
        description="Minimum days to keep reduced exposure (cooldown mode)",
    )


class AssetVolIndexTimingRule(BaseModel):
    """
    Per-asset volatility-index timing rule applied daily to weights:
    - signal is the volatility index LEVEL (e.g. VIX/GVZ), expected aligned to CN next trading day
    - thresholds are computed from rolling/expanding quantiles and shifted by 1 day (no lookahead)
    - when triggered (higher-vol bucket), scale that asset's weight by the tier exposure (cash remainder)
    """

    code: str = Field(min_length=1, description="ETF code")
    index: str = Field(
        description="Vol index code: VIX|GVZ (Cboe) | WAVOL (asset weekly rolling ann vol)"
    )
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
    min_periods: int = Field(
        default=20,
        ge=2,
        le=2520,
        description="Minimum observations before thresholds become active.",
    )


class AssetTrendRule(BaseModel):
    """
    Per-asset trend filter rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    op: str = Field(
        default=">",
        description="Comparison operator between close and MA: = | != | > | < | >= | <=",
    )
    stage: str = Field(default="entry", description="Rule stage: entry | exit | both")
    trend_sma_window: int = Field(
        default=20, ge=1, description="MA window (trading days, qfq close-based)"
    )
    trend_ma_type: str = Field(
        default="sma",
        description="MA type: sma|ema|vma(variable/adaptive) (self close vs self MA)",
    )


class AssetBiasRule(BaseModel):
    """
    Per-asset BIAS filter rule (qfq close-based):
    BIAS = close / MA(window) - 1.
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    op: str = Field(
        default=">", description="Comparison operator: = | != | > | < | >= | <="
    )
    stage: str = Field(default="entry", description="Rule stage: entry | exit | both")
    bias_type: str = Field(default="bias", description="BIAS signal type: bias|bias_v")
    bias_ma_window: int = Field(
        default=20, ge=2, description="MA window (trading days)"
    )
    level_window: str = Field(
        default="all",
        description="Threshold lookback window: 30d|90d|180d|1y|3y|5y|10y|all(expanding)",
    )
    threshold_type: str = Field(default="quantile", description="quantile|fixed")
    quantile: float = Field(
        default=95.0,
        gt=0.0,
        lt=100.0,
        description="Percentile value when threshold_type=quantile, e.g. 95",
    )
    fixed_value: float = Field(
        default=10.0,
        ge=0.0,
        description="Fixed threshold (%) when threshold_type=fixed, e.g. 10 means 10%",
    )
    min_periods: int = Field(
        default=20,
        ge=2,
        le=2520,
        description="Minimum observations before quantile threshold becomes active",
    )


class AssetRsiRule(BaseModel):
    """
    Per-asset RSI filter rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    rsi_window: int = Field(
        default=14, ge=1, description="RSI window (trading days, fixed to 14)"
    )
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0)
    rsi_block_overbought: bool = Field(
        default=True, description="If true, exclude assets with RSI > overbought"
    )
    rsi_block_oversold: bool = Field(
        default=False, description="If true, exclude assets with RSI < oversold"
    )


class AssetChopRule(BaseModel):
    """
    Per-asset choppiness filter rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    chop_mode: str = Field(default="er", description="er|adx")
    chop_window: int = Field(
        default=20, ge=2, description="Efficiency Ratio window (trading days)"
    )
    chop_er_threshold: float = Field(
        default=0.25, gt=0.0, description="ER < threshold => choppy => exclude"
    )
    chop_adx_window: int = Field(
        default=20, ge=2, description="ADX window (trading days)"
    )
    chop_adx_threshold: float = Field(
        default=20.0, gt=0.0, description="ADX < threshold => choppy => exclude"
    )


class AssetVolMonitorRule(BaseModel):
    """
    Per-asset volatility monitor (position sizing) rule (pre-trade; qfq close-based).
    """

    code: str = Field(min_length=1, description="ETF code or '*' for default rule")
    vol_window: int = Field(
        default=20, ge=1, description="Realized vol window (trading days)"
    )
    vol_target_ann: float = Field(
        default=0.20, gt=0.0, description="Annualized target vol for sizing"
    )
    vol_max_ann: float = Field(
        default=0.60,
        gt=0.0,
        description="Annualized hard stop vol; above -> no risk position",
    )


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
    rebalance: str = Field(
        default="weekly", description="daily/weekly/monthly/quarterly/yearly"
    )
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
        description="Base position sizing among selected assets: adaptive(equal among selected) | fixed(each uses 1/|top_k|) | inverse_vol(inverse annualized volatility) | risk_budget(ATR risk budget).",
    )
    daily_rebalance: bool = Field(
        default=False,
        description="Enable daily rebalance of held assets toward target weights (using daily close signal, next-day execution by exec_price).",
    )
    risk_budget_atr_window: int = Field(
        default=20, ge=2, description="ATR window for risk-budget sizing"
    )
    risk_budget_pct: float = Field(
        default=0.01,
        ge=0.001,
        le=0.03,
        description="Per-asset NAV risk budget for 1 ATR move (0.01 = 1%)",
    )
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
    cost_bps: float = Field(default=2.0, ge=0.0)
    slippage_rate: float = Field(
        default=0.001,
        ge=0.0,
        description="One-way adverse slippage spread (absolute price diff)",
    )
    capacity_window_years: Literal[1, 3, 5] = Field(
        default=1,
        description="Capacity statistics window in years: 1|3|5 (default 1 year).",
    )
    stop_scheme: str = Field(
        default="none",
        description="Stop-loss scheme: none|atr|equity_budget. Backward-compatible: when stop_scheme is omitted and atr_stop_mode!=none, stop_scheme falls back to atr.",
    )
    equity_stop_risk_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.05,
        description="Per-trade initial risk budget as % of total equity for equity_budget stop scheme (0.02 = 2%).",
    )
    atr_stop_mode: str = Field(
        default="none",
        description="Universal ATR stop mode: none|static|trailing|tightening",
    )
    atr_stop_atr_basis: str = Field(
        default="latest", description="ATR basis for dynamic modes: entry|latest"
    )
    atr_stop_reentry_mode: str = Field(
        default="reenter",
        description="Re-entry after ATR stop: reenter|wait_next_entry",
    )
    atr_stop_execution_mode: str = Field(
        default="intraday",
        description="ATR stop execution mode: intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    atr_stop_execution_time: Literal["open", "close", "full_day"] | None = Field(
        default=None,
        description="ATR stop execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
    )
    atr_stop_window: int = Field(
        default=14, ge=2, description="ATR window for universal stop"
    )
    atr_stop_n: float = Field(
        default=2.0, gt=0.0, description="ATR stop distance multiplier n"
    )
    atr_stop_m: float = Field(
        default=0.5,
        gt=0.0,
        description="ATR tightening step m (used by tightening mode)",
    )
    # Pre-trade risk controls (all optional; defaults keep previous behavior)
    trend_filter: bool = Field(
        default=False, description="Enable trend filter gating (pre-trade)"
    )
    trend_exit_filter: bool = Field(
        default=False,
        description="Enable trend-based daily exit gating (post-entry; next-day execution)",
    )
    trend_sma_window: int = Field(
        default=20,
        ge=1,
        description="MA window for trend filter (trading days, qfq close-based)",
    )
    trend_ma_type: str = Field(
        default="sma",
        description="Trend MA type: sma|ema|vma(variable/adaptive) (self close vs self MA)",
    )
    bias_filter: bool = Field(
        default=False, description="Enable BIAS filter gating (pre-trade)"
    )
    bias_exit_filter: bool = Field(
        default=False,
        description="Enable BIAS-based daily exit gating (post-entry; next-day execution)",
    )
    bias_type: str = Field(default="bias", description="BIAS signal type: bias|bias_v")
    bias_ma_window: int = Field(
        default=20, ge=2, description="BIAS MA window (trading days)"
    )
    bias_level_window: str = Field(
        default="all",
        description="BIAS threshold lookback window: 30d|90d|180d|1y|3y|5y|10y|all",
    )
    bias_threshold_type: str = Field(
        default="quantile", description="BIAS threshold type: quantile|fixed"
    )
    bias_quantile: float = Field(
        default=95.0,
        gt=0.0,
        lt=100.0,
        description="BIAS percentile threshold (0,100), e.g. 95",
    )
    bias_fixed_value: float = Field(
        default=10.0,
        ge=0.0,
        description="BIAS fixed threshold in percent, e.g. 10 means 10%",
    )
    bias_min_periods: int = Field(
        default=20,
        ge=2,
        le=2520,
        description="Minimum observations for BIAS quantile threshold",
    )
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
                raise ValueError(
                    "floating_benchmark_code is required when top_k_mode=floating"
                )
            if bm_code not in [str(x) for x in (self.codes or [])]:
                raise ValueError(
                    "floating_benchmark_code must be in codes when top_k_mode=floating"
                )
        bm = (
            str(getattr(self, "benchmark_mode", "EW_REBAL") or "EW_REBAL")
            .strip()
            .upper()
        )
        if bm not in {"EW_REBAL", "RP_REBAL", "IVOL_REBAL", "ALL"}:
            raise ValueError(
                "benchmark_mode must be one of: EW_REBAL|RP_REBAL|IVOL_REBAL|ALL"
            )
        stop_scheme = (
            str(getattr(self, "stop_scheme", "none") or "none").strip().lower()
        )
        if stop_scheme not in {"none", "atr", "equity_budget"}:
            raise ValueError("stop_scheme must be one of: none|atr|equity_budget")
        atr_mode = str(getattr(self, "atr_stop_mode", "none") or "none").strip().lower()
        if atr_mode not in {"none", "static", "trailing", "tightening"}:
            raise ValueError(
                "atr_stop_mode must be one of: none|static|trailing|tightening"
            )
        atr_basis = str(getattr(self, "atr_stop_atr_basis", "latest") or "latest")
        atr_basis = atr_basis.strip().lower()
        if atr_basis not in {"entry", "latest"}:
            raise ValueError("atr_stop_atr_basis must be one of: entry|latest")
        atr_reentry = (
            str(getattr(self, "atr_stop_reentry_mode", "reenter") or "reenter")
            .strip()
            .lower()
        )
        if atr_reentry not in {"reenter", "wait_next_entry"}:
            raise ValueError(
                "atr_stop_reentry_mode must be one of: reenter|wait_next_entry"
            )
        fields_set = set(getattr(self, "__pydantic_fields_set__", set()) or set())
        stop_scheme_explicit = "stop_scheme" in fields_set
        if (not stop_scheme_explicit) and stop_scheme == "none" and atr_mode != "none":
            # Backward compatibility: historical clients only send atr_stop_mode.
            self.stop_scheme = "atr"
            stop_scheme = "atr"
        if stop_scheme == "atr" and atr_mode == "none":
            raise ValueError(
                "stop_scheme=atr requires atr_stop_mode to be one of: static|trailing|tightening"
            )
        if (
            stop_scheme == "atr"
            and atr_mode == "tightening"
            and float(getattr(self, "atr_stop_n", 0.0))
            <= float(getattr(self, "atr_stop_m", 0.0))
        ):
            raise ValueError(
                "atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening"
            )
        _validate_overlay_execution_mode_time(
            execution_mode=str(
                getattr(self, "atr_stop_execution_mode", "intraday") or "intraday"
            ),
            execution_time=getattr(self, "atr_stop_execution_time", None),
            mode_field="atr_stop_execution_mode",
            time_field="atr_stop_execution_time",
        )
        raw_exec_time = getattr(self, "atr_stop_execution_time", None)
        exec_time_v = (
            None
            if raw_exec_time is None
            else str(raw_exec_time).strip().lower() or None
        )
        if stop_scheme == "equity_budget":
            if exec_time_v is None:
                self.atr_stop_execution_time = "close"
            elif exec_time_v != "close":
                raise ValueError(
                    "equity_budget stop only supports close execution (intraday-close or next_day-close)"
                )
        return self


class RotationCalendarEffectRequest(RotationBacktestRequest):
    anchors: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Anchor list depends on rebalance: weekly -> weekday 0=Mon..4=Fri; monthly -> day-of-month 1..28; quarterly/yearly -> Nth trading day in period (1..)",
    )
    exec_prices: list[str] = Field(
        default_factory=lambda: ["open", "close", "oc2"],
        description="Execution price list: open|close|oc2 (OC average)",
    )


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
    asof: str = Field(
        description="YYYYMMDD (usually the latest available trading day in backtest range)"
    )


class SimPortfolioCreateRequest(BaseModel):
    name: str = Field(default="默认账户", description="Portfolio name")
    initial_cash: float = Field(
        default=1_000_000.0, gt=0.0, description="Initial cash (base_ccy units)"
    )


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
    rebalance_shift: str = Field(
        default="prev", description="prev|next when anchor falls on non-trading day"
    )
    anchor_weekday: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optional: if set, compute only one anchor weekday (1=Mon..5=Fri) to reduce payload/runtime.",
    )


class RTakeProfitTier(BaseModel):
    r_multiple: float = Field(
        gt=0.0,
        description="Activate drawdown take-profit when peak floating profit reaches this R multiple",
    )
    retrace_ratio: float = Field(
        gt=0.0,
        lt=1.0,
        description="Allowed pullback ratio from peak floating profit once activated",
    )


class RProfitScaleoutTier(BaseModel):
    r_multiple: float = Field(
        gt=0.0,
        description="Activate tiered profit scale-out when floating profit reaches this R multiple",
    )
    reduce_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Position fraction to reduce at this tier relative to initial position at entry",
    )


class BiasVTakeProfitTier(BaseModel):
    threshold: float = Field(
        gt=0.0,
        description="Trigger threshold in BIAS-V units where this tier activates",
    )
    reduce_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Position fraction to reduce at this BIAS-V threshold relative to initial position at entry",
    )


def _validate_overlay_execution_mode_time(
    *,
    execution_mode: str,
    execution_time: Literal["open", "close", "full_day"] | None,
    mode_field: str,
    time_field: str,
) -> None:
    mode_v = str(execution_mode or "intraday").strip().lower()
    if mode_v not in {"intraday", "next_day"}:
        raise ValueError(f"{mode_field} must be one of: intraday|next_day")
    if execution_time is None:
        return
    time_v = str(execution_time).strip().lower()
    if time_v not in {"open", "close", "full_day"}:
        raise ValueError(f"{time_field} must be one of: open|close|full_day")
    if mode_v == "next_day" and time_v == "full_day":
        raise ValueError(f"{time_field} must be open|close when {mode_field}=next_day")


class TrendBacktestRequest(BaseModel):
    code: str = Field(
        min_length=1,
        max_length=32,
        description="Single ETF code for trend-following backtest",
    )
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(
        default="hfq",
        description="[deprecated] Trend research uses mixed basis like rotation: signal=qfq, nav=none with hfq fallback, benchmark=hfq.",
    )
    initial_account_amount: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional initial trading account amount (CNY). null keeps normalized NAV mode.",
    )
    cost_bps: float = Field(
        default=2.0,
        ge=0.0,
        description="Round-trip transaction cost in bps per turnover",
    )
    slippage_rate: float = Field(
        default=0.001,
        ge=0.0,
        description="One-way adverse slippage spread (absolute price diff)",
    )
    capacity_window_years: Literal[1, 3, 5] = Field(
        default=1,
        description="Capacity statistics window in years: 1|3|5 (default 1 year).",
    )
    exec_price: Literal["open", "close"] = Field(
        default="open",
        description="open|close",
    )
    engine: str | None = Field(
        default=None,
        description="Backtest engine switch: legacy|bt; null uses server default",
    )
    strategy: str = Field(
        default="ma_filter",
        description="ma_filter|ma_cross|donchian|tsmom|linreg_slope|bias|macd_cross|macd_zero_filter|macd_v|random_entry (long/cash); ma_filter uses ma_type sma|ema|kama",
    )
    position_sizing: str = Field(
        default="equal", description="equal|vol_target|fixed_ratio|risk_budget"
    )
    vol_window: int = Field(
        default=20, ge=2, description="Rolling vol window for vol-target sizing"
    )
    vol_target_ann: float = Field(
        default=0.20, gt=0.0, description="Annualized target vol for portfolio scaling"
    )
    fixed_pos_ratio: float = Field(
        default=0.04,
        gt=0.0,
        description="Fixed position ratio when position_sizing=fixed_ratio",
    )
    fixed_overcap_policy: str = Field(
        default="skip", description="Over-cap policy placeholder: skip|extend"
    )
    fixed_max_holdings: int = Field(
        default=10,
        ge=1,
        description="Max holdings placeholder for unified payload schema",
    )
    risk_budget_atr_window: int = Field(
        default=20, ge=2, description="ATR window when position_sizing=risk_budget"
    )
    risk_budget_pct: float = Field(
        default=0.01,
        ge=0.001,
        le=0.03,
        description="NAV risk budget for 1 ATR move (0.01 = 1%)",
    )
    risk_budget_overcap_policy: str = Field(
        default="scale",
        description="When risk-budget new entry exceeds total 100% exposure: scale|skip_entry|replace_entry|leverage_entry",
    )
    risk_budget_max_leverage_multiple: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Max leverage multiple when risk_budget_overcap_policy=leverage_entry",
    )
    vol_regime_risk_mgmt_enabled: bool = Field(
        default=False,
        description="Enable post-entry volatility-regime risk management in risk_budget sizing",
    )
    vol_ratio_fast_atr_window: int = Field(
        default=5,
        ge=2,
        description="Fast ATR window for volatility ratio ATR(fast)/ATR(slow)",
    )
    vol_ratio_slow_atr_window: int = Field(
        default=50,
        ge=2,
        description="Slow ATR window for volatility ratio ATR(fast)/ATR(slow)",
    )
    vol_ratio_expand_threshold: float = Field(
        default=1.45,
        gt=0.0,
        description="If ATR ratio > threshold, volatility is expanded and de-risk is triggered",
    )
    vol_ratio_contract_threshold: float = Field(
        default=0.65,
        gt=0.0,
        description="If ATR ratio < threshold, volatility is contracted and add-risk is triggered",
    )
    vol_ratio_normal_threshold: float = Field(
        default=1.05,
        gt=0.0,
        description="Normal-zone recovery threshold used after expanded/contracted states",
    )
    vol_ratio_extreme_threshold: float = Field(
        default=2.2,
        gt=0.0,
        description="If ATR ratio > threshold, enter extreme volatility tier above expanded state",
    )
    vol_periodic_risk_mgmt_enabled: bool = Field(
        default=False,
        description="Enable periodic ATR-based risk-budget rebalance by share-size threshold; mutually exclusive with vol_regime_risk_mgmt_enabled",
    )
    vol_periodic_rebalance_threshold_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum relative share-size change threshold for periodic volatility rebalance (0.05 = 5%)",
    )
    risk_of_ruin_maxrisk: float = Field(
        default=0.30,
        gt=0.0,
        le=1.0,
        description="Configured ruin threshold maxrisk used in RoR formula ((1-P)/P)^(maxrisk/A)",
    )
    # parameters (some are strategy-specific)
    sma_window: int = Field(
        default=200, ge=2, description="MA filter window (trading days)"
    )
    fast_window: int = Field(
        default=50, ge=2, description="Fast MA window (trading days)"
    )
    slow_window: int = Field(
        default=200, ge=2, description="Slow MA window (trading days)"
    )
    ma_type: str = Field(
        default="sma",
        description="MA type: ma_filter supports sma|ema|kama; ma_cross supports sma|ema|wma",
    )
    kama_er_window: int = Field(
        default=10, ge=2, description="KAMA ER lookback window (used when ma_type=kama)"
    )
    kama_fast_window: int = Field(
        default=2,
        ge=1,
        description="KAMA fast smoothing window (used when ma_type=kama)",
    )
    kama_slow_window: int = Field(
        default=30,
        ge=2,
        description="KAMA slow smoothing window (used when ma_type=kama)",
    )
    kama_std_window: int = Field(
        default=20,
        ge=2,
        description="KAMA std window for hysteresis band: KAMA ± coef*std(KAMA)",
    )
    kama_std_coef: float = Field(
        default=1.0, ge=0.0, le=3.0, description="KAMA std filter coefficient in [0,3]"
    )
    donchian_entry: int = Field(
        default=20, ge=2, description="Donchian entry window (trading days)"
    )
    donchian_exit: int = Field(
        default=10, ge=2, description="Donchian exit window (trading days)"
    )
    mom_lookback: int = Field(
        default=252, ge=2, description="TS momentum lookback (trading days)"
    )
    tsmom_entry_threshold: float = Field(
        default=0.0,
        description="TSMOM entry threshold on momentum score; enter when score > threshold",
    )
    tsmom_exit_threshold: float = Field(
        default=0.0,
        description="TSMOM exit threshold on momentum score; exit when score <= threshold",
    )
    atr_stop_mode: str = Field(
        default="none",
        description="Universal ATR stop mode: none|static|trailing|tightening",
    )
    atr_stop_atr_basis: str = Field(
        default="latest", description="ATR basis for dynamic modes: entry|latest"
    )
    atr_stop_reentry_mode: str = Field(
        default="reenter",
        description="Re-entry after ATR stop: reenter|wait_next_entry",
    )
    atr_stop_execution_mode: str = Field(
        default="intraday",
        description="ATR stop execution mode: intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    atr_stop_execution_time: Literal["open", "close", "full_day"] | None = Field(
        default=None,
        description="ATR stop execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
    )
    atr_stop_window: int = Field(
        default=14, ge=2, description="ATR window for universal stop"
    )
    atr_stop_n: float = Field(
        default=2.0, gt=0.0, description="ATR stop distance multiplier n"
    )
    atr_stop_m: float = Field(
        default=0.5,
        gt=0.0,
        description="ATR tightening step m (used by tightening mode)",
    )
    ma_trailing_stop_enabled: bool = Field(
        default=False,
        description="Enable universal MA/EMA trailing-stop overlay",
    )
    ma_trailing_stop_ma_type: str = Field(
        default="sma",
        description="MA trailing-stop type: sma|ema",
    )
    ma_trailing_stop_execution_mode: str = Field(
        default="intraday",
        description="MA trailing-stop execution mode: intraday|next_day",
    )
    ma_trailing_stop_execution_time: Literal["open", "close", "full_day"] | None = (
        Field(
            default=None,
            description="MA trailing-stop execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
        )
    )
    ma_trailing_stop_effective_delay_days: int = Field(
        default=3,
        ge=1,
        description="MA trailing-stop activation delay in trading days after entry (3 means active from T+3)",
    )
    ma_trailing_stop_reduce_window: int = Field(
        default=10,
        ge=2,
        description="Reduce-line MA window used by MA trailing stop",
    )
    ma_trailing_stop_exit_window: int = Field(
        default=20,
        ge=2,
        description="Exit-line MA window used by MA trailing stop",
    )
    ma_trailing_stop_reduce_fraction: float = Field(
        default=0.33,
        gt=0.0,
        le=1.0,
        description="Reduce fraction when reduce line triggers, relative to initial position at entry",
    )
    r_take_profit_enabled: bool = Field(
        default=False,
        description="Enable universal R-multiple drawdown take-profit overlay",
    )
    r_take_profit_reentry_mode: str = Field(
        default="reenter",
        description="Re-entry after R take-profit: reenter|wait_next_entry",
    )
    r_take_profit_execution_mode: str = Field(
        default="intraday",
        description="R take-profit execution mode: intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    r_take_profit_execution_time: Literal["open", "close", "full_day"] | None = Field(
        default=None,
        description="R take-profit execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
    )
    r_take_profit_tiers: list[RTakeProfitTier] | None = Field(
        default=None,
        description="Tiered config: peak>=R multiple activates pullback-exit threshold, e.g. [{r_multiple:2,retrace_ratio:0.5}]",
    )
    r_profit_scaleout_enabled: bool = Field(
        default=False,
        description="Enable tiered floating-profit scale-out overlay (sell on profit level, no pullback required)",
    )
    r_profit_scaleout_execution_mode: str = Field(
        default="intraday",
        description="Floating-profit scale-out execution mode: intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    r_profit_scaleout_execution_time: Literal["open", "close", "full_day"] | None = (
        Field(
            default=None,
            description="R scale-out execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
        )
    )
    r_profit_scaleout_breakeven_stop_enabled: bool = Field(
        default=True,
        description="Enable breakeven-stop addon after first executed floating-profit scale-out",
    )
    r_profit_scaleout_tiers: list[RProfitScaleoutTier] | None = Field(
        default=None,
        description="Tiered scale-out config; each tier reduces a fraction of current remaining position, e.g. [{r_multiple:2,reduce_fraction:0.4},{r_multiple:3,reduce_fraction:0.3}]",
    )
    bias_v_take_profit_enabled: bool = Field(
        default=False, description="Enable universal BIAS-V take-profit overlay"
    )
    bias_v_take_profit_reentry_mode: str = Field(
        default="reenter",
        description="Re-entry after BIAS-V take-profit: reenter|wait_next_entry",
    )
    bias_v_take_profit_execution_mode: str = Field(
        default="intraday",
        description="BIAS-V take-profit execution mode: intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    bias_v_take_profit_execution_time: Literal["open", "close", "full_day"] | None = (
        Field(
            default=None,
            description="BIAS-V take-profit execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
        )
    )
    bias_v_take_profit_breakeven_stop_enabled: bool = Field(
        default=True,
        description="Enable breakeven-stop addon after first executed BIAS-V take-profit tier",
    )
    bias_v_ma_window: int = Field(
        default=20, ge=2, description="MA window in BIAS-V=(close-MA)/ATR"
    )
    bias_v_atr_window: int = Field(
        default=20, ge=2, description="ATR window in BIAS-V=(close-MA)/ATR"
    )
    bias_v_take_profit_tiers: list[BiasVTakeProfitTier] | None = Field(
        default=None,
        description="Tiered BIAS-V take-profit config; each tier reduces a fraction of current remaining position, e.g. [{threshold:5,reduce_fraction:0.5},{threshold:7,reduce_fraction:0.5}]",
    )
    monthly_risk_budget_enabled: bool = Field(
        default=False,
        description="Enable account-level monthly max-loss risk budget gate before new entries",
    )
    monthly_risk_budget_pct: float = Field(
        default=0.06,
        ge=0.01,
        le=0.06,
        description="Monthly max-loss budget on account NAV (0.06 = 6%)",
    )
    monthly_risk_budget_include_new_trade_risk: bool = Field(
        default=False,
        description="If true, include candidate new-trade risk in monthly budget check",
    )
    # BIAS strategy params
    bias_ma_window: int = Field(
        default=20,
        ge=2,
        description="EMA window N in BIAS=(LN(C)-LN(EMA(C,N)))*100 (trading days)",
    )
    bias_entry: float = Field(
        default=2.0, description="Enter when BIAS > entry (percent)"
    )
    bias_hot: float = Field(
        default=10.0, description="Take-profit exit when BIAS >= hot (percent)"
    )
    bias_cold: float = Field(
        default=-2.0, description="Stop-loss exit when BIAS <= cold (percent)"
    )
    bias_pos_mode: str = Field(
        default="binary",
        description="Position mode for BIAS strategy: binary|continuous",
    )
    macd_fast: int = Field(default=12, ge=2, description="MACD fast EMA window")
    macd_slow: int = Field(default=26, ge=2, description="MACD slow EMA window")
    macd_signal: int = Field(default=9, ge=2, description="MACD signal EMA window")
    macd_v_atr_window: int = Field(
        default=26, ge=2, description="ATR window used by MACD-V normalization"
    )
    macd_v_scale: float = Field(
        default=100.0, gt=0.0, description="Scale factor for MACD-V"
    )
    macd_hist_min: float = Field(
        default=0.0,
        ge=0.0,
        description="MACD histogram minimum absolute height filter for macd_cross (0 disables)",
    )
    macd_v_hist_min: float = Field(
        default=0.0,
        ge=0.0,
        description="MACD-V histogram minimum absolute height filter for macd_v (0 disables)",
    )
    er_filter: bool = Field(
        default=False,
        description="Universal ER entry filter switch (when true, allow entry only if ER >= threshold)",
    )
    er_window: int = Field(
        default=10, ge=2, description="ER lookback window (trading days)"
    )
    er_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0, description="ER entry threshold in [0,1]"
    )
    impulse_entry_filter: bool = Field(
        default=False,
        description="Universal Impulse entry filter switch (Elder Impulse System)",
    )
    impulse_allow_bull: bool = Field(
        default=True, description="Allow new long entries in BULL impulse state"
    )
    impulse_allow_bear: bool = Field(
        default=False, description="Allow new long entries in BEAR impulse state"
    )
    impulse_allow_neutral: bool = Field(
        default=False, description="Allow new long entries in NEUTRAL impulse state"
    )
    ma_entry_filter_enabled: bool = Field(
        default=False,
        description="Universal MA cross entry filter switch (allow entry only when fast MA is above slow MA)",
    )
    ma_entry_filter_type: Literal["sma", "ema"] = Field(
        default="sma",
        description="MA type used by universal MA cross entry filter: sma|ema",
    )
    ma_entry_filter_fast: int = Field(
        default=100,
        ge=2,
        description="Fast MA window for universal MA cross entry filter (trading days)",
    )
    ma_entry_filter_slow: int = Field(
        default=200,
        ge=2,
        description="Slow MA window for universal MA cross entry filter (trading days, must be > fast)",
    )
    er_exit_filter: bool = Field(
        default=False,
        description="Universal ER exit filter switch (when true, exit if ER >= threshold)",
    )
    er_exit_window: int = Field(
        default=10, ge=2, description="ER exit filter lookback window (trading days)"
    )
    er_exit_threshold: float = Field(
        default=0.88, ge=0.0, le=1.0, description="ER exit threshold in [0,1]"
    )
    random_hold_days: int = Field(
        default=20,
        ge=1,
        description="Random-entry strategy base exit: hold N trading days after entry",
    )
    random_seed: int | None = Field(
        default=42,
        description="Random-entry strategy seed for reproducible coin-toss signals; null means system random seed",
    )
    quick_mode: bool = Field(
        default=False,
        description="If true, skip heavy post analyses (return decomposition, entry-condition causal stats, trade_statistics raw traces, event study).",
    )

    @model_validator(mode="after")
    def _validate_ma_entry_filter_windows(self) -> "TrendBacktestRequest":
        if int(self.ma_entry_filter_fast) >= int(self.ma_entry_filter_slow):
            raise ValueError("ma_entry_filter_fast must be < ma_entry_filter_slow")
        if bool(self.vol_regime_risk_mgmt_enabled) and bool(
            self.vol_periodic_risk_mgmt_enabled
        ):
            raise ValueError(
                "vol_regime_risk_mgmt_enabled and vol_periodic_risk_mgmt_enabled cannot both be enabled"
            )
        if not math.isfinite(float(self.vol_periodic_rebalance_threshold_pct)):
            raise ValueError("vol_periodic_rebalance_threshold_pct must be finite")
        _validate_overlay_execution_mode_time(
            execution_mode=self.atr_stop_execution_mode,
            execution_time=self.atr_stop_execution_time,
            mode_field="atr_stop_execution_mode",
            time_field="atr_stop_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.r_take_profit_execution_mode,
            execution_time=self.r_take_profit_execution_time,
            mode_field="r_take_profit_execution_mode",
            time_field="r_take_profit_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.r_profit_scaleout_execution_mode,
            execution_time=self.r_profit_scaleout_execution_time,
            mode_field="r_profit_scaleout_execution_mode",
            time_field="r_profit_scaleout_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.bias_v_take_profit_execution_mode,
            execution_time=self.bias_v_take_profit_execution_time,
            mode_field="bias_v_take_profit_execution_mode",
            time_field="bias_v_take_profit_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.ma_trailing_stop_execution_mode,
            execution_time=self.ma_trailing_stop_execution_time,
            mode_field="ma_trailing_stop_execution_mode",
            time_field="ma_trailing_stop_execution_time",
        )
        return self


class TrendPortfolioBacktestRequest(BaseModel):
    codes: list[str] = Field(min_length=1, description="Portfolio candidate codes")
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    initial_account_amount: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional initial trading account amount (CNY). null keeps normalized NAV mode.",
    )
    cost_bps: float = Field(
        default=2.0,
        ge=0.0,
        description="Round-trip transaction cost in bps per turnover",
    )
    slippage_rate: float = Field(
        default=0.001,
        ge=0.0,
        description="One-way adverse slippage spread (absolute price diff)",
    )
    capacity_window_years: Literal[1, 3, 5] = Field(
        default=1,
        description="Capacity statistics window in years: 1|3|5 (default 1 year).",
    )
    exec_price: Literal["open", "close"] = Field(
        default="open",
        description="open|close",
    )
    engine: str | None = Field(
        default=None,
        description="Backtest engine switch: legacy|bt; null uses server default",
    )
    strategy: str = Field(
        default="ma_filter",
        description="ma_filter|ma_cross|donchian|tsmom|linreg_slope|bias|macd_cross|macd_zero_filter|macd_v|random_entry; ma_filter uses ma_type sma|ema|kama",
    )
    position_sizing: str = Field(
        default="equal", description="equal|vol_target|fixed_ratio|risk_budget"
    )
    vol_window: int = Field(
        default=20, ge=2, description="Rolling vol window for vol-target sizing"
    )
    vol_target_ann: float = Field(
        default=0.20, gt=0.0, description="Annualized target vol for portfolio scaling"
    )
    fixed_pos_ratio: float = Field(
        default=0.04,
        gt=0.0,
        description="Fixed position ratio per active asset when position_sizing=fixed_ratio",
    )
    fixed_overcap_policy: str = Field(
        default="skip",
        description="When fixed-ratio entry exceeds constraints: skip|extend",
    )
    fixed_max_holdings: int = Field(
        default=10,
        ge=1,
        description="Max number of held assets when position_sizing=fixed_ratio",
    )
    risk_budget_atr_window: int = Field(
        default=20, ge=2, description="ATR window when position_sizing=risk_budget"
    )
    risk_budget_pct: float = Field(
        default=0.01,
        ge=0.001,
        le=0.03,
        description="Per-asset NAV risk budget for 1 ATR move (0.01 = 1%)",
    )
    risk_budget_overcap_policy: str = Field(
        default="scale",
        description="When risk-budget new entry exceeds total 100% exposure: scale|skip_entry|replace_entry|leverage_entry",
    )
    risk_budget_max_leverage_multiple: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Max leverage multiple when risk_budget_overcap_policy=leverage_entry",
    )
    vol_regime_risk_mgmt_enabled: bool = Field(
        default=False,
        description="Enable post-entry volatility-regime risk management in risk_budget sizing",
    )
    vol_ratio_fast_atr_window: int = Field(
        default=5,
        ge=2,
        description="Fast ATR window for volatility ratio ATR(fast)/ATR(slow)",
    )
    vol_ratio_slow_atr_window: int = Field(
        default=50,
        ge=2,
        description="Slow ATR window for volatility ratio ATR(fast)/ATR(slow)",
    )
    vol_ratio_expand_threshold: float = Field(
        default=1.45,
        gt=0.0,
        description="If ATR ratio > threshold, volatility is expanded and de-risk is triggered",
    )
    vol_ratio_contract_threshold: float = Field(
        default=0.65,
        gt=0.0,
        description="If ATR ratio < threshold, volatility is contracted and add-risk is triggered",
    )
    vol_ratio_normal_threshold: float = Field(
        default=1.05,
        gt=0.0,
        description="Normal-zone recovery threshold used after expanded/contracted states",
    )
    vol_ratio_extreme_threshold: float = Field(
        default=2.2,
        gt=0.0,
        description="If ATR ratio > threshold, enter extreme volatility tier above expanded state",
    )
    vol_periodic_risk_mgmt_enabled: bool = Field(
        default=False,
        description="Enable periodic ATR-based risk-budget rebalance by share-size threshold; mutually exclusive with vol_regime_risk_mgmt_enabled",
    )
    vol_periodic_rebalance_threshold_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum relative share-size change threshold for periodic volatility rebalance (0.05 = 5%)",
    )
    risk_of_ruin_maxrisk: float = Field(
        default=0.30,
        gt=0.0,
        le=1.0,
        description="Configured ruin threshold maxrisk used in RoR formula ((1-P)/P)^(maxrisk/A)",
    )
    dynamic_universe: bool = Field(
        default=False,
        description="If true, allow dynamic candidate pool by period over union interval",
    )
    sma_window: int = Field(default=200, ge=2)
    fast_window: int = Field(default=50, ge=2)
    slow_window: int = Field(default=200, ge=2)
    ma_type: str = Field(
        default="sma",
        description="MA type: ma_filter supports sma|ema|kama; ma_cross supports sma|ema|wma",
    )
    kama_er_window: int = Field(
        default=10, ge=2, description="KAMA ER lookback window (used when ma_type=kama)"
    )
    kama_fast_window: int = Field(
        default=2,
        ge=1,
        description="KAMA fast smoothing window (used when ma_type=kama)",
    )
    kama_slow_window: int = Field(
        default=30,
        ge=2,
        description="KAMA slow smoothing window (used when ma_type=kama)",
    )
    kama_std_window: int = Field(
        default=20,
        ge=2,
        description="KAMA std window for hysteresis band: KAMA ± coef*std(KAMA)",
    )
    kama_std_coef: float = Field(
        default=1.0, ge=0.0, le=3.0, description="KAMA std filter coefficient in [0,3]"
    )
    donchian_entry: int = Field(default=20, ge=2)
    donchian_exit: int = Field(default=10, ge=2)
    mom_lookback: int = Field(default=252, ge=2)
    tsmom_entry_threshold: float = Field(default=0.0)
    tsmom_exit_threshold: float = Field(default=0.0)
    atr_stop_mode: str = Field(
        default="none", description="none|static|trailing|tightening"
    )
    atr_stop_atr_basis: str = Field(default="latest", description="entry|latest")
    atr_stop_reentry_mode: str = Field(
        default="reenter", description="reenter|wait_next_entry"
    )
    atr_stop_execution_mode: str = Field(
        default="intraday",
        description="intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    atr_stop_execution_time: Literal["open", "close", "full_day"] | None = Field(
        default=None,
        description="ATR stop execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
    )
    atr_stop_window: int = Field(default=14, ge=2)
    atr_stop_n: float = Field(default=2.0, gt=0.0)
    atr_stop_m: float = Field(default=0.5, gt=0.0)
    ma_trailing_stop_enabled: bool = Field(
        default=False,
        description="Enable universal MA/EMA trailing-stop overlay",
    )
    ma_trailing_stop_ma_type: str = Field(
        default="sma",
        description="MA trailing-stop type: sma|ema",
    )
    ma_trailing_stop_execution_mode: str = Field(
        default="intraday",
        description="intraday|next_day",
    )
    ma_trailing_stop_execution_time: Literal["open", "close", "full_day"] | None = (
        Field(
            default=None,
            description="MA trailing-stop execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
        )
    )
    ma_trailing_stop_effective_delay_days: int = Field(
        default=3,
        ge=1,
        description="MA trailing-stop activation delay in trading days after entry (3 means active from T+3)",
    )
    ma_trailing_stop_reduce_window: int = Field(default=10, ge=2)
    ma_trailing_stop_exit_window: int = Field(default=20, ge=2)
    ma_trailing_stop_reduce_fraction: float = Field(default=0.33, gt=0.0, le=1.0)
    r_take_profit_enabled: bool = Field(
        default=False,
        description="Enable universal R-multiple drawdown take-profit overlay",
    )
    r_take_profit_reentry_mode: str = Field(
        default="reenter", description="reenter|wait_next_entry"
    )
    r_take_profit_execution_mode: str = Field(
        default="intraday",
        description="intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    r_take_profit_execution_time: Literal["open", "close", "full_day"] | None = Field(
        default=None,
        description="R take-profit execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
    )
    r_take_profit_tiers: list[RTakeProfitTier] | None = Field(default=None)
    r_profit_scaleout_enabled: bool = Field(
        default=False,
        description="Enable tiered floating-profit scale-out overlay (sell on profit level, no pullback required)",
    )
    r_profit_scaleout_execution_mode: str = Field(
        default="intraday",
        description="intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    r_profit_scaleout_execution_time: Literal["open", "close", "full_day"] | None = (
        Field(
            default=None,
            description="R scale-out execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
        )
    )
    r_profit_scaleout_breakeven_stop_enabled: bool = Field(
        default=True,
        description="Enable breakeven-stop addon after first executed floating-profit scale-out",
    )
    r_profit_scaleout_tiers: list[RProfitScaleoutTier] | None = Field(default=None)
    bias_v_take_profit_enabled: bool = Field(
        default=False, description="Enable universal BIAS-V take-profit overlay"
    )
    bias_v_take_profit_reentry_mode: str = Field(
        default="reenter", description="reenter|wait_next_entry"
    )
    bias_v_take_profit_execution_mode: str = Field(
        default="intraday",
        description="intraday|next_day (intraday takes effect from next trading day after entry)",
    )
    bias_v_take_profit_execution_time: Literal["open", "close", "full_day"] | None = (
        Field(
            default=None,
            description="BIAS-V take-profit execution time: intraday supports open|close|full_day; next_day supports open|close only. null keeps compatibility default (intraday->full_day; next_day->exec_price).",
        )
    )
    bias_v_take_profit_breakeven_stop_enabled: bool = Field(
        default=True,
        description="Enable breakeven-stop addon after first executed BIAS-V take-profit tier",
    )
    bias_v_ma_window: int = Field(
        default=20, ge=2, description="MA window in BIAS-V=(close-MA)/ATR"
    )
    bias_v_atr_window: int = Field(
        default=20, ge=2, description="ATR window in BIAS-V=(close-MA)/ATR"
    )
    bias_v_take_profit_tiers: list[BiasVTakeProfitTier] | None = Field(default=None)
    monthly_risk_budget_enabled: bool = Field(
        default=False,
        description="Enable account-level monthly max-loss risk budget gate before new entries",
    )
    monthly_risk_budget_pct: float = Field(
        default=0.06,
        ge=0.01,
        le=0.06,
        description="Monthly max-loss budget on account NAV (0.06 = 6%)",
    )
    monthly_risk_budget_include_new_trade_risk: bool = Field(
        default=False,
        description="If true, include candidate new-trade risk in monthly budget check",
    )
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
    macd_hist_min: float = Field(default=0.0, ge=0.0)
    macd_v_hist_min: float = Field(default=0.0, ge=0.0)
    er_filter: bool = Field(
        default=False, description="Universal ER entry filter switch"
    )
    er_window: int = Field(default=10, ge=2)
    er_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    impulse_entry_filter: bool = Field(
        default=False,
        description="Universal Impulse entry filter switch (Elder Impulse System)",
    )
    impulse_allow_bull: bool = Field(
        default=True, description="Allow new long entries in BULL impulse state"
    )
    impulse_allow_bear: bool = Field(
        default=False, description="Allow new long entries in BEAR impulse state"
    )
    impulse_allow_neutral: bool = Field(
        default=False, description="Allow new long entries in NEUTRAL impulse state"
    )
    ma_entry_filter_enabled: bool = Field(
        default=False,
        description="Universal MA cross entry filter switch (allow entry only when fast MA is above slow MA)",
    )
    ma_entry_filter_type: Literal["sma", "ema"] = Field(
        default="sma",
        description="MA type used by universal MA cross entry filter: sma|ema",
    )
    ma_entry_filter_fast: int = Field(
        default=100,
        ge=2,
        description="Fast MA window for universal MA cross entry filter (trading days)",
    )
    ma_entry_filter_slow: int = Field(
        default=200,
        ge=2,
        description="Slow MA window for universal MA cross entry filter (trading days, must be > fast)",
    )
    er_exit_filter: bool = Field(
        default=False, description="Universal ER exit filter switch"
    )
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
    quick_mode: bool = Field(
        default=False,
        description="If true, skip heavy post analyses (return decomposition, entry-condition causal stats, trade_statistics raw traces, event study).",
    )

    @model_validator(mode="after")
    def _validate_ma_entry_filter_windows(self) -> "TrendPortfolioBacktestRequest":
        if int(self.ma_entry_filter_fast) >= int(self.ma_entry_filter_slow):
            raise ValueError("ma_entry_filter_fast must be < ma_entry_filter_slow")
        if bool(self.vol_regime_risk_mgmt_enabled) and bool(
            self.vol_periodic_risk_mgmt_enabled
        ):
            raise ValueError(
                "vol_regime_risk_mgmt_enabled and vol_periodic_risk_mgmt_enabled cannot both be enabled"
            )
        if not math.isfinite(float(self.vol_periodic_rebalance_threshold_pct)):
            raise ValueError("vol_periodic_rebalance_threshold_pct must be finite")
        _validate_overlay_execution_mode_time(
            execution_mode=self.atr_stop_execution_mode,
            execution_time=self.atr_stop_execution_time,
            mode_field="atr_stop_execution_mode",
            time_field="atr_stop_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.r_take_profit_execution_mode,
            execution_time=self.r_take_profit_execution_time,
            mode_field="r_take_profit_execution_mode",
            time_field="r_take_profit_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.r_profit_scaleout_execution_mode,
            execution_time=self.r_profit_scaleout_execution_time,
            mode_field="r_profit_scaleout_execution_mode",
            time_field="r_profit_scaleout_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.bias_v_take_profit_execution_mode,
            execution_time=self.bias_v_take_profit_execution_time,
            mode_field="bias_v_take_profit_execution_mode",
            time_field="bias_v_take_profit_execution_time",
        )
        _validate_overlay_execution_mode_time(
            execution_mode=self.ma_trailing_stop_execution_mode,
            execution_time=self.ma_trailing_stop_execution_time,
            mode_field="ma_trailing_stop_execution_mode",
            time_field="ma_trailing_stop_execution_time",
        )
        return self


class AssetGroupSuggestRequest(BaseModel):
    codes: list[str] = Field(
        min_length=2, description="Candidate codes for auto grouping"
    )
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(
        default="hfq", description="Price adjust basis for correlation clustering"
    )
    lookback_days: int = Field(
        default=252, ge=20, description="Rolling lookback days for correlation matrix"
    )
    corr_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=0.99,
        description="Absolute correlation threshold for linking two assets",
    )


class RotationCandidateScreenRequest(BaseModel):
    codes: list[str] = Field(
        min_length=2, description="Candidate codes from preset pool"
    )
    start: str = Field(description="YYYYMMDD")
    end: str = Field(description="YYYYMMDD")
    adjust: str = Field(default="hfq", description="Price adjust basis")
    lookback_days: int = Field(
        default=252,
        ge=20,
        le=2520,
        description="Lookback window for scoring/correlation",
    )
    top_n: int = Field(
        default=12, ge=2, le=200, description="Max number of selected assets"
    )
    min_n: int = Field(
        default=4,
        ge=1,
        le=200,
        description="Minimum selected assets (fallback fill by score)",
    )
    max_pair_corr: float = Field(
        default=0.75,
        ge=0.0,
        le=0.99,
        description="Max absolute pairwise correlation among selected assets",
    )
    signif_horizon_days: int = Field(
        default=20,
        ge=5,
        le=252,
        description="Forward horizon for momentum significance test",
    )
    skip_days: int = Field(
        default=0,
        ge=0,
        le=252,
        description="Skip recent N trading days when constructing momentum signal",
    )
    factor_weights: dict[str, float] | None = Field(
        default=None,
        description="Optional factor weights, keys: mom_63,mom_126,sharpe,win_rate,liquidity,mdd",
    )
    category_quotas: dict[str, int] | None = Field(
        default=None,
        description="Optional minimum quota by inferred category, e.g. {'CN_EQ':2,'US_EQ':2}",
    )


class MonteCarloRequest(BaseModel):
    n_sims: int = Field(
        default=10000, ge=50, le=50000, description="Number of Monte Carlo simulations"
    )
    block_size: int = Field(
        default=5, ge=1, le=252, description="Circular block size in trading days"
    )
    seed: int | None = Field(
        default=None, description="Optional RNG seed for reproducibility"
    )
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
    oos_ratio: float = Field(
        default=0.3, gt=0.0, lt=1.0, description="Fraction of period for OOS (at end)"
    )
    n_bootstrap: int = Field(
        default=50, ge=5, le=500, description="Number of bootstrap resamples"
    )
    block_size: int = Field(
        default=21,
        ge=1,
        description="Block size for circular block bootstrap (trading days)",
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
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
    oos_ratio: float = Field(
        default=0.3, gt=0.0, lt=1.0, description="Fraction of period for OOS (at end)"
    )
    n_bootstrap: int = Field(
        default=50, ge=5, le=500, description="Number of bootstrap resamples"
    )
    block_size: int = Field(
        default=21,
        ge=1,
        description="Block size for circular block bootstrap (trading days)",
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    strategy: str = Field(
        default="ma_filter",
        description="ma_filter|ma_cross|donchian|tsmom|linreg_slope|bias|macd_cross|macd_zero_filter|macd_v|random_entry",
    )
    cost_bps: float = Field(default=2.0, ge=0.0)
    exec_price: Literal["open", "close"] = Field(
        default="open", description="open|close"
    )
    engine: str | None = Field(
        default=None,
        description="Backtest engine switch: legacy|bt; null uses server default",
    )
    param_grid: dict[str, list[Any]] | None = Field(
        default=None,
        description="Optional param grid per strategy; if omitted, a default grid is used.",
    )


class LiveAccountCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    base_ccy: str = Field(default="CNY", min_length=1, max_length=16)
    initial_cash: float = Field(default=0.0, ge=0.0)
    notes: str | None = None


class LiveAccountUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    notes: str | None = None


class LiveAccountOut(BaseModel):
    id: int
    name: str
    base_ccy: str
    initial_cash: float
    notes: str | None = None
    created_at: str


class LiveShareholderAccountCreateRequest(BaseModel):
    shareholder_account: str = Field(min_length=1, max_length=64)
    notes: str | None = None


class LiveShareholderAccountOut(BaseModel):
    id: int
    account_id: int
    shareholder_account: str
    notes: str | None = None
    created_at: str


class LiveStrategyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    strategy_type: str = Field(default="etf_spot", description="etf_spot|bond_repo")
    capital_mode: str | None = Field(
        default=None, description="segregated|shared_account_cash"
    )
    notes: str | None = None


class LiveStrategyUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    strategy_type: str | None = Field(default=None, description="etf_spot|bond_repo")
    capital_mode: str | None = Field(
        default=None, description="segregated|shared_account_cash"
    )
    notes: str | None = None


class LiveStrategyOut(BaseModel):
    id: int
    account_id: int
    name: str
    strategy_type: str = "etf_spot"
    capital_mode: str = "segregated"
    notes: str | None = None
    created_at: str


class LiveAccountCashflowCreateRequest(BaseModel):
    flow_date: str = Field(description="YYYYMMDD")
    amount: float = Field(description="Positive deposit, negative withdraw")
    flow_type: str = Field(
        default="deposit",
        description="deposit|withdraw|transfer_to_strategy|transfer_from_strategy|dividend|manual",
    )
    transfer_id: str | None = Field(default=None, max_length=64)
    notes: str | None = None


class LiveStrategyCashflowCreateRequest(BaseModel):
    flow_date: str = Field(description="YYYYMMDD")
    amount: float = Field(description="Positive inflow, negative outflow")
    flow_type: str = Field(
        default="transfer_in", description="transfer_in|transfer_out|dividend|manual"
    )
    transfer_id: str | None = Field(default=None, max_length=64)
    notes: str | None = None


class LiveStrategyTransferRequest(BaseModel):
    strategy_id: int = Field(ge=1)
    flow_date: str = Field(description="YYYYMMDD")
    amount: float = Field(gt=0.0)
    direction: str = Field(
        default="to_strategy", description="to_strategy|from_strategy"
    )
    transfer_id: str | None = Field(default=None, max_length=64)
    notes: str | None = None


class LiveCashflowOut(BaseModel):
    id: int
    account_id: int | None = None
    strategy_id: int | None = None
    flow_date: str
    amount: float
    flow_type: str
    transfer_id: str | None = None
    notes: str | None = None
    created_at: str


class LiveTradeCreateRequest(BaseModel):
    account_id: int = Field(ge=1)
    strategy_id: int = Field(ge=1)
    shareholder_account_id: int = Field(ge=1)
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(default="", max_length=128)
    trade_date: str = Field(description="YYYYMMDD")
    trade_time: str = Field(default="09:30:00", description="HH:MM[:SS]")
    side: str = Field(description="BUY|SELL")
    price: float = Field(gt=0.0)
    quantity: float = Field(gt=0.0)
    fee: float = Field(default=0.0, ge=0.0)
    amount: float | None = Field(default=None, ge=0.0)
    repo_action: str | None = Field(default=None, description="OPEN|CLOSE")
    repo_principal_amount: float | None = Field(default=None, gt=0.0)
    repo_annual_rate_pct: float | None = Field(default=None, gt=0.0)
    repo_interest_days: int | None = Field(default=None, ge=1)
    repo_day_count_basis: int | None = Field(default=None, ge=1)
    repo_open_trade_id: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = Field(default=None, max_length=128)
    broker_trade_no: str | None = Field(default=None, max_length=128)
    notes: str | None = None


class LiveTradeUpdateRequest(BaseModel):
    account_id: int = Field(ge=1)
    strategy_id: int = Field(ge=1)
    shareholder_account_id: int = Field(ge=1)
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(default="", max_length=128)
    trade_date: str = Field(description="YYYYMMDD")
    trade_time: str = Field(default="09:30:00", description="HH:MM[:SS]")
    side: str = Field(description="BUY|SELL")
    price: float = Field(gt=0.0)
    quantity: float = Field(gt=0.0)
    fee: float = Field(default=0.0, ge=0.0)
    amount: float | None = Field(default=None, ge=0.0)
    repo_action: str | None = Field(default=None, description="OPEN|CLOSE")
    repo_principal_amount: float | None = Field(default=None, gt=0.0)
    repo_annual_rate_pct: float | None = Field(default=None, gt=0.0)
    repo_interest_days: int | None = Field(default=None, ge=1)
    repo_day_count_basis: int | None = Field(default=None, ge=1)
    repo_open_trade_id: int | None = Field(default=None, ge=1)
    broker_trade_no: str | None = Field(default=None, max_length=128)
    notes: str | None = None
    reason: str = Field(min_length=1, max_length=500)


class LiveTradeBatchCreateRequest(BaseModel):
    trades: list[LiveTradeCreateRequest] = Field(min_length=1)


class LiveTradeDeleteRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class LiveTradeOut(BaseModel):
    id: int
    account_id: int
    strategy_id: int
    shareholder_account_id: int
    code: str
    name: str
    trade_date: str
    trade_time: str
    side: str
    price: float
    quantity: float
    fee: float
    amount: float
    repo_action: str | None = None
    repo_principal_amount: float | None = None
    repo_annual_rate_pct: float | None = None
    repo_interest_days: int | None = None
    repo_day_count_basis: int | None = None
    repo_open_trade_id: int | None = None
    idempotency_key: str | None = None
    broker_trade_no: str | None = None
    notes: str | None = None
    created_at: str


class LiveCorporateActionCreateRequest(BaseModel):
    account_id: int | None = Field(default=None, ge=1)
    strategy_id: int | None = Field(default=None, ge=1)
    event_type: str = Field(
        description="cash_dividend|split|share_conversion|code_change"
    )
    code: str = Field(min_length=1, max_length=32)
    new_code: str | None = Field(default=None, max_length=32)
    event_date: str = Field(description="YYYYMMDD")
    effective_date: str = Field(description="YYYYMMDD")
    ratio_factor: float | None = Field(default=None, gt=0.0)
    cash_per_share: float | None = Field(default=None, ge=0.0)
    notes: str | None = None


class LiveCorporateActionOut(BaseModel):
    id: int
    account_id: int | None = None
    strategy_id: int | None = None
    event_type: str
    code: str
    new_code: str | None = None
    event_date: str
    effective_date: str
    ratio_factor: float | None = None
    cash_per_share: float | None = None
    notes: str | None = None
    created_at: str


class LiveReplayRequest(BaseModel):
    account_id: int | None = Field(default=None, ge=1)
    strategy_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_scope(self) -> "LiveReplayRequest":
        if self.account_id is None and self.strategy_id is None:
            raise ValueError("account_id or strategy_id is required")
        return self


class LiveHoldingOut(BaseModel):
    snapshot_date: str
    scope_type: str
    scope_id: int
    account_id: int
    strategy_id: int | None = None
    shareholder_account_id: int | None = None
    shareholder_account: str | None = None
    code: str
    name: str
    quantity: float
    cost_price: float | None = None
    market_price: float | None = None
    cost_value: float
    market_value: float | None = None
    pnl_amount: float | None = None
    pnl_rate: float | None = None
    holding_duration_days: int | None = None
    price_missing: bool
    stale_days: int | None = None


class LiveClosedRoundOut(BaseModel):
    id: int
    scope_type: str
    scope_id: int
    account_id: int
    strategy_id: int | None = None
    round_no: int
    code: str
    name: str
    open_date: str
    close_date: str
    holding_duration_days: int | None = None
    buy_count: int
    sell_count: int
    buy_qty: float
    sell_qty: float
    avg_buy_price: float | None = None
    avg_sell_price: float | None = None
    realized_pnl: float
    return_rate: float | None = None
    total_fee: float


class LiveNavPointOut(BaseModel):
    nav_date: str
    equity: float
    cash: float
    market_value: float
    external_flow: float
    trading_fee: float
    nav_twr: float
    nav_dietz: float
    daily_return_twr: float | None = None
    daily_return_dietz: float | None = None
    selection_return: float | None = None
    timing_return: float | None = None
    position_return: float | None = None
    cost_drag_return: float | None = None
    cash_drag_return: float | None = None
    repo_carry_return: float | None = None
    repo_fee_drag_return: float | None = None


class LivePerformanceOut(BaseModel):
    scope_type: str
    scope_id: int
    return_basis: str
    nav: list[LiveNavPointOut]
    dietz_basis_metrics: dict[str, Any]
    twr_basis_metrics: dict[str, Any]


class LiveAttributionOut(BaseModel):
    scope_type: str
    scope_id: int
    daily: list[dict[str, Any]]
    period: dict[str, Any]


class LiveFeeStatsOut(BaseModel):
    scope_type: str
    scope_id: int
    total_fee: float
    buy_fee: float
    sell_fee: float
    avg_fee_per_trade: float
    by_day: list[dict[str, Any]]
