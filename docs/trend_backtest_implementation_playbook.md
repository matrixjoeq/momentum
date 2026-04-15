# Trend Backtest Implementation Playbook

This document turns the five-round review plan into executable standards for
futures group trend-following backtests.

## 1) Scope Lock (Confirmed)

### 1.1 Confirmed asset domain and engine

- Asset domain: **futures only**.
- Group source: `futures_research` active group.
- Data basis: `FuturesPrice` with `adjust=none` only.
- Engine path: `analysis/futures_trend.py` + `/api/futures/research/trend-backtest`.
- Library priority:
  - indicators: **TA-Lib first**
  - backtest execution: **backtesting first**
  - fallback implementation only when these libraries are unavailable.

### 1.2 Confirmed execution semantics

- Signal and execution both use non-adjusted futures data (`none`).
- Supported execution price:
  - `close` (default)
  - `open`
- Universe mode binding:
  - `dynamic_universe=true` -> dynamic union aggregation
  - `dynamic_universe=false` -> static intersection aggregation

### 1.3 Confirmed cost semantics

- `cost_bps` represents commission with explicit side semantics:
  - `fee_side=one_way`: input value is applied per fill.
  - `fee_side=two_way`: input value is round-trip total; per-fill value is half.
- `slippage_type` supports:
  - `percent`: direct relative spread ratio
  - `price_spread`: absolute spread converted to ratio by symbol price reference
- `slippage_side` semantics are aligned with `fee_side`.
- Baseline profile:
  - `cost_bps=5`
  - `fee_side=two_way`
  - `slippage_type=percent`
  - `slippage_value=0.0005`
  - `slippage_side=two_way`

## 2) Data Readiness Standard

Run before each futures group trend backtest.

### 2.1 Mandatory checks

- Per-symbol minimum observations (default >= 252; override by run context).
- Date coverage and requested window consistency.
- Missing ratio against group union timeline.
- Extreme one-day absolute return sentinel.
- `none` series availability for all selected futures symbols.

### 2.2 Operational command

From repository root:

- `python3 scripts/check_trend_backtest_readiness.py --asset-domain futures --codes RB0,IF0 --start 20180101 --end 20251231`

The command returns non-zero when hard readiness checks fail.

## 3) Engine Contract Freeze

### 3.1 Request contract

Use `FuturesTrendBacktestRequest` as the source of truth for validation:

- Universe and window:
  - `group_name` (optional, fallback active group)
  - `range_key` / `start_date` / `end_date`
  - `dynamic_universe`
- Strategy and execution:
  - `exec_price`
  - `fast_ma` / `slow_ma`
  - `position_size_pct`
  - `min_points`
- Cost model:
  - `cost_bps`, `fee_side`
  - `slippage_type`, `slippage_value`, `slippage_side`

### 3.2 Response contract

- `meta`: fixed semantics + effective runtime parameters.
- `series`:
  - `strategy_nav`
  - `benchmark_nav` (group equal-weight by selected universe mode)
- `summary`:
  - strategy total return
  - benchmark total return
  - excess total return
- `symbols`:
  - per-symbol points, window, return, trades, cost normalization.

### 3.3 Error boundary contract

Must return explicit failures for:

- empty group / no active group
- invalid enum values (`exec_price`, cost/slippage sides/types)
- invalid MA constraints (`slow_ma <= fast_ma`)
- insufficient data points
- invalid date/range inputs

## 4) Validation Framework

### 4.1 Validation levels

1. **API contract**:
   - success payload
   - invalid semantics payload
2. **Engine behavior**:
   - cost-side normalization
   - dynamic vs static aggregation path
3. **Robustness protocol**:
   - IS/OOS + walk-forward (to be applied in research workflow)

### 4.2 Current mandatory test set

- `tests/test_api_futures_research.py` (includes trend-backtest contract tests)
- `tests/test_trend_backtest_readiness_checker.py`
- Existing trend regression suite retained for baseline comparison:
  - `tests/test_analysis_trend.py`
  - `tests/test_analysis_trend_portfolio.py`
  - `tests/test_strategy_execution_timing_regression.py`
  - `tests/test_slippage_spread_semantics.py`

Recommended command:

- `./.venv/bin/python3 -m pytest -q tests/test_api_futures_research.py tests/test_trend_backtest_readiness_checker.py tests/test_analysis_trend.py tests/test_analysis_trend_portfolio.py tests/test_strategy_execution_timing_regression.py tests/test_slippage_spread_semantics.py`

## 5) Production Gate

All gates must pass before exposing futures trend results as stable output.

### 5.1 Gate checklist

- **Scope gate:** request/response semantics match this document.
- **Data gate:** readiness checker passed for the same group and time range.
- **Contract gate:** futures trend API contract tests passed.
- **Regression gate:** no breakage in existing strategy timing/cost regressions.
- **Repro gate:** run metadata retained for replay.
- **Docs gate:** cost side semantics and execution basis explicitly disclosed.

### 5.2 Release artifact package

Each release must include:

- Decision record:
  - exec basis (`open`/`close`)
  - dynamic/static mode
  - fee/slippage side semantics
- Data snapshot summary:
  - group symbols
  - effective symbol count
  - skipped symbols and reasons
- Validation report:
  - test command
  - pass/fail summary
  - known limitations

## 6) Known Limitations (Current Implementation)

- Current futures benchmark is equal-weight buy-and-hold on `none` close returns
  under selected dynamic/static mode.
- Roll-aware continuous-contract semantics are not yet implemented and should be
  treated as a dedicated follow-up enhancement.
