AGENTS.md

Overview

- This file documents how coding agents should build, lint, test, and style this codebase.
- It also defines naming, error handling, and import conventions to keep the project coherent across contributors.
- **Skill installation:** Any request to install a Skill must follow the mandatory “Installing Skills” workflow in § Security and secrets (vet → report → confirm → install only after approval).
- **Superpowers workflow (mandatory):** For **every** task, read and follow the **using-superpowers** skill (`superpowers`) and run the work through this pipeline in order: **规划** (plan) → **拆解** (decompose) → **执行** (execute) → **审查** (review) → **复盘** (retrospect). Do not skip ahead to coding without planning and decomposition; after changes land, review results (tests, behavior, scope) and briefly retrospect before treating the task as done.
- Cursor rules and Copilot guidelines are included if present in the repository. If not, note their absence.

Quick Start: local dev setup

- Default PyPI index for this repo: Tsinghua mirror via root `pip.conf`. Before `pip` commands from the repo root: `export PIP_CONFIG_FILE="$(pwd)/pip.conf"` (Unix) or set `PIP_CONFIG_FILE` to the repo `pip.conf` path (Windows).
- Create a virtual environment: `python3 -m venv .venv`.
- Activate it: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows).
- Install development dependencies: `python3 -m pip install -e '.[dev]'` (with `PIP_CONFIG_FILE` set as above, or pass `-i https://pypi.tuna.tsinghua.edu.cn/simple`).
- Optional: install tooling for formatting/checking: `pip install ruff black pytest-cov mypy`.
- For packaging: `pip install build` and `python -m build`.

Note: the project uses setuptools with a pyproject.toml and dev extras for tests.

1. Build / Lint / Test commands

- Build distributions: `python -m build` (produces dist/_.whl and dist/_.tar.gz).
- Install in editable mode (dev): `python3 -m pip install -e '.[dev]'` (same index as above).
- **pytest:** Always run tests with the project virtualenv’s Python (not a system interpreter and not bare `pytest` on PATH unless you know it is this venv). After `pip install -e '.[dev]'`, use:
  - Unix / macOS: `.venv/bin/python3 -m pytest …`
  - Windows: `.venv\Scripts\python.exe -m pytest …`
- Run unit tests: `.venv/bin/python3 -m pytest -q` (adjust the interpreter path on Windows as above).
- Run a single test explicitly:
  - `.venv/bin/python3 -m pytest tests/path/to/module.py::TestClass::test_method -q`
  - `.venv/bin/python3 -m pytest tests/path/to/module.py::test_function -q`
- Run tests with coverage:
  - `.venv/bin/python3 -m pytest --cov=src --cov-report=term-missing -q`
- Lint checks (recommended):
  - `ruff check src tests` (also formats with `ruff format`)
  - `black --check src tests` (or `ruff format --exit-nonzero-on-fix` if you use Ruff as formatter)
- Type checks (optional):
  - `mypy src` (requires mypy config if used)
- Quick full verify (lint + tests):
  - `ruff check src tests && .venv/bin/python3 -m pytest -q --maxfail=1 --disable-warnings`

2. Code style guidelines

- Goals: readability, reproducibility, and minimal surprises for new contributors.

- Imports
  - Group imports in this order: standard library, third-party, local application imports.
  - Separate groups with a blank line.
  - Use absolute imports; prefer explicit module paths over wildcard imports.
  - Avoid unused imports; prefer explicit dependencies.
  - If circular dependencies arise, refactor modules into smaller pieces or lazy-import inside functions.

- Formatting
  - Target line length: 88 characters (PEP 8 default) or as configured by project tools.
  - End-of-line newline at file end; no trailing whitespace.
  - Prefer default Python formatting over ad-hoc tweaks; rely on formatters when possible.
  - Use blank lines to separate top-level definitions and logical blocks.

- Types
  - Use typing: List, Dict, Optional, Sequence, Union, Callable as appropriate.
  - For forward refs, enable `from __future__ import annotations` at top of modules.
  - Annotate public APIs and model interfaces; avoid opaque `Any` for public boundaries.

- Naming conventions
  - Functions and variables: snake_case.
  - Classes: CamelCase.
  - Constants: UPPER_SNAKE_CASE.
  - Modules/packages: lowercase; avoid clashes with Python stdlib names.

- Error handling
  - Do not catch broad exceptions (e.g., `except Exception`).
  - Prefer specific error types and document failure modes.
  - Wrap external call failures with contextual messages.
  - Propagate errors upward unless a clear recovery path exists.

- Docstrings
  - Provide concise module, class, and function docstrings.
  - Follow a consistent style (Google or NumPy style). The project currently favors concise one-liners for simple functions and fuller docs for public APIs.

- Testing
  - Tests should be deterministic and fast where possible.
  - Name tests clearly: `test_<feature>_<behavior>`.
  - Use pytest markers for integration tests; keep them opt-in.
  - Tests should not rely on external network calls unless explicitly marked as integration.
  - **API contract tests:** Every route must have coverage (see **API contract test rule** under Maintenance). Contract tests validate HTTP + JSON as real clients send them (including values that look like UI inputs, e.g. percent fields vs Pydantic decimal bounds), not only in-process calls with already-normalized models.
- Logging
  - Use Python `logging` for runtime diagnostics.
  - Do not print directly in library code; use `logger.info`/`logger.debug` instead.
  - Configure a per-module logger: `logger = logging.getLogger(__name__)`.

- API code style (FastAPI)
  - Type hints on request/response models.
  - Use Pydantic models for input validation; avoid ad-hoc dicts for payloads.
  - Path operations should be small; extract business logic into services.
  - Return appropriate HTTP status codes and errors via FastAPI exceptions where sensible.

- Database access (SQLAlchemy 2+)
  - Use the 2.0 style with `select(...)` and result mappings.
  - Session handling via context managers where possible.
  - Model attributes should be explicit and well-documented.

- Security and secrets
  - Do not embed credentials; use environment variables or vaults.
  - Validate inputs and guard against injection risks in SQL and templating.
  - **Installing Skills (mandatory workflow — do not skip):**
    1. **Vet first:** Before installing any Skill (SkillHub, GitHub, or other), run the `skill-vetter` protocol: review the skill’s source and all files for red flags, permissions, and risk level.
    2. **Report:** Share the vetting report with the user (risk level, verdict, notes).
    3. **Confirm:** Explicitly ask for user confirmation. Reject high-risk skills by default; only proceed if the user gives explicit approval.
    4. **Install only after approval:** Run the install step only after the user has approved. Never install a skill before steps 1–3 are complete.
  - Skill installation safety policy (summary):
    - Before installing any Skill, run `skill-vetter` for a security check first.
    - Share the vetting result with the user and explicitly ask for confirmation.
    - By default, reject high-risk Skills; only proceed when the user gives explicit special approval.
    - Only install the Skill after the user explicitly approves.

- Maintenance and evolves
  - Add/adjust unit tests when changing public interfaces.
  - **API contract test rule (mandatory):** Every published HTTP API operation exposed by the app (each distinct HTTP method and path, typically under `/api/...`) must have at least one automated **contract test** that hits the route through the ASGI stack (e.g. Starlette/FastAPI test client) with a JSON body and headers as a real client would. The test must cover the request/response boundary: valid payloads yield the intended success status (usually 200); invalid or client-bug-shaped payloads that the UI might emit (e.g. scaled fields, optional fields omitted vs sent, values that fail Pydantic `Field` constraints) must be covered where missing coverage has caused or could cause production 422/400 regressions. Pure in-process calls that bypass HTTP and only use already-validated Python models do **not** satisfy this rule by themselves. New routes or request-schema changes must add or update the corresponding contract tests in the same change. If a route is intentionally internal-only, deprecated, or exempt, document the exception and rationale in the route docstring or in this file.
  - **Engine consistency rule (mandatory):** For shared strategy concepts (execution timing, corporate-action fallback, NAV compounding, turnover/cost attribution), keep behavior consistent across rotation/trend/holding engines. If one engine changes calculation semantics, update the others (or centralize the logic in shared helpers) and add regression tests proving parity.
  - **Strict execution-timing NAV rule (mandatory):** Daily NAV must follow execution side + price basis consistently across all engines. On an execution day: **open-buy includes same-day return**, **open-sell excludes same-day return**, **close-buy excludes same-day return**, **close-sell includes same-day return**. Implement this with explicit execution-day weight state transitions (pre-trade vs post-trade weights), and add regression tests for entry/exit-day return attribution whenever timing logic is changed.
  - **Price-adjustment rule for NAV computation (mandatory):**
    - Benchmark NAV used for strategy comparison must always be computed with **post-adjusted prices (`hfq`)**.
    - Strategy NAV must always prioritize **raw/unadjusted prices (`none`)** for daily NAV computation.
    - When raw-price series has abnormal jump risk caused by corporate actions (e.g., dividend, split, reverse split), the affected daily return point(s) must switch to **`hfq` fallback** for NAV calculation to avoid artificial discontinuities.
    - Apply the same fallback semantics consistently across all strategy engines and keep the behavior covered by regression tests.
  - **Strategy benchmark NAV rule (mandatory):** Benchmark **return series and NAV** used for comparison charts, excess metrics, and information ratio must follow these definitions (benchmark series are separate from execution-aligned **strategy** returns; benchmark always uses **HFQ (`hfq`) close** prices, consistent with the price-adjustment rule above):
    - **Basic holding / baseline (`compute_baseline`):** The benchmark is **buy-and-hold on the user-specified comparison asset (`benchmark_code`)** using **daily simple returns from HFQ close** compounded to a NAV. Do **not** build the benchmark from the portfolio’s `ret_common`, forward-aligned execution returns, or other strategy price bases.
    - **Single-asset trend backtest and single-asset calendar-timing:** The **BUY_HOLD** comparison line (and excess return vs strategy) uses **exec_price-aligned** buy-and-hold daily returns on the selected symbol: **close** → HFQ **close-to-close**; **open** → same-day **open→close** (none with hfq fallback on corporate-action days); **oc2** → **50%** same-day open→close **+ 50%** HFQ close-to-close next day (same OC2 return blend as other engines). This is **not** always pure HFQ close-to-close when `exec_price` is open or oc2.
    - **All other strategies** (multi-asset trend portfolio, rotation, portfolio calendar timing, holding-enhanced, etc.): The benchmark is a **daily equal-weight portfolio** over the **current candidate pool / group** (whether the pool is **dynamic expanding** or **static intersection**), using **HFQ close** **day-over-day** returns, **rebalanced every trading day**, with **no transaction costs or slippage**. **Dynamic universe:** equal weight across assets that have a valid HFQ close that day (e.g. column mean with `skipna=True`); avoid forward-filling the entire cross-section before daily returns if that would break dynamic-universe meaning. **Static / intersection universe:** after alignment, treat missing asset returns as zero before the cross-sectional equal-weight mean.
    - Prefer shared helpers (e.g. `hfq_close_buy_hold_returns` and `hfq_close_daily_equal_weight_returns` in `analysis/baseline.py`) when implementing or changing benchmarks; keep behavior consistent across engines and add or update regression tests when benchmark semantics change.
  - **Strategy group-selection rule (mandatory):** Any strategy that supports portfolio mode must expose selectable candidate-pool group binding (same group system as other strategies), and must run calculations strictly against the currently selected group.
  - **Strategy parameter persistence rule (mandatory):** All strategy parameters (including mode switches, execution/cost settings, group selection, and strategy-specific controls) must be persisted and restored across page refreshes.
  - **Multi-strategy coverage rule (mandatory):** Multi-strategy composition (MIX) must support all available strategy subtypes in the research page. When a new strategy is introduced, fixed-strategy library save/load and MIX inclusion/aggregation support must be implemented in the same delivery. Users must also be able to persist **named MIX presets** (sub-strategy IDs, order, weight mode, custom weights, and related MIX UI fields) as first-class saved entries alongside single-strategy fixed strategies, so a composition can be reloaded without manually re-adding each leg.
  - **Strategy research parity rule (mandatory):** Every strategy research section on `research.html` must provide all three capabilities at the same time: (1) quick-jump navigation entry, (2) candidate-pool group preset selector, and (3) fixed-strategy library + MIX composition support. New strategy sections are not complete unless these three are implemented together.
  - **Strategy baseline charting/reporting rule (mandatory):** Every strategy page/output (single strategy and portfolio strategy) must include the following baseline analytics set, with benchmark/excess parts shown only when benchmark exists:
    1. Strategy NAV curve and benchmark NAV curve (if any) using log scale; subplot shows strategy NAV RSI.
    2. Strategy-vs-benchmark ratio curve (if benchmark exists) using log scale; main ratio chart must support Bollinger Bands using three middle-band MA windows (MA60, MA120, MA250). Default visible band is MA250; MA60 and MA120 are available as optional overlays. For each MA window, upper/lower bands must be computed from the same window's rolling mean and rolling std (mean ± 2\*std). Subplot shows ratio RSI.
    3. Strategy and benchmark (if any) drawdown curves.
    4. 40-day return spread between strategy and benchmark (if benchmark exists).
    5. Strategy rolling returns for 6m/1y/2y/3y.
    6. Strategy rolling drawdowns for 6m/1y/2y/3y.
    7. Excess rolling returns (if benchmark exists) for 6m/1y/2y/3y.
    8. Excess rolling drawdowns (if benchmark exists) for 6m/1y/2y/3y.
    9. Strategy performance metrics table covering: cumulative return, annualized return, annualized volatility, max drawdown, max-drawdown recovery duration, Sharpe, Sortino, Calmar, Ulcer Index, Ulcer Performance Index, avg daily turnover, avg annual turnover, weekly/monthly/quarterly/yearly win-rate-payoff-Kelly (exclude zero), excess annualized return (if benchmark exists), excess information ratio (if benchmark exists).
    10. Daily return distribution stats (simple and log-return modes): histogram + current-value marker line + stats table with sample size, max, min, mean, std, skewness, kurtosis, quantiles, current value.
    11. Per-trade return distribution (overall and by-asset): stats tables with trade count, win/loss/flat counts, win rate (exclude zero), payoff (exclude zero), Kelly (exclude zero), per-trade max/min/mean/std/quantiles, profit-trade max/min/mean/std (exclude zero), loss-trade max/min/mean/std (exclude zero), and per-trade histogram.
    12. Per-asset holding-period (trading days) distribution: histogram + stats table with min/max/mean/std/quantiles.
    13. Per-asset return contribution, risk contribution, and risk-return ratio.
    14. Period return tables (weekly/monthly/quarterly/yearly) with sort by date/return asc/desc, pagination (12 rows/page), page jump, first/last page controls; include strategy return, benchmark return (if any), excess return (if any).
    15. Current holdings table with code, name, entry date, holding duration (trading days), position weight, holding return.
    16. Next-trading-day plan with planned execution date, buy list (target weight and buy delta), sell list (current weight and sell delta).
    17. Narrative research-report style interpretation section.
  - **Strategy layout rule (mandatory):** Place baseline analytics from overview to detail top-to-bottom; cluster highly related charts/tables together; use 1-3 columns per row for readability and comparison.
  - **Futures trend portfolio sizing (mandatory):** Futures research supports `backtest_mode=portfolio|single`. **Single** uses one listed symbol from the active group with full `position_size_pct` (no cross-sectional sizing). **Portfolio** supports `position_sizing=equal` (1/k among MA-long names each day; weights lag one day into returns, matching ETF trend portfolio semantics) and `position_sizing=risk_budget` (ATR targets with `risk_budget_overcap_policy` including `scale`, `skip_entry`, `replace_entry`, `leverage_entry`, and `risk_budget_max_leverage_multiple`). Portfolio returns are **not** produced by a multi-asset `backtesting.py` session; they combine per-contract single-asset NAV returns with the daily weight matrix (so gross exposure may exceed 1 under leverage policy without misusing `size>1` contract counts).
  - **Futures monthly risk budget gate (mandatory):** When `monthly_risk_budget_enabled` is true **and** `backtest_mode=portfolio`, apply `analysis.trend._apply_monthly_risk_budget_gate` (also re-exported from `analysis.bt_trend` for callers) to the **post-sizing** daily weight matrix (after equal or risk-budget sizing), using execution `Close` and Wilder ATR from execution HLC with `atr_stop_window`. Semantics match ETF trend portfolio: per-calendar-month realized-loss accumulation; open positions carried across months contribute **current** holding risk into the new month’s budget headroom; new-entry veto uses `budget_used + optional_new_trade_risk >= monthly cap` with inclusive boundary; parameters mirror ETF (`monthly_risk_budget_pct` in `[1%,6%]`, `monthly_risk_budget_include_new_trade_risk`, `atr_stop_mode`/`atr_stop_*` for `_position_risk_from_stop_params`). **Single-asset mode** does not apply the gate (`monthly_risk_budget_effective=false`).
  - **Futures trend research price basis (mandatory):** On the futures research trend backtest, separate synthetic continuous series by role: **benchmark NAV / buy-and-hold** uses **backward-adjusted (hfq**, stored as `{root}889`) when synthesized; **signals** (entries/exits and any TA/vol/risk logic built on close) use **forward-adjusted (qfq**, `{root}888`) **close**, aligned by trading date with execution bars; **trade execution and strategy return compounding** use **no-adjustment (none**, `{root}88`) OHLCV in `backtesting.py`. If `{root}88` is absent (main contract only), fall back to the listed symbol’s **none** rows for all three roles and surface `main_contract_none` in API metadata — do not silently mix incompatible bases.
  - **Futures indicators/backtest library rule (mandatory):** For futures backend research implementation, prefer mature libraries before custom code: use **TA-Lib (`talib`)** for technical indicator calculations and use the **`backtesting`** library for backtests. Only implement custom indicator/backtest logic when library capability cannot satisfy required semantics, and document the gap/reason in code comments or PR notes.
  - **TA-Lib/backtesting API verification rule (mandatory):** Before writing or modifying any code that calls **`talib`** or **`backtesting`**, you must first check the corresponding official API documentation to confirm parameter names, return semantics, and execution behavior (for example sizing, order timing, and trade finalization behavior). Do not code against assumptions; verify usage first and then implement.
  - Update AGENTS.md with any new guidelines that emerge.

- Cursor rules
  - Cursor-based rules: none found in this repository.
- Copilot rules
  - Copilot instruction file: not present in this repository.

3. Where to put rules (and how to extend)

- If you add Cursor rules: create `.cursor/rules/your-rule.md` and reference them in this file.
- If you add Copilot rules: add `.github/copilot-instructions.md` with guidance for code generation.

4. Quick wins for maintainers

- Run `.venv/bin/python3 -m pytest -q` locally to verify core behavior before submitting PRs (see the **pytest** bullet in section 1).
- Run `ruff check` and `black --check` to keep code style consistent.
- Add small, focused tests for any new feature or bug fix.
- When adding or changing an API route, add or extend an HTTP-level contract test so every method/path remains covered (see **API contract test rule** above).

Appendix: repository references

- Tests live under `/tests`.
- Source code lives under `/src` with top-level package `etf_momentum`.
- The project uses pyproject.toml to declare packaging and pytest config.

Appendix: web UI (shared stylesheet)

- **Shared theme:** `src/etf_momentum/web/terminal.css` holds the common “research terminal” styles (CSS variables, typography tokens, panels, forms, tables, theme toggle, print rules, and components used by the main research page such as `.tabBtn` / `.reportCard`). Prefer editing this file when changing cross-page look-and-feel instead of duplicating large `<style>` blocks.
- **Serving:** The FastAPI app mounts the `web/` directory at **`/static`** (when that directory exists). Pages load the sheet with `<link rel="stylesheet" href="/static/terminal.css" />`. Example URL: `GET /static/terminal.css`. Contract coverage for this path lives in `tests/test_app_root.py` (`test_static_shared_terminal_css`).
- **Per-page overrides:** Each HTML file should keep a small inline `<style>` after the link for page-only rules (chart heights, tab layout differences, margins). Use the local/system font stack defined in `terminal.css`; do not add external font dependencies (e.g., Google Fonts) in page `<head>`.
- **Exception:** `research_crude_oil.html` uses its own layout and variables; it does not link `terminal.css`. If you change the shared theme, you do not need to mirror every token there unless you intentionally align that page.
- **Local files:** Opening HTML via `file://` will not resolve `/static/terminal.css`; use the dev server (or another HTTP origin) to preview styled pages.
