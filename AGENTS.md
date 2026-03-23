AGENTS.md

Overview
- This file documents how coding agents should build, lint, test, and style this codebase.
- It also defines naming, error handling, and import conventions to keep the project coherent across contributors.
- **Skill installation:** Any request to install a Skill must follow the mandatory “Installing Skills” workflow in § Security and secrets (vet → report → confirm → install only after approval).
- Cursor rules and Copilot guidelines are included if present in the repository. If not, note their absence.

Quick Start: local dev setup
- Create a virtual environment: `python -m venv .venv`.
- Activate it: `source .venv/bin/activate` (Unix) or `.
\venv\Scripts\activate` (Windows).
- Install development dependencies: `pip install -e '.[dev]'`.
- Optional: install tooling for formatting/checking: `pip install ruff black pytest-cov mypy`.
- For packaging: `pip install build` and `python -m build`.

Note: the project uses setuptools with a pyproject.toml and dev extras for tests.

1) Build / Lint / Test commands
- Build distributions: `python -m build` (produces dist/*.whl and dist/*.tar.gz).
- Install in editable mode (dev): `pip install -e '.[dev]'`.
- Run unit tests: `pytest -q`.
- Run a single test explicitly:
  - `pytest tests/path/to/module.py::TestClass::test_method -q`
  - `pytest tests/path/to/module.py::test_function -q`
- Run tests with coverage:
  - `pytest --cov=src --cov-report=term-missing -q`
- Lint checks (recommended):
  - `ruff check src tests` (also formats with `ruff format`)
  - `black --check src tests` (or `ruff format --exit-nonzero-on-fix` if you use Ruff as formatter)
- Type checks (optional):
  - `mypy src` (requires mypy config if used)
- Quick full verify (lint + tests):
  - `ruff check src tests && pytest -q --maxfail=1 --disable-warnings`

2) Code style guidelines
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
  - **Engine consistency rule (mandatory):** For shared strategy concepts (execution timing, corporate-action fallback, NAV compounding, turnover/cost attribution), keep behavior consistent across rotation/trend/holding engines. If one engine changes calculation semantics, update the others (or centralize the logic in shared helpers) and add regression tests proving parity.
  - **Price-adjustment rule for NAV computation (mandatory):**
    - Benchmark NAV used for strategy comparison must always be computed with **post-adjusted prices (`hfq`)**.
    - Strategy NAV must always prioritize **raw/unadjusted prices (`none`)** for daily NAV computation.
    - When raw-price series has abnormal jump risk caused by corporate actions (e.g., dividend, split, reverse split), the affected daily return point(s) must switch to **`hfq` fallback** for NAV calculation to avoid artificial discontinuities.
    - Apply the same fallback semantics consistently across all strategy engines and keep the behavior covered by regression tests.
  - **Strategy group-selection rule (mandatory):** Any strategy that supports portfolio mode must expose selectable candidate-pool group binding (same group system as other strategies), and must run calculations strictly against the currently selected group.
  - **Strategy parameter persistence rule (mandatory):** All strategy parameters (including mode switches, execution/cost settings, group selection, and strategy-specific controls) must be persisted and restored across page refreshes.
  - **Multi-strategy coverage rule (mandatory):** Multi-strategy composition (MIX) must support all available strategy subtypes in the research page. When a new strategy is introduced, fixed-strategy library save/load and MIX inclusion/aggregation support must be implemented in the same delivery.
  - **Strategy baseline charting/reporting rule (mandatory):** Every strategy page/output (single strategy and portfolio strategy) must include the following baseline analytics set, with benchmark/excess parts shown only when benchmark exists:
    1. Strategy NAV curve and benchmark NAV curve (if any) using log scale; subplot shows strategy NAV RSI.
    2. Strategy-vs-benchmark ratio curve (if benchmark exists) using log scale; main ratio chart must support Bollinger Bands using three middle-band MA windows (MA20, MA60, MA250). Default visible band is MA250; MA20 and MA60 are available as optional overlays. For each MA window, upper/lower bands must be computed from the same window's rolling mean and rolling std (mean ± 2*std). Subplot shows ratio RSI.
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
  - Update AGENTS.md with any new guidelines that emerge.

- Cursor rules
  - Cursor-based rules: none found in this repository.
- Copilot rules
  - Copilot instruction file: not present in this repository.

3) Where to put rules (and how to extend)
- If you add Cursor rules: create `.cursor/rules/your-rule.md` and reference them in this file.
- If you add Copilot rules: add `.github/copilot-instructions.md` with guidance for code generation.

4) Quick wins for maintainers
- Run `pytest -q` locally to verify core behavior before submitting PRs.
- Run `ruff check` and `black --check` to keep code style consistent.
- Add small, focused tests for any new feature or bug fix.

Appendix: repository references
- Tests live under `/tests`.
- Source code lives under `/src` with top-level package `etf_momentum`.
- The project uses pyproject.toml to declare packaging and pytest config.
