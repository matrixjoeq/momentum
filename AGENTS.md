AGENTS.md

Overview
- This file documents how coding agents should build, lint, test, and style this codebase.
- It also defines naming, error handling, and import conventions to keep the project coherent across contributors.
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

- Maintenance and evolves
  - Add/adjust unit tests when changing public interfaces.
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
