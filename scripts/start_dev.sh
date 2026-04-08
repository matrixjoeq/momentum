#!/usr/bin/env bash
# Stop local MySQL (if any), start + bootstrap it, then run the ETF Momentum API (uvicorn).
# Requires: bash, mysqld/mysql in PATH, and .venv with the project installed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MYSQL_SOCK="${ROOT_DIR}/data/mysql/mysql.sock"
PY="${ROOT_DIR}/.venv/bin/python"
LOCAL_MYSQL="${ROOT_DIR}/scripts/local_mysql.sh"

if [[ ! -x "$PY" ]]; then
  echo "Virtualenv Python not found: $PY" >&2
  echo "From repo root: python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

cd "$ROOT_DIR"

"${LOCAL_MYSQL}" stop
"${LOCAL_MYSQL}" start

for ((i = 0; i < 20; i++)); do
  if [[ -S "${MYSQL_SOCK}" ]]; then
    break
  fi
  sleep 0.5
done
if [[ ! -S "${MYSQL_SOCK}" ]]; then
  echo "mysqld socket not ready: ${MYSQL_SOCK}" >&2
  exit 1
fi

"${LOCAL_MYSQL}" bootstrap

exec "${PY}" -m uvicorn etf_momentum.app:app \
  --reload \
  --reload-dir "${ROOT_DIR}/src" \
  --reload-exclude "tests/*" \
  --port 8000
