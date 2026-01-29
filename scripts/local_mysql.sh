#!/usr/bin/env bash
set -euo pipefail

# Local mysqld helper for this repo.
#
# Goal: run a local MySQL with datadir under ./data/mysql/ so the database lives inside the project.
#
# Requirements:
# - mysqld available in PATH
# - MySQL 8.x recommended
#
# Usage:
#   ./scripts/local_mysql.sh init
#   ./scripts/local_mysql.sh start
#   ./scripts/local_mysql.sh bootstrap
#   ./scripts/local_mysql.sh stop
#   ./scripts/local_mysql.sh status
#
# After init, put connection info into ./data/.env.local (gitignored).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MYSQL_DIR="${ROOT_DIR}/data/mysql"
CONF_FILE="${MYSQL_DIR}/my.cnf"
SOCKET_FILE="${MYSQL_DIR}/mysql.sock"
PID_FILE="${MYSQL_DIR}/mysql.pid"
LOG_FILE="${MYSQL_DIR}/mysql.err"

PORT="${MOMENTUM_MYSQL_PORT:-3306}"

mkdir -p "${MYSQL_DIR}"

load_env_local() {
  local env_file="${ROOT_DIR}/data/.env.local"
  if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    set -a && source "${env_file}" && set +a
  fi
}

write_conf() {
  if [[ -f "${CONF_FILE}" ]]; then
    return
  fi
  cat > "${CONF_FILE}" <<EOF
[mysqld]
datadir=${MYSQL_DIR}/data
socket=${SOCKET_FILE}
pid-file=${PID_FILE}
log-error=${LOG_FILE}
port=${PORT}
bind-address=127.0.0.1
mysqlx=0
skip-name-resolve=1

[client]
socket=${SOCKET_FILE}
port=${PORT}
EOF
}

cmd_init() {
  write_conf
  mkdir -p "${MYSQL_DIR}/data"
  if [[ -d "${MYSQL_DIR}/data/mysql" ]]; then
    echo "already initialized: ${MYSQL_DIR}/data"
    return
  fi
  echo "initializing mysqld datadir=${MYSQL_DIR}/data (insecure, no root password)..."
  mysqld --defaults-file="${CONF_FILE}" --initialize-insecure
  echo "init done. now run: $0 start"
}

cmd_start() {
  load_env_local
  write_conf
  mkdir -p "${MYSQL_DIR}/data"
  if [[ -f "${PID_FILE}" ]]; then
    echo "pid file exists: ${PID_FILE} (already running?)"
    exit 0
  fi
  echo "starting mysqld ..."
  # Prefer --daemonize if supported (MySQL 8). Fallback: background process.
  if mysqld --help --verbose 2>/dev/null | grep -q -- "--daemonize"; then
    mysqld --defaults-file="${CONF_FILE}" --daemonize
  else
    nohup mysqld --defaults-file="${CONF_FILE}" >/dev/null 2>&1 &
  fi
  echo "mysqld started. socket=${SOCKET_FILE} port=${PORT}"
}

cmd_bootstrap() {
  load_env_local
  write_conf
  if [[ ! -S "${SOCKET_FILE}" ]]; then
    echo "mysql socket not found: ${SOCKET_FILE}. run: $0 start"
    exit 1
  fi
  local db="${MOMENTUM_MYSQL_DB:-momentum}"
  local user="${MOMENTUM_MYSQL_USER:-momentum}"
  local pass="${MOMENTUM_MYSQL_PASSWORD:-momentum}"
  echo "bootstrapping database/user on local mysqld (socket auth)..."
  mysql --protocol=SOCKET --socket="${SOCKET_FILE}" -uroot <<EOF
CREATE DATABASE IF NOT EXISTS \`${db}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '${user}'@'127.0.0.1' IDENTIFIED BY '${pass}';
CREATE USER IF NOT EXISTS '${user}'@'localhost' IDENTIFIED BY '${pass}';
GRANT ALL PRIVILEGES ON \`${db}\`.* TO '${user}'@'127.0.0.1';
GRANT ALL PRIVILEGES ON \`${db}\`.* TO '${user}'@'localhost';
FLUSH PRIVILEGES;
EOF
  echo "bootstrap done. db=${db} user=${user}"
}

cmd_stop() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "no pid file: ${PID_FILE} (not running?)"
    exit 0
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  echo "stopping mysqld pid=${pid} ..."
  kill "${pid}" || true
  sleep 1
  if [[ -f "${PID_FILE}" ]]; then
    rm -f "${PID_FILE}"
  fi
  echo "stopped."
}

cmd_status() {
  if [[ -f "${PID_FILE}" ]]; then
    echo "RUNNING pid=$(cat "${PID_FILE}") socket=${SOCKET_FILE} port=${PORT}"
  else
    echo "STOPPED"
  fi
}

case "${1:-}" in
  init) cmd_init ;;
  start) cmd_start ;;
  bootstrap) cmd_bootstrap ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: $0 {init|start|bootstrap|stop|status}"; exit 2 ;;
esac

