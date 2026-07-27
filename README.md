# Momentum

ETF / 场外基金 / 期货研究与实盘记录平台，提供：

- 数据池管理与行情抓取（ETF、场外基金、期货）
- 多策略研究（Baseline、Rotation、Trend、Calendar Timing、Macro/VIX 扩展）
- 回测与蒙特卡洛 / OOS Bootstrap
- 实盘账户、交易、持仓、回放与归因 API
- 对应的 Web 研究页面与管理页面

## 当前支持功能（代码现状）

### 1) 数据与资产池

- ETF 池：增删改查、单标的/批量抓取、批次查询与回滚
- 场外基金池：增删改查、净值抓取与研究状态管理
- 期货池：增删改查、合约抓取、连续合约合成、研究状态与分组管理
- 全局基准池：管理基准资产及其价格序列

### 2) 研究引擎与分析 API

- Baseline：等权/风险平价/逆波动、日历效应、蒙特卡洛、分布分析
- Rotation：动量轮动、候选筛选、日历效应、next-execution-plan、OOS bootstrap、蒙特卡洛
- Trend：单标的与组合回测、OOS bootstrap、多类风控组件
- Macro/VIX：宏观四步分析、VIX 信号与波动代理择时
- Futures Research：相关性、覆盖率、趋势回测、轮动回测
- Sim GBM：phase1~phase4 及 A/B 显著性分析链路

### 3) 实盘记录（`/api/live/*`）

- 账户/策略管理
- 现金流、策略间转账
- 交易录入（单笔、批量、修改、删除）
- 公司行为、回放、持仓、绩效、归因、费用统计

### 4) Web 页面

- 研究首页：`/research`
- 场外基金：`/off-fund-pool`、`/research/off-fund`
- 期货：`/futures-pool`、`/research/futures`
- 黄金专题：`/research/gold`
- 纳指-VIX：`/research/nasdaq-vix`
- 实盘记录：`/trading-records`

## 快速开始

### 1) 环境准备

```bash
cd /path/to/momentum
export PIP_CONFIG_FILE="$(pwd)/pip.conf"
python3 -m venv .venv
.venv/bin/python3 -m pip install -U pip setuptools wheel
.venv/bin/python3 -m pip install -e ".[dev]"
```

> Windows PowerShell 可使用 `.\.venv\Scripts\python.exe -m pip ...` 与 `-m uvicorn ...` 同步操作。

### 2) 数据库配置（默认 MySQL）

项目当前默认走 MySQL（见 `src/etf_momentum/settings.py`）：

- `MOMENTUM_MYSQL_HOST`（默认 `127.0.0.1`）
- `MOMENTUM_MYSQL_PORT`（默认 `3306`）
- `MOMENTUM_MYSQL_USER`（默认 `momentum`）
- `MOMENTUM_MYSQL_PASSWORD`（默认 `momentum`）
- `MOMENTUM_MYSQL_DB`（默认 `momentum`）
- 可选：`MOMENTUM_DB_URL`（云端可直接给 SQLAlchemy DSN）

建议把本地配置写到 `data/.env.local`（项目已支持自动读取）。

参考模板：`env.example`。

### 3) 启动服务

```bash
.venv/bin/python3 -m uvicorn etf_momentum.app:app --reload --reload-dir ./src --reload-exclude "tests/*" --port 8000
```

常用入口：

- `http://127.0.0.1:8000/`（首页）
- `http://127.0.0.1:8000/research`（研究页）
- `http://127.0.0.1:8000/docs`（OpenAPI 文档）
- `http://127.0.0.1:8000/health`（健康检查）

## 常用 API（示例）

- ETF：
  - `GET /api/etf`
  - `POST /api/etf`
  - `POST /api/etf/{code}/fetch`
  - `POST /api/fetch-all`
- 研究：
  - `POST /api/analysis/baseline`
  - `POST /api/analysis/rotation`
  - `POST /api/analysis/trend`
  - `POST /api/analysis/trend/portfolio`
- 实盘：
  - `POST /api/live/accounts`
  - `POST /api/live/trades`
  - `POST /api/live/replay`
  - `GET /api/live/performance`

## 测试与校验

### 全量测试

```bash
.venv/bin/python3 -m pytest -q
```

### 指定测试

```bash
.venv/bin/python3 -m pytest tests/test_api_analysis_baseline.py -q
```

### 并行测试（可选）

```bash
.venv/bin/python3 -m pytest -q -n auto
```

### 代码风格（可选）

```bash
ruff check src tests
```

## 关键配置项

- `MOMENTUM_DEFAULT_START_DATE` / `MOMENTUM_DEFAULT_END_DATE`
- `MOMENTUM_LOG_LEVEL`
- `MOMENTUM_AUTO_SYNC_ENABLED`（是否开启进程内自动同步）
- `MOMENTUM_SYNC_TOKEN`（保护 `/api/admin/sync/fixed-pool`）
- `MOMENTUM_TREND_BACKTEST_ENGINE`（`legacy` / `bt`）
- `MOMENTUM_FRED_API_KEY`（宏观数据拓展）

## 项目结构

- 源码：`src/etf_momentum`
- 测试：`tests`
- 文档：`docs`
- 脚本：`scripts`

## 备注

- 本 README 反映当前仓库功能与入口；若新增策略/页面/API，请同步更新。
