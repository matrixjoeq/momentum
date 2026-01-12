# 微信小程序项目文档：ETF 周度动量轮动（模拟实盘）

## 1. 项目概述
- **项目名称**：Momentum Rotation Mini-Program（模拟实盘）
- **目标**：将现有轮动研究提炼为微信小程序，用于**长期跟踪、记录、复盘** 4 只 ETF 的周度轮动模拟实盘交易结果，并同时比较 **周一~周五** 5 个调仓日方案。
- **候选池（固定）**：
  - `159915` 创业板ETF（易方达）
  - `511010` 国债ETF（国泰）
  - `513100` 纳指ETF（国泰）
  - `518880` 黄金ETF（华安）

## 2. 策略规格（固定）
### 2.1 固定策略参数
- **调仓频率**：周频（weekly）
- **TopK**：1
- **动量回看**：20 交易日
- **skip_days**：0
- **交易成本**：0 bps
- **风控开关**：全部关闭（risk_off/trend/rsi/vol/chop/corr/inertia/rr/dd/timing 均关闭）
- **执行价口径**：开盘价（open）
- **休市处理**：若锚点日非交易日，**提前至上一交易日**（shift=prev）
- **五套方案**：同时模拟 weekly anchor weekday = 0..4（MON..FRI）

### 2.2 定义与约定
- **决策日 / 执行日**：策略在“决策日”生成下一段持仓，**从下一交易日开始生效**（与现有后端一致）。
- **净值**：以执行价口径的日收益推进（本项目固定为 open 口径）。
- **数据字段**：行情只需要 **open/close**（high/low 可选）。

## 3. 当前后端能力（已落地）
### 3.1 五套周频开盘价模拟接口（已实现）
- **接口**：`POST /api/analysis/rotation/weekly5-open`
- **输入**：`{ "start": "YYYYMMDD", "end": "YYYYMMDD" }`
- **行为**：固定 4 标的 + 固定参数，返回周一~周五 5 套完整回测结果。
- **输出**：
  - `meta`：固定配置摘要
  - `by_anchor`：`{"0": res0, ..., "4": res4}`，每个 `resX` 为完整 `compute_rotation_backtest` payload（含 `nav/metrics/period_details/...`）

### 3.2 交易日历模块（已实现）
- `src/etf_momentum/calendar/trading_calendar.py`
- 关键函数：
  - `is_trading_day(date)`
  - `shift_to_trading_day(date, shift="prev|next")`
  - `trading_days(start, end)`
- 依赖：`exchange-calendars`

### 3.3 数据抓取与校验（已调整）
- 抓取：只强制要求 open/close；high/low 可缺省。
- ingestion 默认 `max_abs_return=2.0`（若 pool 未设置校验策略）。

## 4. 系统架构与模块边界
### 4.1 架构
- **后端服务**：FastAPI（本仓库）+ SQLite（本地/云端可迁移）
- **小程序**：微信小程序（原生或 uni-app），仅通过 HTTP 调用后端 API
- **数据源**：akshare（ETF 日线）+ exchange-calendars（交易日历）

### 4.2 模块划分
- **Market Data**
  - 抓取 ETF open/close（可扩展）
  - 数据质量校验与入库
- **Calendar**
  - 交易日判断、决策日计算（用于实盘模拟提醒）
- **Strategy Compute**
  - 运行 fixed strategy 的 5 套 backtest（已实现）
  - 输出 period_details（供小程序复盘展示）
- **Simulation Ledger（本期新增重点）**
  - 把“策略建议/回测输出”落为“可审计的模拟成交、持仓、净值快照”
  - 支持用户确认成交（模拟实盘执行）

## 5. 数据库设计（SQLite，建议 schema）
> 说明：现有库已有 ETF pool/price/ingestion batch 等表；下面是小程序“模拟实盘记账”需要新增的业务表。

### 5.1 表：`sim_portfolio`
- **用途**：一个模拟账户（可支持多个账户；默认 1 个）。
- 字段（建议）：
  - `id` (PK)
  - `name`（如“默认账户”）
  - `base_ccy`（默认 CNY）
  - `initial_cash`（默认 1_000_000）
  - `created_at`

### 5.2 表：`sim_strategy_config`
- **用途**：策略配置版本（本项目固定参数，但仍建议写入版本用于审计）。
- 字段：
  - `id` (PK)
  - `portfolio_id` (FK sim_portfolio.id)
  - `codes_json`（固定 4 标的）
  - `rebalance`（固定 weekly）
  - `lookback_days`（固定 20）
  - `top_k`（固定 1）
  - `exec_price`（固定 open）
  - `rebalance_shift`（固定 prev）
  - `risk_controls_json`（固定 all_off）
  - `created_at`

### 5.3 表：`sim_variant`
- **用途**：五套方案（周一~周五）中的一个。
- 字段：
  - `id` (PK)
  - `portfolio_id` (FK)
  - `config_id` (FK sim_strategy_config.id)
  - `anchor_weekday`（0..4）
  - `label`（MON..FRI）
  - `is_active`（用户选择“主跟踪方案”）

### 5.4 表：`sim_decision`
- **用途**：每次周度决策的快照（信号、目标持仓、理由）。
- 字段：
  - `id` (PK)
  - `variant_id` (FK sim_variant.id)
  - `decision_date`（策略决策日）
  - `effective_date`（下一交易日）
  - `picked_code`（Top1）
  - `scores_json`（4 标的分数）
  - `prev_code`（上期持仓）
  - `reason_json`（本项目可简化：仅“top1_by_momentum”）
  - `created_at`

### 5.5 表：`sim_trade`
- **用途**：模拟成交记录（用户确认后产生）。
- 字段：
  - `id` (PK)
  - `variant_id` (FK)
  - `trade_date`
  - `code`
  - `side`（BUY/SELL）
  - `price`（固定 open）
  - `qty`（可选：用份额；或用金额）
  - `amount`（成交金额）
  - `decision_id`（FK sim_decision.id，可为空）
  - `created_at`

### 5.6 表：`sim_position_daily`
- **用途**：每日持仓快照（用于净值曲线与复盘）。
- 字段：
  - `id` (PK)
  - `variant_id` (FK)
  - `trade_date`
  - `positions_json`（{code: qty}）
  - `cash`
  - `nav`
  - `mdd`（可选）
  - `created_at`

## 6. 后端 API 设计（工程化清单）
> 说明：现有 `/api/analysis/rotation/weekly5-open` 可作为“计算引擎”；下面定义小程序需要的“记账/查询/任务”API。

### 6.1 账户与方案
- `POST /api/sim/portfolio`
  - 创建模拟账户（initial_cash）
- `GET /api/sim/portfolio`
  - 列表
- `POST /api/sim/portfolio/{id}/init-fixed-strategy`
  - 初始化固定策略 + 五个 variant（MON..FRI）
- `POST /api/sim/variant/{id}/set-active`
  - 选择主跟踪方案（is_active=true，其它 false）

### 6.2 数据更新与决策生成
- `POST /api/sim/sync-market`
  - 更新 4 标的行情（open/close）
  - 返回：更新日期范围、异常
- `POST /api/sim/compute/weekly5-open`
  - 调用 `/api/analysis/rotation/weekly5-open` 计算指定区间结果
  - 产出：用于写入 decision/对比展示（不直接记账）
- `POST /api/sim/decision/generate`
  - 对每个 variant 生成“下一期决策”（decision_date/effective_date/picked_code/scores）
  - 需要交易日历：预先判断下一次决策日（shift=prev）

### 6.3 执行与记账
- `POST /api/sim/trade/preview`
  - 给定 variant + decision_id，生成买卖清单（从当前持仓 → 目标持仓）
- `POST /api/sim/trade/confirm`
  - 用户确认后写入 sim_trade，并更新当日/后续 position_daily（或进入队列异步结算）
- `POST /api/sim/mark-to-market`
  - 按每日 open/close 推进净值与持仓快照（建议每日收盘后跑）

### 6.4 查询与展示
- `GET /api/sim/variant/{id}/status`
  - 当前持仓、现金、nav、下一次决策日
- `GET /api/sim/variant/{id}/nav?start&end`
  - 净值曲线
- `GET /api/sim/variant/{id}/decisions?start&end`
  - 周度决策列表
- `GET /api/sim/variant/{id}/trades?start&end`
  - 成交列表

## 7. 小程序页面（信息架构与交互）
### 7.1 页面列表
- **P0：首页（今日）**
  - 今日是否交易日
  - 5 套方案：当前持仓 + nav + 近 4 周收益简表
  - 主跟踪方案高亮
- **P1：五套对比**
  - 5 套方案净值曲线（可切换显示）
  - 指标卡：累计收益、最大回撤、周度胜率（可后续）
- **P2：方案详情（单方案）**
  - 当前持仓、下一次决策日
  - 周度决策列表（点进可看当期分数与换仓）
  - 成交与持仓流水
- **P3：执行确认**
  - 显示“本期应卖/应买”、参考开盘价、金额/份额
  - 一键确认 -> 写入 sim_trade
- **P4：设置**
  - 选择主跟踪方案（MON..FRI）
  - 初始资金、记账方式（按金额/按份额）等（V1）

### 7.2 核心交互（最短路径）
- 打开小程序 → 首页查看建议
- 到执行日 → 进入执行确认 → 一键确认 → 自动记账更新
- 平时看净值/复盘：方案详情页

## 8. 任务调度与运行方式
### 8.1 本地 / 单机模式（MVP）
- 小程序提供“手动刷新数据/重新计算/生成决策”按钮
- 后端不依赖定时任务也能工作

### 8.2 线上模式（V1）
- 每日定时：
  - T+1 早上：更新行情 + 若为执行日则生成 trade preview
  - 收盘后：mark-to-market 更新净值
- 需要：
  - 后端部署在可访问域名（HTTPS）
  - 小程序后台配置合法域名

## 9. 验收标准（可量化）
### 9.1 功能验收
- 能在小程序内看到 5 套周度调仓方案的实时状态与历史净值
- 能在执行日完成“模拟成交确认”，并在台账中追溯（decision → trades → positions → nav）
- 休市/周末场景：下一次决策日能按交易日历提前计算（shift=prev）

### 9.2 一致性验收
- “后端计算输出的目标持仓”与“小程序生成的成交单”一致
- 每条成交可追溯到对应的决策快照（或标记为手工成交）

## 10. 开发里程碑（建议）
### M1（1~2 周）：后端记账 + 小程序 MVP
- 后端：新增 sim_* 表与 API（portfolio/variant/decision/trade/nav）
- 小程序：P0/P2/P3 最小闭环

### M2（1 周）：自动化与体验增强
- 定时任务、提醒、导出、异常告警（数据缺口/抓取失败）

## 11. 风险与对策
- **交易日历准确性**：exchange-calendars 覆盖大多数节假日；极端临时休市需要补数据源或手工维护 override。
- **开盘价执行假设**：真实成交会有滑点；后续可加入滑点与分批成交模型。
- **数据源变动**：akshare 字段变化需容错；已降级为 open/close 必选。

## 12. 附录：与仓库代码的映射
- 固定策略五套接口：`src/etf_momentum/api/routes.py`（`/analysis/rotation/weekly5-open`）
- 固定策略核心回测：`src/etf_momentum/strategy/rotation.py`
- 日历效应（可参考）：`src/etf_momentum/analysis/calendar_effect.py`
- 交易日历模块：`src/etf_momentum/calendar/trading_calendar.py`
- 数据抓取：`src/etf_momentum/data/akshare_fetcher.py`
- 入库校验：`src/etf_momentum/data/ingestion.py`、`src/etf_momentum/validation/price_validate.py`

