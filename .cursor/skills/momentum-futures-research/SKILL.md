---
name: momentum-futures-research
description: >
  Momentum 工程的期货研究执行规范（trend / rotation / lot-account）。在用 TA-Lib 或
  backtesting 写指标/回测、处理期货 889 连续序列、组合 sizing 与月度风险预算 gate
  时使用。触发词："期货研究"、"futures research"、"trend/rotation 期货"、
  "TA-Lib/talib"、"backtesting 库"、"889 连续合约"、"risk budget/风险预算"、
  "position sizing/ATR"。本技能是 AGENTS.md 期货相关条款的可执行入口。
---

# Momentum 期货研究技能

权威定义以 `AGENTS.md`（Futures trend portfolio sizing、Futures monthly risk
budget gate、Futures trend research price basis、Futures correlation matrix、
Futures indicators/backtest library rule、TA-Lib/backtesting API verification
rule）与 `SOUL.md` 为准；连续合约规则见 `docs/futures-continuous-rules.md`。
冲突时以 `SOUL.md` / `AGENTS.md` 为准。

## 何时使用

实现/修改期货 trend、rotation、lot-account 回测、指标、sizing、风险预算 gate。

## 关键文件

- 引擎：`src/etf_momentum/analysis/futures_trend.py`、`futures_rotation.py`、
  `futures_lot_account.py`；ETF 侧共享 `analysis/trend.py`、`bt_trend.py`
  （`_apply_monthly_risk_budget_gate`）。
- 前端：`src/etf_momentum/web/futures_research.html`、`futures_pool.html`。

## 库使用先验证（强制）

- [ ] 写/改任何 `talib` 或 `backtesting` 调用前，先查官方 API 文档确认参数名、
      返回语义、执行行为（sizing、下单时点、trade finalization）。不凭假设编码。
- [ ] 优先用成熟库（`talib` 指标、`backtesting` 回测）；库无法满足语义才自定义，
      并在代码注释/PR 说明缺口原因。

## 价格基准（强制）

- [ ] 信号、执行/策略收益复利（backtesting.py 与 lot 引擎）、基准买入持有都**只**用
      合成 hfq 连续序列 `{root}889`；缺失则明确报错（不回退 88/888/主力 none）。
- [ ] correlation 用 `{root}889` hfq close 日对数收益 Pearson，标签为池代码。

## 组合 sizing（强制）

- [ ] `backtest_mode=single`：取分组中一个上市合约，用满 `position_size_pct`，无
      横截面 sizing。
- [ ] `backtest_mode=portfolio`：`equal`（每日 MA-long 名单 1/k，权重滞后一天进收益，
      与 ETF trend 组合一致）或 `risk_budget`（ATR 目标 + `risk_budget_overcap_policy`
      = scale/skip_entry/replace_entry/leverage_entry + `risk_budget_max_leverage_multiple`）。
- [ ] 组合收益**不**由多资产 backtesting.py 会话产生，而是单资产 NAV 收益 × 每日权重
      矩阵组合（杠杆下 gross 可 >1，不滥用 size>1 合约数）。

## 月度风险预算 gate（强制）

- [ ] `monthly_risk_budget_enabled=true` 且 `backtest_mode=portfolio` 时，对**post-sizing**
      每日权重矩阵应用 `analysis.trend._apply_monthly_risk_budget_gate`
      （`bt_trend` 也再导出），用执行 Close 与 Wilder ATR（`atr_stop_window`）。
- [ ] 语义对齐 ETF：按自然月累计已实现亏损；跨月持仓把当前持仓风险计入新月预算
      余量；新开仓否决用 `budget_used + optional_new_trade_risk >= 月度上限`（含边界）。
- [ ] 参数镜像 ETF（`monthly_risk_budget_pct` ∈ [1%,6%]、include_new_trade_risk、
      `atr_stop_mode`/`atr_stop_*`）。single 模式不应用 gate。

## 执行步骤

1. 规划/拆解：确定 mode、sizing、是否启用风险预算 gate。
2. 执行前先查 talib/backtesting API 文档。
3. 实现 → 与 ETF 引擎保持 parity（见 `momentum-strategy-research` 引擎一致性）。
4. 审查：跑上方检查清单 + 回归测试 + `code-review`。
5. 复盘：自定义实现的缺口写进注释/PR/`AGENTS.md`。

## 交付证据

查证的 API 行为、价格基准与 sizing/gate 语义、parity 回归测试、跑过的测试。
