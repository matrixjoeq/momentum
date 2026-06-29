---
name: momentum-strategy-research
description: >
  Momentum 工程的策略研究执行规范（ETF/场外基金/期货组合策略）。在 research.html
  或 analysis/ 引擎里新增或修改任何策略段、基准、NAV、归因、图表/报表时使用。
  触发词："新增策略"、"修改策略研究"、"加一个策略段"、"strategy research"、
  "research.html 策略"、"基准 NAV"、"baseline 图表"、"次日计划"、"固定策略库/MIX"。
  本技能是 AGENTS.md 相关强约束条款的可执行入口，不取代条款本身。
---

# Momentum 策略研究技能

权威定义以仓库根目录 `AGENTS.md`（§ Maintenance and evolves）与 `SOUL.md` 为准。
本技能把这些强约束转成"何时触发 + 执行步骤 + 检查清单 + 交付证据"。冲突时以
`SOUL.md` / `AGENTS.md` 为准。

## 何时使用

- 在 `src/etf_momentum/web/research.html` 新增或修改任何策略研究段。
- 在 `src/etf_momentum/analysis/` 修改策略引擎（`rotation.py` / `trend.py` /
  `bt_trend.py` / `baseline.py` / `calendar_timing_strategy.py` 等）。
- 改动基准、NAV、执行时点、归因、图表/报表、参数持久化、分组绑定、MIX。

## 关键文件

- 引擎：`src/etf_momentum/analysis/rotation.py`、`trend.py`、`bt_trend.py`、
  `baseline.py`（含 `hfq_close_buy_hold_returns`、
  `hfq_close_daily_equal_weight_returns`）、`calendar_timing_strategy.py`。
- 前端：`src/etf_momentum/web/research.html`、共享样式 `web/terminal.css`。
- 路由：`src/etf_momentum/api/routes.py`。

## 执行步骤（规划 → 拆解 → 执行 → 审查 → 复盘）

1. 规划：用 `brainstorming` 明确策略语义（候选池/分组、执行时点、成本、基准）。
2. 拆解：列出受影响引擎、API、前端段、测试。
3. 执行：实现引擎 → API（Pydantic 模型）→ research.html 段 → 测试。
4. 审查：跑下方检查清单 + `code-review`。
5. 复盘：记录新出现的语义到 `AGENTS.md`，必要时用 `self-improving`。

## 检查清单（交付前逐条核对）

研究段三件套（research.html 每个策略段必须同时具备）：

- [ ] 快捷导航入口。
- [ ] 候选池分组预设选择器（严格按当前选中分组计算）。
- [ ] 固定策略库保存/加载 + MIX 组合支持（新策略与 MIX 聚合同批交付）。

参数与分组：

- [ ] 所有参数（模式开关、执行/成本、分组、策略专属控件）页面刷新后可恢复。
- [ ] 组合模式暴露可选分组绑定，且只对选中分组计算。

基准 NAV（始终用 `hfq` close；区分场景）：

- [ ] baseline/基础持有：买入持有 `benchmark_code`，HFQ close 日简单收益复利。
- [ ] 单资产 trend / 单资产日历择时：BUY_HOLD 用 exec_price 对齐
      （close→C2C；open→当日 O→C；oc2→50% O→C + 50% 次日 C2C）。
- [ ] 其它组合策略：候选池/分组每日等权 HFQ close 日间收益，无成本；动态池
      用当日有效 close 列均值（skipna），静态池缺失收益按 0。
- [ ] 优先复用 `analysis/baseline.py` 的共享 helper。

执行时点 NAV（执行日，pre/post-trade 权重显式切换）：

- [ ] open-buy 含当日收益；open-sell 不含；close-buy 不含；close-sell 含。
- [ ] 价格复权：策略 NAV 优先 `none`，公司行动跳变点回退 `hfq`；基准恒用 `hfq`。

引擎一致性：

- [ ] 执行时点、公司行动回退、NAV 复利、换手/成本归因在 rotation/trend/holding
      间一致；改一处需同步其它或抽到共享 helper，并加回归测试证明 parity。

Baseline 图表/报表集（17 项，benchmark 部分仅在有基准时显示）：

- [ ] 1 NAV 曲线(对数,含 RSI 子图) 2 比值曲线(对数,Bollinger MA60/120/250,默认
      MA250,含 RSI) 3 回撤曲线 4 40 日收益差 5 滚动收益 6m/1y/2y/3y 6 滚动回撤
      7 超额滚动收益 8 超额滚动回撤 9 绩效指标表（累计/年化/波动/最大回撤/恢复/
      Sharpe/Sortino/Calmar/Ulcer/UPI/换手/周月季年胜率赔率 Kelly/超额年化/信息比）
      10 日收益分布(简单/对数) 11 每笔收益分布(总体/分资产) 12 持有期分布
      13 分资产收益/风险贡献与比 14 周期收益表(分页 12/页) 15 当前持仓表
      16 次日计划(买卖清单与目标权重 delta) 17 叙事研报解读。
- [ ] 布局从总览到细节自上而下，相关图表聚类，1–3 列/行。

API 合同测试：

- [ ] 新增/改动路由按 `momentum-api-contract` 技能补 ASGI 级合同测试。

## 交付证据（写入回复/PR）

公式核对项、检查的 scope、跑过的测试、基准/执行时点语义说明、已知非可加语义。
