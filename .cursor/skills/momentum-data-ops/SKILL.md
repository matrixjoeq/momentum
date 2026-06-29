---
name: momentum-data-ops
description: >
  Momentum 工程的数据抓取、校验与同步执行规范。在拉取/入库 ETF、场外基金、期货、
  宏观数据，处理价格复权、公司行动、期货连续合约合成，或做 EOD 同步时使用。
  触发词："抓数据/拉数据"、"数据同步"、"akshare"、"宏观数据"、"复权 none/hfq"、
  "公司行动/除权除息"、"期货 889/连续合约"、"EOD 同步"、"data ingestion"。
  本技能是 AGENTS.md 价格复权/期货连续等条款的可执行入口，不取代条款本身。
---

# Momentum 数据运维技能

权威定义以 `AGENTS.md`（价格复权规则、期货连续 889 规则、correlation 规则）与
`SOUL.md` 为准，连续合约细节见 `docs/futures-continuous-rules.md` /
`docs/futures-continuous-spec.md`。冲突时以 `SOUL.md` / `AGENTS.md` 为准。

## 何时使用

抓取/入库行情或宏观数据、处理复权与公司行动、合成期货连续序列、跑 EOD 同步。

## 关键文件

- 数据源：`src/etf_momentum/data/`（`akshare_fetcher.py`、
  `akshare_eastmoney_fetcher.py`、`multi_source_fetcher.py`、
  `off_fund_ingestion.py`、`futures_ingestion.py`、
  `futures_contract_ingestion.py`、`cboe_fetcher.py`、`yahoo_fetcher.py`、
  `fred_fetcher.py`、`sina_fetcher.py`、`stooq_fetcher.py`）。
- 校验：`src/etf_momentum/validation/`（价格校验与策略推断）。
- 同步：`src/etf_momentum/scheduler/market_sync.py`、
  `src/etf_momentum/services/`，管理 API `/api/admin/sync/fixed-pool`。
- 宏观脚本：`scripts/update_macro.py`。

## 执行步骤

1. 规划：确认目标资产类、数据源优先级与回退源、时间窗口。
2. 拆解：抓取 → 校验 → 入库 → 复权处理 → （期货）连续合成 → 同步。
3. 执行：复用既有 fetcher，缺失能力再扩展；外部调用失败用带上下文消息包裹。
4. 审查：跑下方检查清单。
5. 复盘：把新数据源/字段语义补进 `AGENTS.md` 或 `docs/`。

## 检查清单

价格复权（与 NAV 计算一致）：

- [ ] 策略 NAV 优先 `none`（原始/不复权）日收益。
- [ ] 公司行动（分红/拆分/合股）导致原始价跳变的当日点回退 `hfq`。
- [ ] 基准序列与 NAV 恒用 `hfq` close；各引擎回退语义一致并有回归测试。

期货连续合约：

- [ ] trend/correlation 用合成后复权连续序列 `{root}889`；缺失则明确报错
      （不回退 `{root}88`/`888`/主力 `*0` none）。
- [ ] correlation 用 `{root}889` hfq close 的日对数收益 Pearson，标签为池代码。

数据质量与稳健：

- [ ] 多源回退（akshare/腾讯/新浪等）按优先级；失败有上下文错误。
- [ ] 校验缺口/异常跳变；不静默吞错（不 `except Exception`）。
- [ ] 需要密钥的源（如 FRED `MOMENTUM_FRED_API_KEY`）走环境变量，不硬编码。

同步：

- [ ] 自动同步默认关闭（`MOMENTUM_AUTO_SYNC_ENABLED=false`），按需经管理 API。
- [ ] 涉及路由按 `momentum-api-contract` 技能补合同测试。

## 交付证据

数据源与回退链、复权/连续合成处理、校验结果、跑过的测试。
