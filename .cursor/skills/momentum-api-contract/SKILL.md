---
name: momentum-api-contract
description: >
  Momentum 工程的 API 合同测试执行规范。在新增/修改任何 /api/... 路由或请求/响应
  schema 时使用，确保每个 HTTP 方法+路径都有 ASGI 级合同测试覆盖真实请求边界。
  触发词："新增 API/路由"、"改接口"、"合同测试/contract test"、"422/400"、
  "FastAPI 路由"、"Pydantic 校验"、"routes.py"、"test client"。
  本技能是 AGENTS.md "API contract test rule" 的可执行入口，不取代条款本身。
---

# Momentum API 合同测试技能

权威定义以 `AGENTS.md`（§ Testing：API contract tests；§ Maintenance：API
contract test rule）与 `SOUL.md` 为准。冲突时以 `SOUL.md` / `AGENTS.md` 为准。

## 何时使用

新增或修改任何 `/api/...` 路由、请求体/响应体 schema、Pydantic `Field` 约束。

## 关键文件

- 路由：`src/etf_momentum/api/routes.py`、`src/etf_momentum/api/live_trading.py`。
- 应用：`src/etf_momentum/app.py`（ASGI 入口，静态挂载 `/static`）。
- 现有合同测试：`tests/test_app_root.py` 及各 `tests/test_*` 文件。

## 核心规则

- 每个对外 HTTP operation（每个不同的方法+路径）至少一个**自动化合同测试**，经
  ASGI 栈（Starlette/FastAPI TestClient）以真实客户端方式发 JSON body + headers。
- 测试必须覆盖请求/响应边界：
  - 合法载荷 → 预期成功状态（通常 200）。
  - UI 可能发出的"客户端 bug 形态"非法载荷 → 预期 422/400（例如被缩放的百分比
    字段、可选字段送/不送、违反 Pydantic `Field` 约束的值）。
- 纯进程内、绕过 HTTP、只用已校验 Python 模型的调用**不**满足本规则。
- 新路由或 schema 改动必须**同批**新增/更新对应合同测试。
- 故意内部专用/弃用/豁免的路由，在路由 docstring 或 `AGENTS.md` 写明例外与理由。

## 执行步骤

1. 拆解：列出本次新增/改动的每个方法+路径。
2. 执行：为每条写至少一个成功用例 + 至少一个 UI 形态非法负例。
3. 用项目 venv 跑：`.venv/bin/python3 -m pytest tests/test_xxx.py -q`。
4. 审查：确认覆盖率无遗漏路由；曾导致 422/400 回归的形态必须有负例。
5. 复盘：豁免项写明理由。

## 检查清单

- [ ] 本次每个方法+路径都有 ASGI 级合同测试。
- [ ] 每条至少 1 成功 + 1 UI 形态非法负例（百分比缩放、可选字段、Field 约束）。
- [ ] 测试默认不依赖外网（除非显式标记 integration）。
- [ ] 与路由/schema 改动同批提交。
- [ ] 豁免路由有 docstring/AGENTS.md 说明。

## 交付证据

新增/更新的合同测试文件与用例、跑过的 pytest 命令与结果、豁免说明（如有）。
