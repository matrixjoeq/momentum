# 后期任务：`bt_trend` 与 `trend` 去重、真源统一

**目标**：`bt_trend` 只负责 **TA-Lib 指标** 与 **`backtesting` 回测**；其余策略含义、风控、统计、持仓与 `trend` 完全复用，避免同一功能两套实现。

**真源**：`src/etf_momentum/analysis/trend.py`（`trend` 为默认实现来源）。

**依赖关系**：当前 `trend.py` **不** import `bt_trend`，可从 `bt_trend` 安全导入 `trend`。合并时注意分批、跑全量趋势相关测试。

---

## 已完成（单一真源 + 再导出）

以下已在 `bt_trend` 从 `trend` 导入（与 `trend` 内为同一函数对象），不再维护副本：

- `_apply_monthly_risk_budget_gate`
- `_apply_atr_stop`
- `_stop_fill_return`
- `_apply_intraday_stop_execution_single`
- `_apply_intraday_stop_execution_portfolio`

---

## P0 — 高风险重复（优先收敛到 `trend`）

| 区域          | `bt_trend` 中仍为独立副本的符号（示例）                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 月度门控配套  | `_month_key`, `_position_risk_from_stop_params`, `_risk_budget_dynamic_weights`                                                                                                                                                                                                                                                                                                                                                     |
| 止盈叠加      | `_normalize_r_take_profit_tiers`, `_apply_r_multiple_take_profit`, `_apply_bias_v_take_profit`, `_extract_atr_plan_stops_from_trace`, `_latest_entry_exec_price_with_slippage`                                                                                                                                                                                                                                                      |
| 持仓/信号构造 | `_pos_from_donchian`, `_pos_from_tsmom`, `_pos_from_band`, `_pos_from_random_entry_hold`, `_stable_code_seed`（Donchian 辅助：`bt_trend` 另有 `_donchian_prev_high` / `_donchian_prev_low`）                                                                                                                                                                                                                                        |
| Impulse / ER  | `_compute_impulse_state`, `_efficiency_ratio`, `_apply_er_entry_filter`, `_apply_er_exit_filter`, `_apply_impulse_entry_filter`                                                                                                                                                                                                                                                                                                     |
| 组合与权重    | `_reduce_active_codes_by_group`, `_trade_returns_from_weight_series`, `_trade_returns_from_weight_df`                                                                                                                                                                                                                                                                                                                               |
| 统计与分桶    | `_dist_stats`, `_trade_stats_from_returns`, `_rolling_pack`, `_round_half_up`, `_bucketize_*`, `_series_index_to_date_str`, `_build_entry_signal_date_map`, `_attach_entry_condition_bins_to_trades`, `_normal_two_sided_p_from_z`, `_two_proportion_z_test`, `_welch_t_test_normal_approx`, `_bh_qvalues`, `_stratified_permutation_pvalue`, `_sorted_condition_buckets`, `_stable_seed_from_text`, `_build_entry_condition_stats` |

---

## P1 — Talib 留在 BT，fallback 与 `trend` 对齐

- `_ema`, `_moving_average`, `_macd_core`, `_atr_from_hlc`：保留 Talib 分支；**pandas / Wilder fallback 应唯一化**（从 `trend` 导入或抽公共子模块），与 `trend` 的 `_ema` / `_atr_from_hlc` 等一致。
- `_kama_fallback` vs `trend._kama`：统一兜底实现。

---

## P2 — 同名不同义 / 接口差异（合并时需封装）

- **`_rolling_linreg_slope`**：`trend` 为 `ndarray` 版；`bt_trend` 为 `Series` + `_rolling_linreg_slope_raw`。建议改名 BT 侧包装函数，或 BT 仅调用 `trend` 核心。
- **`_period_returns`**：`trend` 返回 `pd.DataFrame`；`bt_trend` 返回 `list[dict]` 且使用 `_tsmom_rocp`。合并前需统一业务口径或保留薄适配层。

---

## P3 — 明确保留在 `bt_trend`（非去重范围）

- Talib：`_talib_enabled`, `_talib_unary_series`, `_prefer_legacy_on_diff`, `_*_fallback`（在统一 fallback 后可能瘦身）。
- Backtesting：` _build_bt_frame`, `_run_single_backtesting`, `_build_signal_position`, `compute_trend_backtest_bt`, `compute_trend_portfolio_backtest_bt`。
- BT 专用适配：`_validate_bt_single_inputs`, `_build_meta_params`, `_forward_simple_return`, `_ratio_simple_return`, `_clone_like_input`, `_as_nav`, `_metrics_from_ret` 等。

---

## 验收建议

- 每批去重后：`pytest` 覆盖 `tests/test_analysis_trend.py`、BT 相关测试、组合/期货趋势相关测试。
- 对仍从 `bt_trend` 再导出给外部的符号，在变更说明中注明「实现位于 `trend`」。

---

_本文档由代码对比检查结论整理，用于跟踪后续重构；更新进度时请在本文件勾选或追加 PR 链接。_
