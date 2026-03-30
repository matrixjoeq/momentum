from __future__ import annotations

import datetime as dt

from etf_momentum.analysis import calendar_effect as cal_effect
from etf_momentum.analysis.rotation import RotationAnalysisInputs


def test_rotation_calendar_effect_uses_strategy_only_backtest(monkeypatch):
    called_include_benchmarks: list[bool] = []

    def _fake_compute_rotation_backtest(_db, _inp, *, include_benchmarks: bool = True):
        called_include_benchmarks.append(bool(include_benchmarks))
        return {
            "metrics": {
                "strategy": {
                    "annualized_return": 0.12,
                    "sharpe_ratio": 1.1,
                    "calmar_ratio": 0.8,
                    "sortino_ratio": 1.4,
                    "ulcer_index": 4.2,
                    "ulcer_performance_index": 1.8,
                    "information_ratio": float("nan"),
                    "max_drawdown": -0.15,
                    "avg_daily_turnover": 0.05,
                }
            },
            "nav": {
                "dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "series": {"ROTATION": [1.0, 1.01, 1.015]},
            },
        }

    monkeypatch.setattr(cal_effect, "compute_rotation_backtest", _fake_compute_rotation_backtest)

    base = RotationAnalysisInputs(
        codes=["510300", "511010"],
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 2, 15),
        rebalance="weekly",
    )
    out = cal_effect.compute_rotation_calendar_effect(
        db=None,  # type: ignore[arg-type]
        base=base,
        anchors=[1],
        exec_prices=["open"],
    )

    assert called_include_benchmarks == [False]
    row = (out.get("grid") or [])[0]
    assert row.get("ok") is True
    assert "metrics" in row
