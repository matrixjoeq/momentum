from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest


HTML_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "etf_momentum"
    / "web"
    / "off_fund_research.html"
)


def _javascript_function(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated JavaScript function: {name}")


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values)


def test_risk_adjusted_pair_momentum_formula_and_warmup() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the embedded chart calculation")

    n = 400
    base_returns = [0.0002 + 0.001 * math.sin(i / 7) for i in range(1, n)]
    peer_returns = [0.0004 + 0.0015 * math.cos(i / 11) for i in range(1, n)]
    peer_returns[89] = 0.08  # Verify the requested no-clipping behavior.
    base_close = [100.0]
    peer_close = [80.0]
    for base_return, peer_return in zip(base_returns, peer_returns):
        base_close.append(base_close[-1] * math.exp(base_return))
        peer_close.append(peer_close[-1] * math.exp(peer_return))

    base_vol = {t: _sample_std(base_returns[t - 60 : t]) for t in range(60, n)}
    peer_vol = {t: _sample_std(peer_returns[t - 60 : t]) for t in range(60, n)}
    relative = {
        t: peer_returns[t - 1] / peer_vol[t] - base_returns[t - 1] / base_vol[t]
        for t in range(60, n)
    }
    momentum = {t: sum(relative[j] for j in range(t - 39, t + 1)) for t in range(99, n)}
    expected_raw_99 = momentum[99]
    expected_z_340 = (
        momentum[340] - statistics.mean(momentum[t] for t in range(99, 341))
    ) / _sample_std([momentum[t] for t in range(99, 341)])
    expected_ma_340 = statistics.mean(momentum[t] for t in range(99, 341))

    html = HTML_PATH.read_text(encoding="utf-8")
    function_names = [
        "_rollingMean",
        "_rollingStd",
        "_rollingSampleStd",
        "_rollingSum",
        "_rollingZScore",
        "_normalizedLogReturnsByDate",
        "_alignOffFundPairSeries",
        "_computeRsi14",
        "_computeOffFundPairMetrics",
    ]
    functions = "\n".join(_javascript_function(html, name) for name in function_names)
    aligned = {
        "dates": [f"day-{i}" for i in range(n)],
        "baseClose": base_close,
        "peerClose": peer_close,
    }
    script = f"""
const OFF_FUND_PAIR_VOLATILITY_WINDOW = 60;
const OFF_FUND_PAIR_MOMENTUM_WINDOW = 40;
const OFF_FUND_PAIR_MOMENTUM_Z_WINDOW = 242;
const OFF_FUND_PAIR_MIN_DAILY_VOLATILITY = 1e-8;
const OFF_FUND_PAIR_MOMENTUM_Z_REQUIRED_SAMPLES = 341;
{functions}
const aligned = {json.dumps(aligned)};
const toSeries = (dates, closes) => ({{
  dates,
  closeByDate: new Map(dates.map((date, i) => [date, closes[i]])),
}});
const productionAligned = _alignOffFundPairSeries(
  toSeries(aligned.dates, aligned.baseClose),
  toSeries(aligned.dates, aligned.peerClose),
);
const metrics = _computeOffFundPairMetrics(productionAligned);
const shortAligned = {{
  dates: productionAligned.dates.slice(0, 340),
  baseClose: productionAligned.baseClose.slice(0, 340),
  peerClose: productionAligned.peerClose.slice(0, 340),
  baseNormalizedLogReturn: productionAligned.baseNormalizedLogReturn.slice(0, 340),
  peerNormalizedLogReturn: productionAligned.peerNormalizedLogReturn.slice(0, 340),
}};
const shortMetrics = _computeOffFundPairMetrics(shortAligned);
const scaled = _computeOffFundPairMetrics({{
  dates: aligned.dates,
  baseClose: aligned.baseClose.map((x) => x * 13),
  peerClose: aligned.peerClose.map((x) => x * 0.07),
}});
const swapped = _computeOffFundPairMetrics({{
  dates: aligned.dates,
  baseClose: aligned.peerClose,
  peerClose: aligned.baseClose,
}});
const firstFinite = (values) => values.findIndex(
  (value) => value !== null && Number.isFinite(Number(value)),
);
console.log(JSON.stringify({{
  rawFirst: firstFinite(metrics.riskAdjustedMomentum40),
  zFirst: firstFinite(metrics.riskAdjustedMomentum40Z242),
  maFirst: firstFinite(metrics.riskAdjustedMomentum40Ma242),
  shortZFirst: firstFinite(shortMetrics.riskAdjustedMomentum40Z242),
  raw99: metrics.riskAdjustedMomentum40[99],
  z340: metrics.riskAdjustedMomentum40Z242[340],
  ma340: metrics.riskAdjustedMomentum40Ma242[340],
  scaledRaw99: scaled.riskAdjustedMomentum40[99],
  scaledZ340: scaled.riskAdjustedMomentum40Z242[340],
  swappedRaw99: swapped.riskAdjustedMomentum40[99],
  swappedZ340: swapped.riskAdjustedMomentum40Z242[340],
}}));
"""
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["rawFirst"] == 99
    assert result["zFirst"] == 340
    assert result["maFirst"] == 340
    assert result["shortZFirst"] == -1
    assert result["raw99"] == pytest.approx(expected_raw_99, abs=1e-10)
    assert result["z340"] == pytest.approx(expected_z_340, abs=1e-10)
    assert result["ma340"] == pytest.approx(expected_ma_340, abs=1e-10)
    assert result["scaledRaw99"] == pytest.approx(result["raw99"], abs=1e-10)
    assert result["scaledZ340"] == pytest.approx(result["z340"], abs=1e-10)
    assert result["swappedRaw99"] == pytest.approx(-result["raw99"], abs=1e-10)
    assert result["swappedZ340"] == pytest.approx(-result["z340"], abs=1e-10)


def test_risk_adjusted_momentum_is_below_simple_return_spread() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'name: "40日收益差(对比-基准)"' in html
    assert 'name: "风险调整40日相对动量(60/40)"' in html
    assert html.index('name: "40日收益差(对比-基准)"') < html.index(
        'name: "风险调整40日相对动量(60/40)"'
    )
    assert 'title: "净值"' in html
    assert 'title: "比值"' in html
    # Ratio panel uses log scale (same convention as research.html ratio charts).
    ratio_title_idx = html.index('title: "比值"')
    ratio_axis_block = html[ratio_title_idx : ratio_title_idx + 120]
    assert 'type: "log"' in ratio_axis_block
    assert 'title: "40日收益差"' in html
    assert 'title: "相对动量"' in html
    assert 'title: "RSI"' in html
    assert 'name: "相对动量MA242"' in html
    assert "Z242: %{customdata}" in html
    assert "OFF_FUND_PAIR_MOMENTUM_Z_WINDOW = 242" in html
    assert "OFF_FUND_PAIR_MOMENTUM_Z_REQUIRED_SAMPLES" in html
    assert "不做截尾" in html
