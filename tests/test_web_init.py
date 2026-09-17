from __future__ import annotations

from pathlib import Path


def test_web_package_importable() -> None:
    import etf_momentum.web  # noqa: F401


def test_off_fund_pair_comparison_supports_60_slots() -> None:
    web_dir = Path(__file__).parents[1] / "src" / "etf_momentum" / "web"
    off_fund_html = (web_dir / "off_fund_research.html").read_text(encoding="utf-8")

    assert "const OFF_FUND_PAIR_MAX_SLOT_COUNT = 60;" in off_fund_html


def test_dca_base_amount_frontends_allow_zero_and_reject_negative() -> None:
    web_dir = Path(__file__).parents[1] / "src" / "etf_momentum" / "web"
    research_html = (web_dir / "research.html").read_text(encoding="utf-8")
    off_fund_html = (web_dir / "off_fund_research.html").read_text(encoding="utf-8")

    validation = "Number.isFinite(dca_base_amount) && dca_base_amount >= 0"
    off_validation = "Number.isFinite(dcaBaseAmount) && dcaBaseAmount >= 0"
    state_validation = "Number.isFinite(dcaBaseRaw) && dcaBaseRaw >= 0"
    assert validation in research_html
    assert off_validation in off_fund_html
    assert state_validation in off_fund_html
    assert "定投底仓金额必须是大于等于0的数字。" in research_html
    assert "定投底仓金额必须是大于等于0的数字。" in off_fund_html
    no_contribution_message = "底仓为0时，每期金额必须大于0且频率不能为无追加。"
    assert no_contribution_message in research_html
    assert no_contribution_message in off_fund_html


def test_research_garch_six_model_compare_is_opt_in() -> None:
    web_dir = Path(__file__).parents[1] / "src" / "etf_momentum" / "web"
    research_html = (web_dir / "research.html").read_text(encoding="utf-8")
    assert 'id="distGarchModelCompareOn"' in research_html
    assert '<input id="distGarchModelCompareOn" type="checkbox" />' in research_html
    assert "include_model_comparison: includeCompare" in research_html
    assert "计算并比对六个波动率模型" in research_html


def test_week_extrema_probability_heatmap_uses_dark_color_for_high_values() -> None:
    web_dir = Path(__file__).parents[1] / "src" / "etf_momentum" / "web"
    research_html = (web_dir / "research.html").read_text(encoding="utf-8")
    heatmap_start = research_html.index("heatEl.id,\n            [")
    heatmap_end = research_html.index(
        "{ responsive: true, displayModeBar: false }", heatmap_start
    )
    heatmap = research_html[heatmap_start:heatmap_end]

    assert '[0.0, "#fff7ec"]' in heatmap
    assert '[1.0, "#7f0000"]' in heatmap
    assert "reversescale: false" in heatmap
