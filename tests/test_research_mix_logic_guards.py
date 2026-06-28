from pathlib import Path


def test_research_mix_custom_weight_default_is_one() -> None:
    html = Path("src/etf_momentum/web/research.html").read_text(encoding="utf-8")
    assert "Number((_readCustomMixWeightMap() || {})[x.id] || 1)" in html
    assert "Number((_readCustomMixWeightMap() || {})[x.id] || 0)" not in html


def test_research_mix_weighted_leaf_total_return_uses_contrib_guard() -> None:
    html = Path("src/etf_momentum/web/research.html").read_text(encoding="utf-8")
    assert "let hasReturnContrib = false;" in html
    assert "hasReturnContrib = true;" in html
    assert "total_return: hasReturnContrib" in html
    assert "total_return: Number.isFinite(totalReturnFromContrib)" not in html
