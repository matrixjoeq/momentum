from __future__ import annotations

from etf_momentum.validation.policy_infer import infer_policy_name


def test_infer_star_related() -> None:
    r = infer_policy_name(code="588000", name="科创50ETF")
    assert r.policy_name == "star_related_30"


def test_infer_chinext_related() -> None:
    r = infer_policy_name(code="159915", name="创业板ETF")
    assert r.policy_name == "chinext_related_20"


def test_infer_bond() -> None:
    r = infer_policy_name(code="511010", name="国债ETF")
    assert r.policy_name == "bond_10_strict"


def test_infer_qdii_commod() -> None:
    r = infer_policy_name(code="513100", name="纳指ETF(QDII)")
    assert r.policy_name == "qdii_commod_fx"


def test_infer_default() -> None:
    r = infer_policy_name(code="510300", name="沪深300ETF")
    assert r.policy_name == "cn_stock_etf_10"

