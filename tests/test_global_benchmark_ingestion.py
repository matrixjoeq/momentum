from __future__ import annotations

from etf_momentum.data.global_benchmark_ingestion import (
    _tencent_symbol_candidates,
    _to_yahoo_symbol,
    detect_code_format,
    resolve_provider_candidates,
)


def test_detect_code_format_auto_basic_cases() -> None:
    assert detect_code_format("000300", None) == "cn_6"
    assert detect_code_format("^GSPC", None) == "yahoo"
    assert detect_code_format("HSI", None) == "hk_index"
    assert detect_code_format("MSCIWORLD", None) == "msci"
    assert detect_code_format("abc", None) == "symbol"


def test_detect_code_format_respects_non_auto_hint() -> None:
    assert detect_code_format("000300", "yahoo") == "yahoo"
    assert detect_code_format("^GSPC", "msci") == "msci"


def test_resolve_provider_candidates_auto_vs_hint() -> None:
    assert (
        resolve_provider_candidates(code_format="cn_6", provider_hint="auto")[0]
        == "tencent"
    )
    assert (
        resolve_provider_candidates(code_format="msci", provider_hint="auto")[0]
        == "sina_global"
    )
    assert resolve_provider_candidates(code_format="yahoo", provider_hint="stooq") == [
        "stooq"
    ]


def test_to_yahoo_symbol_for_hk_and_cn_formats() -> None:
    assert _to_yahoo_symbol("HSI", code_format="hk_index") == "^HSI"
    assert _to_yahoo_symbol("^HSI", code_format="hk_index") == "^HSI"
    assert _to_yahoo_symbol("000300", code_format="cn_6") == "000300.SZ"
    assert _to_yahoo_symbol("600519", code_format="cn_6") == "600519.SS"


def test_tencent_symbol_candidates_include_sh_sz_fallback() -> None:
    assert _tencent_symbol_candidates("000300") == ["000300", "sh000300", "sz000300"]
    assert _tencent_symbol_candidates("sh000300") == ["sh000300", "000300", "sz000300"]
