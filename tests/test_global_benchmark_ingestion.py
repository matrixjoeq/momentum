from __future__ import annotations

from etf_momentum.data.global_benchmark_ingestion import (
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
