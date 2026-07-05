from __future__ import annotations

from etf_momentum.data.global_benchmark_defaults import (
    DEFAULT_GLOBAL_BENCHMARK_UNIVERSE,
)


def test_default_global_benchmark_universe_has_unique_codes() -> None:
    codes = [x.code for x in DEFAULT_GLOBAL_BENCHMARK_UNIVERSE]
    assert len(codes) == len(set(codes))
    assert "000300" in codes
    assert "^GSPC" in codes


def test_default_global_benchmark_universe_fields_are_present() -> None:
    for x in DEFAULT_GLOBAL_BENCHMARK_UNIVERSE:
        assert str(x.code).strip()
        assert str(x.name).strip()
        assert x.price.series_kind == "price"
        assert x.total_return.series_kind == "total_return"
        assert str(x.price.code_format).strip()
        assert str(x.total_return.code_format).strip()
