from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultGlobalBenchmarkSpec:
    code: str
    name: str
    code_format: str
    provider_hint: str
    start_date: str | None = None
    end_date: str | None = None


# Phase-3 default universe:
# - Keep a compact, representative set across CN/HK/US/EU/JP.
# - Prefer symbols with long public history and stable provider support.
DEFAULT_GLOBAL_BENCHMARK_UNIVERSE: tuple[DefaultGlobalBenchmarkSpec, ...] = (
    DefaultGlobalBenchmarkSpec(
        code="000300",
        name="沪深300",
        code_format="cn_6",
        provider_hint="auto",
        start_date="20050101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="000905",
        name="中证500",
        code_format="cn_6",
        provider_hint="auto",
        start_date="20070101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="000852",
        name="中证1000",
        code_format="cn_6",
        provider_hint="auto",
        start_date="20140101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^HSI",
        name="恒生指数",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^GSPC",
        name="标普500",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^NDX",
        name="纳斯达克100",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^DJI",
        name="道琼斯工业指数",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^RUT",
        name="罗素2000",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^STOXX50E",
        name="Euro Stoxx 50",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^FTSE",
        name="富时100",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
    DefaultGlobalBenchmarkSpec(
        code="^N225",
        name="日经225",
        code_format="yahoo",
        provider_hint="auto",
        start_date="20000101",
    ),
)
