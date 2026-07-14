from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSourceSpec:
    provider: str
    symbol: str


@dataclass(frozen=True)
class GlobalBenchmarkSeriesSpec:
    series_kind: str  # price | total_return
    code_format: str
    candidates: tuple[SeriesSourceSpec, ...]
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class DefaultGlobalBenchmarkSpec:
    code: str
    name: str
    price: GlobalBenchmarkSeriesSpec
    total_return: GlobalBenchmarkSeriesSpec


def _cn_exchange_suffix(code: str) -> str:
    c = str(code or "").strip()
    return ".SS" if c.startswith(("5", "6", "9")) else ".SZ"


def _cn_price_candidates(code: str) -> tuple[SeriesSourceSpec, ...]:
    c = str(code or "").strip()
    return (
        SeriesSourceSpec(provider="tencent", symbol=c),
        SeriesSourceSpec(provider="yahoo", symbol=f"{c}{_cn_exchange_suffix(c)}"),
        SeriesSourceSpec(provider="stooq", symbol=f"{c}{_cn_exchange_suffix(c)}"),
    )


_CN_TR_ALIAS: dict[str, tuple[str, ...]] = {
    # Commonly used CSI total-return aliases in public terminals/vendors.
    # Keep as candidates; acceptance API will verify and solidify usable one.
    "000300": ("H00300",),
    "000905": ("H00905",),
    "000852": ("H00852",),
    "932000": ("H32000",),
}


def _cn_total_return_candidates(
    code: str, *, name: str
) -> tuple[SeriesSourceSpec, ...]:
    c = str(code or "").strip()
    n = str(name or "").strip().upper()
    # Existing index names containing trailing R are commonly total-return variants.
    if n.endswith("R"):
        return (
            SeriesSourceSpec(provider="tencent", symbol=c),
            SeriesSourceSpec(provider="yahoo", symbol=f"{c}{_cn_exchange_suffix(c)}"),
            SeriesSourceSpec(provider="stooq", symbol=f"{c}{_cn_exchange_suffix(c)}"),
        )
    aliases = _CN_TR_ALIAS.get(c, tuple())
    out: list[SeriesSourceSpec] = []
    for a in aliases:
        out.append(SeriesSourceSpec(provider="tencent", symbol=a))
        out.append(
            SeriesSourceSpec(provider="yahoo", symbol=f"{a}{_cn_exchange_suffix(c)}")
        )
    return tuple(out)


def _global_price_candidates(code: str) -> tuple[SeriesSourceSpec, ...]:
    c = str(code or "").strip()
    return (
        SeriesSourceSpec(provider="yahoo", symbol=c),
        SeriesSourceSpec(provider="stooq", symbol=c),
    )


_GLOBAL_TR_ALIAS: dict[str, tuple[str, ...]] = {
    "^GSPC": ("^SP500TR",),
    "^NDX": ("^NDXT",),
    "^DJI": ("^DJITR",),
    "^RUT": ("^RUTTR",),
    "^FTSE": ("^FTSETR",),
    "^FCHI": ("^FCHITR",),
    "^GDAXI": ("^DAXTR",),
    "^N225": ("^N225TR",),
    "^HSI": ("^HSITR",),
    "^HSTECH": ("^HSTECHTR",),
    "^KS11": ("^KS11TR",),
    "^STOXX50E": ("^SX5ETR",),
}


def _global_total_return_candidates(code: str) -> tuple[SeriesSourceSpec, ...]:
    c = str(code or "").strip()
    aliases = _GLOBAL_TR_ALIAS.get(c, tuple())
    out: list[SeriesSourceSpec] = []
    for a in aliases:
        out.append(SeriesSourceSpec(provider="yahoo", symbol=a))
        out.append(SeriesSourceSpec(provider="stooq", symbol=a))
    return tuple(out)


def _build_cn_spec(code: str, name: str, start_date: str) -> DefaultGlobalBenchmarkSpec:
    price = GlobalBenchmarkSeriesSpec(
        series_kind="price",
        code_format="cn_6",
        candidates=_cn_price_candidates(code),
        start_date=start_date,
    )
    tr = GlobalBenchmarkSeriesSpec(
        series_kind="total_return",
        code_format="cn_6",
        candidates=_cn_total_return_candidates(code, name=name),
        start_date=start_date,
    )
    return DefaultGlobalBenchmarkSpec(
        code=code, name=name, price=price, total_return=tr
    )


def _build_global_spec(
    code: str, name: str, start_date: str
) -> DefaultGlobalBenchmarkSpec:
    price = GlobalBenchmarkSeriesSpec(
        series_kind="price",
        code_format="yahoo",
        candidates=_global_price_candidates(code),
        start_date=start_date,
    )
    tr = GlobalBenchmarkSeriesSpec(
        series_kind="total_return",
        code_format="yahoo",
        candidates=_global_total_return_candidates(code),
        start_date=start_date,
    )
    return DefaultGlobalBenchmarkSpec(
        code=code, name=name, price=price, total_return=tr
    )


_CN_BENCHMARKS: tuple[tuple[str, str, str], ...] = (
    ("000300", "沪深300", "20041231"),
    ("000905", "中证500", "20041231"),
    ("000852", "中证1000", "20041231"),
    ("932000", "中证2000", "20131231"),
    ("399673", "创业板50", "20100531"),
    ("000688", "科创50", "20191231"),
    ("000922", "中证红利", "20041231"),
    ("932365", "中证现金流", "20131231"),
    ("480080", "国证成长100R", "20121231"),
    ("480081", "国证价值100R", "20121231"),
    ("931589", "300成长创新R", "20041231"),
    ("931586", "300价值稳健R", "20041231"),
    ("931591", "1000成长创新R", "20041231"),
    ("931588", "1000价值稳健R", "20041231"),
    ("931775", "中证全指房地产R", "20041231"),
    ("932077", "中证全指能源行业R", "20041231"),
    ("932078", "中证全指材料行业R", "20041231"),
    ("932079", "中证全指工业行业R", "20041231"),
    ("932080", "中证全指可选行业R", "20041231"),
    ("932081", "中证全指消费行业R", "20041231"),
    ("932082", "中证全指医药行业R", "20041231"),
    ("932083", "中证全指金融行业R", "20041231"),
    ("932084", "中证全指信息行业R", "20041231"),
    ("932085", "中证全指通信行业R", "20041231"),
    ("932086", "中证全指公用行业R", "20041231"),
)


_GLOBAL_BENCHMARKS: tuple[tuple[str, str, str], ...] = (
    ("^HSI", "恒生指数", "19640731"),
    ("^HSTECH", "恒生科技指数", "20141231"),
    ("^DJI", "道琼斯工业指数", "18991231"),
    ("^GSPC", "标普500", "19280103"),
    ("^NDX", "纳斯达克100", "19850201"),
    ("^RUT", "罗素2000", "19781231"),
    ("^FTSE", "富时100", "19840103"),
    ("^FCHI", "CAC40", "19871231"),
    ("^GDAXI", "DAX30", "19871231"),
    ("^STOXX50E", "Euro Stoxx 50", "19911231"),
    ("^N225", "日经225", "19490516"),
    ("^KS11", "韩国综合指数", "19800104"),
)


DEFAULT_GLOBAL_BENCHMARK_UNIVERSE: tuple[DefaultGlobalBenchmarkSpec, ...] = (
    *(_build_cn_spec(code, name, start) for code, name, start in _CN_BENCHMARKS),
    *(
        _build_global_spec(code, name, start)
        for code, name, start in _GLOBAL_BENCHMARKS
    ),
)


def default_benchmark_spec_by_code(code: str) -> DefaultGlobalBenchmarkSpec | None:
    key = str(code or "").strip()
    for x in DEFAULT_GLOBAL_BENCHMARK_UNIVERSE:
        if x.code == key:
            return x
    return None


def default_series_spec_by_code(
    code: str, series_kind: str
) -> GlobalBenchmarkSeriesSpec | None:
    spec = default_benchmark_spec_by_code(code)
    if spec is None:
        return None
    kind = str(series_kind or "").strip().lower()
    if kind == "price":
        return spec.price
    if kind == "total_return":
        return spec.total_return
    return None
