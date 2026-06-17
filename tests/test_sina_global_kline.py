from etf_momentum.data.sina_fetcher import (
    _parse_sina_global_kline,
    normalize_sina_global_symbol,
)

# Trimmed sample mirroring Sina's GlobalFuturesService JSONP payload shape.
_SAMPLE = (
    "/*<script>location.href='//sina.com';</script>*/\n"
    'var _=XAU=([{"date":"2006-06-19","open":"580.350","high":"580.650",'
    '"low":"564.380","close":"565.250","volume":"0","position":"0","s":"0.000"},'
    '{"date":"2006-06-20","open":"565.000","high":"570.000","low":"560.000",'
    '"close":"568.100","volume":"0","position":"0","s":"0.000"}]);'
)


def test_normalize_sina_global_symbol_maps_spot_tickers():
    assert normalize_sina_global_symbol("XAUUSD") == "XAU"
    assert normalize_sina_global_symbol("xagusd") == "XAG"
    # Already-Sina symbols and unknowns pass through (upper-cased).
    assert normalize_sina_global_symbol("GC") == "GC"
    assert normalize_sina_global_symbol(" cl ") == "CL"


def test_parse_sina_global_kline_extracts_records():
    records = _parse_sina_global_kline(_SAMPLE)
    assert len(records) == 2
    assert records[0]["date"] == "2006-06-19"
    assert records[0]["close"] == "565.250"
    assert records[-1]["close"] == "568.100"


def test_parse_sina_global_kline_handles_garbage():
    assert _parse_sina_global_kline("") == []
    assert _parse_sina_global_kline("no array here") == []
    assert _parse_sina_global_kline("var _=XAU=([broken);") == []
