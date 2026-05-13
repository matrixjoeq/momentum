from __future__ import annotations

import pandas as pd

from etf_momentum.data.futures_synthesize import (
    _collect_auto_corrections,
    _attach_switch_suffix_column,
    _build_adjusted_continuous,
    _build_hold_table,
    _contract_yymm_suffix,
    _replay_dominant,
    _sanitize_settle_column,
)


def test_build_hold_table_matches_replay_script_shape() -> None:
    dts = pd.to_datetime(["2024-01-02", "2024-01-03"])
    contract_data = {
        "RB2405": pd.DataFrame(
            {
                "date": dts,
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.1, 1.2],
                "volume": [10, 11],
                "hold": [100, 80],
                "settle": [1.05, 1.15],
                "amount": [1000, 1200],
            }
        ),
        "RB2409": pd.DataFrame(
            {
                "date": dts,
                "open": [2.0, 2.1],
                "high": [2.2, 2.3],
                "low": [1.9, 2.0],
                "close": [2.1, 2.2],
                "volume": [20, 21],
                "hold": [90, 120],
                "settle": [2.05, 2.15],
                "amount": [2000, 2200],
            }
        ),
    }

    tbl = _build_hold_table(contract_data)
    assert list(tbl.columns) == ["RB2405", "RB2409"]
    assert list(tbl.index) == list(dts)
    assert float(tbl.loc[dts[0], "RB2405"]) == 100.0
    assert float(tbl.loc[dts[1], "RB2409"]) == 120.0


def test_replay_dominant_switch_rule_matches_script() -> None:
    dts = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rb2405 = pd.DataFrame(
        {
            "date": dts,
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [100, 110, 120],
            "hold": [100, 100, 90],
            "settle": [10.0, 10.1, 10.2],
            "amount": [10000, 11000, 12000],
        }
    )
    rb2409 = pd.DataFrame(
        {
            "date": dts,
            "open": [11.0, 11.1, 11.2],
            "high": [11.2, 11.3, 11.4],
            "low": [10.9, 11.0, 11.1],
            "close": [11.1, 11.2, 11.3],
            "volume": [200, 210, 220],
            "hold": [90, 200, 220],
            "settle": [11.0, 11.1, 11.2],
            "amount": [20000, 21000, 22000],
        }
    )

    replay_df, switch_df = _replay_dominant(
        {"RB2405": rb2405, "RB2409": rb2409}, switch_threshold=1.1
    )

    assert list(replay_df["dominant_symbol"]) == ["RB2405", "RB2405", "RB2409"]
    assert len(switch_df) == 1
    assert switch_df.iloc[0]["date"] == "2024-01-04"
    assert switch_df.iloc[0]["from_symbol"] == "RB2405"
    assert switch_df.iloc[0]["to_symbol"] == "RB2409"


def test_replay_dominant_never_returns_to_former_main() -> None:
    """Once a contract has been dominant, it cannot become dominant again."""
    dts = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rb2405 = pd.DataFrame(
        {
            "date": dts,
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [10.2, 10.3, 10.4, 10.5],
            "low": [9.9, 10.0, 10.1, 10.2],
            "close": [10.1, 10.2, 10.3, 10.4],
            "volume": [100, 110, 120, 130],
            "hold": [100, 100, 90, 400],
            "settle": [10.0, 10.1, 10.2, 10.3],
            "amount": [10000, 11000, 12000, 13000],
        }
    )
    rb2409 = pd.DataFrame(
        {
            "date": dts,
            "open": [11.0, 11.1, 11.2, 11.3],
            "high": [11.2, 11.3, 11.4, 11.5],
            "low": [10.9, 11.0, 11.1, 11.2],
            "close": [11.1, 11.2, 11.3, 11.4],
            "volume": [200, 210, 220, 230],
            "hold": [90, 200, 220, 100],
            "settle": [11.0, 11.1, 11.2, 11.3],
            "amount": [20000, 21000, 22000, 23000],
        }
    )

    replay_df, _switch_df = _replay_dominant(
        {"RB2405": rb2405, "RB2409": rb2409}, switch_threshold=1.1
    )

    assert list(replay_df["dominant_symbol"]) == [
        "RB2405",
        "RB2405",
        "RB2409",
        "RB2409",
    ]


def test_forward_adjust_pre_delta_aligns_prev_day_close_to_new_contract() -> None:
    """888/qfq: at T−1 (last old-dominant bar), adjusted close equals new contract close."""
    d0, d1, d2 = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    old_sym, new_sym = "RB2405", "RB2409"
    old_df = pd.DataFrame(
        {
            "date": [d0, d1, d2],
            "open": [10.0, 10.0, 9.0],
            "high": [10.2, 10.2, 9.1],
            "low": [9.8, 9.8, 8.9],
            "close": [10.0, 10.0, 9.0],
            "volume": [1, 1, 1],
            "hold": [100, 100, 200],
            "settle": [10.0, 10.0, 9.0],
        }
    ).set_index("date")
    new_df = pd.DataFrame(
        {
            "date": [d0, d1, d2],
            "open": [8.0, 8.0, 9.0],
            "high": [8.1, 8.1, 9.1],
            "low": [7.9, 7.9, 8.9],
            "close": [8.0, 8.0, 9.0],
            "volume": [1, 1, 1],
            "hold": [50, 100, 200],
            "settle": [8.0, 8.0, 9.0],
        }
    ).set_index("date")
    panel = {old_sym: old_df, new_sym: new_df}
    replay88 = pd.DataFrame(
        {
            "date": [d0, d1, d2],
            "dominant_symbol": [old_sym, old_sym, new_sym],
            "open": [10.0, 10.0, 9.0],
            "high": [10.2, 10.2, 9.1],
            "low": [9.8, 9.8, 8.9],
            "close": [10.0, 10.0, 9.0],
            "volume": [1, 1, 1],
            "hold": [100, 100, 200],
            "settle": [10.0, 10.0, 9.0],
        }
    )
    switch_df = pd.DataFrame(
        {
            "date": ["2024-01-04"],
            "from_symbol": [old_sym],
            "to_symbol": [new_sym],
        }
    )
    q888, _ = _build_adjusted_continuous(replay88, switch_df, panel)
    row_d1 = q888.set_index("date").loc[d1]
    assert abs(float(row_d1["close"]) - float(new_df.loc[d1, "close"])) < 1e-9


def test_backward_adjust_post_delta_aligns_switch_day_open_to_old_contract() -> None:
    """889/hfq: on roll day T, adjusted open equals old-main open on the same day.

    Raw 88 at T quotes the new dominant; post_delta = open_old(T) − open_new(T) is added
    to all dates ≥ T, so open_889(T) = open_new + post_delta = open_old(T). This is the
    complement to 888's sign convention and is not the same bug class as the old qfq flip.
    """
    d0, d1, d2 = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    old_sym, new_sym = "RB2405", "RB2409"
    old_df = pd.DataFrame(
        {
            "date": [d0, d1, d2],
            "open": [10.0, 10.0, 10.5],
            "high": [10.2, 10.2, 10.6],
            "low": [9.8, 9.8, 8.9],
            "close": [10.0, 10.0, 9.5],
            "volume": [1, 1, 1],
            "hold": [100, 100, 200],
            "settle": [10.0, 10.0, 9.5],
        }
    ).set_index("date")
    new_df = pd.DataFrame(
        {
            "date": [d0, d1, d2],
            "open": [8.0, 8.0, 9.0],
            "high": [8.1, 8.1, 9.1],
            "low": [7.9, 7.9, 8.9],
            "close": [8.0, 8.0, 9.0],
            "volume": [1, 1, 1],
            "hold": [50, 100, 200],
            "settle": [8.0, 8.0, 9.0],
        }
    ).set_index("date")
    panel = {old_sym: old_df, new_sym: new_df}
    replay88 = pd.DataFrame(
        {
            "date": [d0, d1, d2],
            "dominant_symbol": [old_sym, old_sym, new_sym],
            "open": [10.0, 10.0, 9.0],
            "high": [10.2, 10.2, 9.1],
            "low": [9.8, 9.8, 8.9],
            "close": [10.0, 10.0, 9.0],
            "volume": [1, 1, 1],
            "hold": [100, 100, 200],
            "settle": [10.0, 10.0, 9.0],
        }
    )
    switch_df = pd.DataFrame(
        {
            "date": ["2024-01-04"],
            "from_symbol": [old_sym],
            "to_symbol": [new_sym],
        }
    )
    _, q889 = _build_adjusted_continuous(replay88, switch_df, panel)
    row_d2 = q889.set_index("date").loc[d2]
    assert abs(float(row_d2["open"]) - float(old_df.loc[d2, "open"])) < 1e-9


def test_contract_yymm_suffix() -> None:
    assert _contract_yymm_suffix("RB2606", root="RB") == "2606"
    assert _contract_yymm_suffix("IF2503", root="IF") == "2503"
    assert _contract_yymm_suffix("XX1", root="X") is None


def test_attach_switch_suffix_on_roll_date() -> None:
    dts = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    replay_df = pd.DataFrame(
        {
            "date": dts,
            "dominant_symbol": ["RB2405", "RB2405", "RB2409"],
        }
    )
    switch_df = pd.DataFrame(
        {
            "date": ["2024-01-04"],
            "from_symbol": ["RB2405"],
            "to_symbol": ["RB2409"],
        }
    )
    out = _attach_switch_suffix_column(replay_df, switch_df, root="RB")
    assert out["dominant_contract_suffix"].tolist() == [None, None, "2409"]


def test_sanitize_settle_fallback_for_non_positive_and_outlier_values() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-12-29", "2021-12-30", "2021-12-31"]),
            "open": [6079.0, 5925.0, 5917.0],
            "high": [6082.0, 5942.0, 5948.0],
            "low": [5921.0, 5895.0, 5901.0],
            "close": [5929.0, 5917.0, 5932.0],
            "settle": [5974.0, 54.0, 0.0],
        }
    )
    out, fixed = _sanitize_settle_column(df, code="A2405")
    assert fixed == 2
    # Keep reasonable settle
    assert float(out.loc[0, "settle"]) == 5974.0
    # Outlier fallback to close
    assert float(out.loc[1, "settle"]) == 5917.0
    # Non-positive fallback to close
    assert float(out.loc[2, "settle"]) == 5932.0


def test_sanitize_settle_keeps_values_when_deviation_is_reasonable() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "settle": [101.8, 100.9],
        }
    )
    out, fixed = _sanitize_settle_column(df, code="RB2405")
    assert fixed == 0
    assert out["settle"].tolist() == [101.8, 100.9]


def test_collect_auto_corrections_replaces_points_exceeding_threshold() -> None:
    dts = pd.to_datetime(["2024-01-02", "2024-01-03"])
    replay88 = pd.DataFrame(
        {
            "date": dts,
            "open": [100.0, 100.0],
            "high": [110.0, 110.0],
            "low": [95.0, 95.0],
            "close": [100.0, 100.0],
            "settle": [100.0, 100.0],
            "volume": [1.0, 1.0],
            "amount": [1.0, 1.0],
            "hold": [1.0, 1.0],
            "dominant_contract_suffix": [None, None],
            "roll_from_symbol": [None, None],
            "roll_to_symbol": [None, None],
            "roll_from_open": [None, None],
            "roll_from_close": [None, None],
            "roll_to_open": [None, None],
            "roll_to_close": [None, None],
        }
    )
    main0 = pd.DataFrame(
        {
            "date": dts,
            "open": [100.0, 80.0],  # day2 APE=25%
            "high": [110.0, 108.0],
            "low": [95.0, 95.0],
            "close": [100.0, 100.0],
            "settle": [100.0, 70.0],  # day2 APE=42.86%
            "volume": [1.0, 1.0],
            "amount": [1.0, 1.0],
            "hold": [1.0, 1.0],
        }
    )
    corrected, points = _collect_auto_corrections(replay88, main0, rel_p95_max=0.05)
    corrected_idx = corrected.set_index("date")
    d2 = pd.Timestamp("2024-01-03")
    assert float(corrected_idx.loc[d2, "open"]) == 80.0
    assert float(corrected_idx.loc[d2, "settle"]) == 70.0
    assert len(points) == 2
    assert sorted({str(p["field"]) for p in points}) == ["open", "settle"]
