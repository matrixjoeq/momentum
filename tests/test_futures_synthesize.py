from __future__ import annotations

import pandas as pd

from etf_momentum.data.futures_synthesize import _build_hold_table, _replay_dominant


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
