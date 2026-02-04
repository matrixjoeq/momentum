import datetime as dt

import numpy as np
import pandas as pd

from etf_momentum.strategy.rotation import (
    _apply_asset_vol_index_rules,
    _tiered_exposure_from_level_quantiles,
)  # pylint: disable=import-error


def test_tiered_exposure_quantiles_shift_1_no_lookahead():
    # levels on 3 consecutive CN trading days (already aligned series).
    dates = pd.to_datetime([dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)])
    levels = pd.Series([1.0, 2.0, 3.0], index=dates, dtype=float)

    expo, meta = _tiered_exposure_from_level_quantiles(
        levels,
        level_window="30d",
        quantiles=[0.5],
        exposures=[1.0, 0.0],
        min_periods=2,
    )

    assert meta["window"] == "30d"
    # With min_periods=2 and shift(1), day1/day2 are warm-up => full exposure.
    # On day3, threshold uses only [day1, day2] => median=1.5, level=3.0 is high bucket => exposure 0.
    assert expo.to_numpy(dtype=float).tolist() == [1.0, 1.0, 0.0]


def test_apply_asset_vol_index_rules_scales_weights_daily():
    dates = pd.to_datetime([dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)])
    w = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates, dtype=float)

    # Provide vol index levels indexed by dt.date to exercise index normalization.
    vix_cn = pd.Series(
        [1.0, 2.0, 3.0],
        index=[dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        dtype=float,
    )

    meta = _apply_asset_vol_index_rules(
        w,
        rules=[
            {
                "code": "AAA",
                "index": "VIX",
                "level_window": "30d",
                "level_quantiles": [0.5],
                "level_exposures": [1.0, 0.0],
                "min_periods": 2,
            }
        ],
        vol_index_close={"VIX": vix_cn},
    )

    assert meta["enabled"] is True
    assert len(meta["rules"]) == 1
    assert np.allclose(w["AAA"].to_numpy(dtype=float), np.array([1.0, 1.0, 0.0], dtype=float))

