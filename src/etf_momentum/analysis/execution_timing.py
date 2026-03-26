from __future__ import annotations

import numpy as np
import pandas as pd


def apply_execution_timing_to_weights(
    weights_open_effective: pd.DataFrame,
    *,
    exec_price: str,
) -> pd.DataFrame:
    """
    Convert open-effective daily weights into execution-time-aware weights.

    Input convention:
    - `weights_open_effective[t]` means target weights are already effective on day t.
      (Equivalent to "can eat full day-t return".)

    Output convention by `exec_price`:
    - open: full day-t return belongs to day-t target weights.
    - close: day-t target weights start from t+1 return (cannot eat day-t return).
    - oc2: day-t return is split 50/50 between previous and current target weights.

    Note: this **oc2** is a **weight** split between days, not the same object as **OC2 daily return**
    in backtests (there, oc2 is typically 50% open-leg + 50% close-leg return on the price series).
    """
    ep = str(exec_price or "open").strip().lower()
    if ep not in {"open", "close", "oc2"}:
        raise ValueError("exec_price must be one of: open|close|oc2")
    w = weights_open_effective.astype(float)
    w_prev = w.shift(1).fillna(0.0)
    if ep == "open":
        return w
    if ep == "close":
        return w_prev
    return (0.5 * w + 0.5 * w_prev).astype(float)


def forward_returns(px: pd.DataFrame) -> pd.DataFrame:
    """
    One-step forward returns, indexed by the execution day t.

    ret[t] = px[t+1] / px[t] - 1
    """
    p = px.astype(float).replace([np.inf, -np.inf], np.nan)
    ret = (p.shift(-1).div(p) - 1.0).replace([np.inf, -np.inf], np.nan)
    return ret.fillna(0.0).astype(float)


def forward_align_returns(ret: pd.DataFrame) -> pd.DataFrame:
    """
    Shift a return matrix forward by one day:

    out[t] = ret[t+1]
    """
    r = ret.astype(float).replace([np.inf, -np.inf], np.nan)
    return r.shift(-1).fillna(0.0).astype(float)


def corporate_action_mask(
    gross_none: pd.DataFrame,
    gross_hfq: pd.DataFrame,
    *,
    dev_threshold: float = 0.02,
    ratio_threshold: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified corporate-action cliff detector used by strategy engines.

    Inputs are gross-return matrices aligned to the same execution horizon:
      gross = 1 + return_on_that_horizon

    Returns:
    - corp_factor: gross_hfq / gross_none
    - mask: day/code where fallback to hfq return should be used
    """
    gn = gross_none.astype(float).replace([np.inf, -np.inf], np.nan)
    gh = gross_hfq.astype(float).replace([np.inf, -np.inf], np.nan)
    corp_factor = (gh / gn).replace([np.inf, -np.inf], np.nan)
    dev = (corp_factor - 1.0).abs()
    rt = float(ratio_threshold)
    mask = (dev > float(dev_threshold)) | (corp_factor > rt) | (corp_factor < (1.0 / rt))
    return corp_factor, mask

