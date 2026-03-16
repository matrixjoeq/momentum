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

