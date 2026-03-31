from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    if arr.size <= 0:
        return {"p05": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p95": float("nan")}
    return {
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _bootstrap_ci_mean(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = values[np.isfinite(values)]
    if arr.size <= 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(max(100, n_bootstrap)), dtype=float)
    for i in range(means.size):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def entry_dates_from_exposure(exposure: pd.Series, *, eps: float = 1e-12) -> list[str]:
    e = pd.to_numeric(exposure, errors="coerce").astype(float).fillna(0.0)
    prev = e.shift(1).fillna(0.0)
    m = (prev <= float(eps)) & (e > float(eps))
    return e.index[m].date.astype(str).tolist()


def compute_event_study(
    *,
    dates: pd.Index,
    daily_returns: pd.Series,
    entry_dates: list[str],
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    random_trials: int = 1000,
    bootstrap_trials: int = 1000,
) -> dict[str, Any]:
    idx = pd.to_datetime(dates)
    ret = pd.to_numeric(daily_returns, errors="coerce").astype(float).reindex(idx).fillna(0.0).to_numpy(dtype=float)
    n = int(len(ret))
    if n <= 1:
        return {"entry_count": 0, "horizons": list(horizons), "windows": {}}

    entry_idx_all: list[int] = []
    for d in entry_dates or []:
        try:
            ts = pd.to_datetime(str(d)).normalize()
        except (TypeError, ValueError):
            continue
        if ts in idx:
            entry_idx_all.append(int(idx.get_loc(ts)))
    entry_idx_all = sorted(set(entry_idx_all))

    windows: dict[str, Any] = {}
    for h in horizons:
        h2 = int(max(1, h))
        valid_max = n - h2
        if valid_max < 0:
            continue

        gross = 1.0 + ret
        # window return from t to t+h-1
        wr_all = np.full(n, np.nan, dtype=float)
        for i in range(valid_max + 1):
            wr_all[i] = float(np.prod(gross[i : i + h2]) - 1.0)

        sig_idx = [i for i in entry_idx_all if 0 <= i <= valid_max and np.isfinite(wr_all[i])]
        sig = np.asarray([wr_all[i] for i in sig_idx], dtype=float)
        sig = sig[np.isfinite(sig)]
        sig_n = int(sig.size)
        sig_mean = float(np.mean(sig)) if sig_n else float("nan")
        sig_ci_lo, sig_ci_hi = _bootstrap_ci_mean(sig, n_bootstrap=int(bootstrap_trials), seed=1000 + h2)

        elig_idx = np.arange(valid_max + 1, dtype=int)
        elig_wr = wr_all[: valid_max + 1]
        elig_idx = elig_idx[np.isfinite(elig_wr)]
        elig_wr = elig_wr[np.isfinite(elig_wr)]
        trial_means: list[float] = []
        if sig_n > 0 and elig_idx.size > 0:
            rng = np.random.default_rng(2000 + h2)
            replace = bool(sig_n > elig_idx.size)
            tN = int(max(200, random_trials))
            for _ in range(tN):
                pick = rng.choice(elig_idx, size=sig_n, replace=replace)
                vals = wr_all[pick]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    trial_means.append(float(np.mean(vals)))
        trial_arr = np.asarray(trial_means, dtype=float) if trial_means else np.asarray([], dtype=float)
        rnd_mean = float(np.mean(trial_arr)) if trial_arr.size else float("nan")
        rnd_ci_lo = float(np.percentile(trial_arr, 2.5)) if trial_arr.size else float("nan")
        rnd_ci_hi = float(np.percentile(trial_arr, 97.5)) if trial_arr.size else float("nan")
        p_out = (
            float((np.sum(trial_arr >= sig_mean) + 1) / (trial_arr.size + 1))
            if (trial_arr.size and np.isfinite(sig_mean))
            else float("nan")
        )

        windows[f"{h2}d"] = {
            "signal": {
                "n": sig_n,
                "mean": sig_mean,
                "std": float(np.std(sig, ddof=1)) if sig_n >= 2 else float("nan"),
                "win_rate": float(np.mean(sig > 0.0)) if sig_n else float("nan"),
                "quantiles": _quantiles(sig),
                "ci95_mean": [sig_ci_lo, sig_ci_hi],
            },
            "random_baseline": {
                "n_trials": int(trial_arr.size),
                "mean_of_trial_means": rnd_mean,
                "ci95_trial_mean": [rnd_ci_lo, rnd_ci_hi],
                "std_of_trial_means": float(np.std(trial_arr, ddof=1)) if trial_arr.size >= 2 else float("nan"),
            },
            "comparison": {
                "mean_diff_vs_random": (sig_mean - rnd_mean) if (np.isfinite(sig_mean) and np.isfinite(rnd_mean)) else float("nan"),
                "p_value_outperform_random": p_out,
                "significant_5pct": bool(np.isfinite(p_out) and p_out < 0.05),
            },
        }

    return {
        "entry_count": int(len(entry_idx_all)),
        "horizons": [int(h) for h in horizons],
        "windows": windows,
    }

