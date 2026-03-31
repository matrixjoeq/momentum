from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BUCKET_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "2pct": {
        "edges": (-np.inf, -0.02, 0.0, 0.02, np.inf),
        "labels": ("lt_m2", "m2_0", "p0_p2", "ge_p2"),
    },
    "5pct": {
        "edges": (-np.inf, -0.05, -0.02, 0.0, 0.02, 0.05, np.inf),
        "labels": ("lt_m5", "m5_m2", "m2_0", "p0_p2", "p2_p5", "ge_p5"),
    },
    "10pct": {
        "edges": (-np.inf, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, np.inf),
        "labels": ("lt_m10", "m10_m5", "m5_m2", "m2_0", "p0_p2", "p2_p5", "p5_p10", "ge_p10"),
    },
}


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


def _bucket_probs(arr: np.ndarray) -> dict[str, float]:
    x = arr[np.isfinite(arr)]
    if x.size <= 0:
        return {k: float("nan") for k in BUCKET_PROFILE_SPECS["5pct"]["labels"]}
    counts, _ = np.histogram(x, bins=np.asarray(BUCKET_PROFILE_SPECS["5pct"]["edges"], dtype=float))
    probs = counts.astype(float) / float(x.size)
    return {k: float(v) for k, v in zip(BUCKET_PROFILE_SPECS["5pct"]["labels"], probs)}


def _bucket_probs_with_spec(arr: np.ndarray, *, edges: tuple[float, ...], labels: tuple[str, ...]) -> dict[str, float]:
    x = arr[np.isfinite(arr)]
    if x.size <= 0:
        return {k: float("nan") for k in labels}
    counts, _ = np.histogram(x, bins=np.asarray(edges, dtype=float))
    probs = counts.astype(float) / float(x.size)
    return {k: float(v) for k, v in zip(labels, probs)}


def _bucket_profiles(arr: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, spec in BUCKET_PROFILE_SPECS.items():
        out[name] = _bucket_probs_with_spec(
            arr,
            edges=tuple(spec["edges"]),
            labels=tuple(spec["labels"]),
        )
    return out


def entry_dates_from_exposure(exposure: pd.Series, *, eps: float = 1e-12) -> list[str]:
    e = pd.to_numeric(exposure, errors="coerce").astype(float).fillna(0.0)
    prev = e.shift(1).fillna(0.0)
    m = (prev <= float(eps)) & (e > float(eps))
    return e.index[m].date.astype(str).tolist()


def entry_dates_from_weight_membership_change(weights: pd.DataFrame, *, eps: float = 1e-12) -> list[str]:
    """
    Rotation-style entry events:
    - Each date where the held symbol membership changes vs previous date
      is treated as one entry event date.
    - Membership means weight > eps.
    """
    if weights is None or weights.empty:
        return []
    w = weights.apply(pd.to_numeric, errors="coerce").astype(float).fillna(0.0)
    active = w.gt(float(eps))
    prev_active = active.shift(1, fill_value=False)
    changed = active.ne(prev_active).any(axis=1)
    return w.index[changed].date.astype(str).tolist()


def compute_event_study(
    *,
    dates: pd.Index,
    daily_returns: pd.Series,
    entry_dates: list[str],
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    random_trials: int = 1000,
    bootstrap_trials: int = 1000,
    net_cost_threshold: float = 0.0,
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
        sig_profit = float(np.mean(sig > 0.0)) if sig_n else float("nan")
        sig_profit_net = float(np.mean(sig > float(net_cost_threshold))) if sig_n else float("nan")
        sig_buckets = _bucket_probs(sig)
        sig_tail_loss_2 = float(np.mean(sig <= -0.02)) if sig_n else float("nan")
        sig_tail_gain_2 = float(np.mean(sig >= 0.02)) if sig_n else float("nan")
        sig_tail_loss_5 = float(np.mean(sig <= -0.05)) if sig_n else float("nan")
        sig_tail_gain_5 = float(np.mean(sig >= 0.05)) if sig_n else float("nan")

        elig_idx = np.arange(valid_max + 1, dtype=int)
        elig_wr = wr_all[: valid_max + 1]
        elig_idx = elig_idx[np.isfinite(elig_wr)]
        elig_wr = elig_wr[np.isfinite(elig_wr)]
        trial_means: list[float] = []
        trial_profit: list[float] = []
        trial_profit_net: list[float] = []
        trial_bucket_probs: dict[str, dict[str, list[float]]] = {
            name: {k: [] for k in tuple(spec["labels"])}
            for name, spec in BUCKET_PROFILE_SPECS.items()
        }
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
                    trial_profit.append(float(np.mean(vals > 0.0)))
                    trial_profit_net.append(float(np.mean(vals > float(net_cost_threshold))))
                    b_profiles = _bucket_profiles(vals)
                    for pname, pb in b_profiles.items():
                        for bk, vv in pb.items():
                            if np.isfinite(vv):
                                trial_bucket_probs[pname][bk].append(float(vv))
        trial_arr = np.asarray(trial_means, dtype=float) if trial_means else np.asarray([], dtype=float)
        rnd_mean = float(np.mean(trial_arr)) if trial_arr.size else float("nan")
        rnd_ci_lo = float(np.percentile(trial_arr, 2.5)) if trial_arr.size else float("nan")
        rnd_ci_hi = float(np.percentile(trial_arr, 97.5)) if trial_arr.size else float("nan")
        trial_profit_arr = np.asarray(trial_profit, dtype=float) if trial_profit else np.asarray([], dtype=float)
        trial_profit_net_arr = np.asarray(trial_profit_net, dtype=float) if trial_profit_net else np.asarray([], dtype=float)
        rnd_profit = float(np.mean(trial_profit_arr)) if trial_profit_arr.size else float("nan")
        rnd_profit_ci = [
            float(np.percentile(trial_profit_arr, 2.5)) if trial_profit_arr.size else float("nan"),
            float(np.percentile(trial_profit_arr, 97.5)) if trial_profit_arr.size else float("nan"),
        ]
        rnd_profit_net = float(np.mean(trial_profit_net_arr)) if trial_profit_net_arr.size else float("nan")
        rnd_profit_net_ci = [
            float(np.percentile(trial_profit_net_arr, 2.5)) if trial_profit_net_arr.size else float("nan"),
            float(np.percentile(trial_profit_net_arr, 97.5)) if trial_profit_net_arr.size else float("nan"),
        ]
        rnd_bucket_mean: dict[str, float] = {}
        rnd_bucket_ci95: dict[str, list[float]] = {}
        rnd_bucket_profiles_mean: dict[str, dict[str, float]] = {}
        rnd_bucket_profiles_ci95: dict[str, dict[str, list[float]]] = {}
        for pname, spec in BUCKET_PROFILE_SPECS.items():
            labels = tuple(spec["labels"])
            one_mean: dict[str, float] = {}
            one_ci: dict[str, list[float]] = {}
            for bk in labels:
                arr_b = np.asarray(trial_bucket_probs[pname].get(bk, []), dtype=float)
                one_mean[bk] = float(np.mean(arr_b)) if arr_b.size else float("nan")
                one_ci[bk] = [
                    float(np.percentile(arr_b, 2.5)) if arr_b.size else float("nan"),
                    float(np.percentile(arr_b, 97.5)) if arr_b.size else float("nan"),
                ]
            rnd_bucket_profiles_mean[pname] = one_mean
            rnd_bucket_profiles_ci95[pname] = one_ci
        # backward compatibility: 5pct profile
        rnd_bucket_mean = dict(rnd_bucket_profiles_mean.get("5pct", {}))
        rnd_bucket_ci95 = dict(rnd_bucket_profiles_ci95.get("5pct", {}))
        p_out = (
            float((np.sum(trial_arr >= sig_mean) + 1) / (trial_arr.size + 1))
            if (trial_arr.size and np.isfinite(sig_mean))
            else float("nan")
        )
        p_profit = (
            float((np.sum(trial_profit_arr >= sig_profit) + 1) / (trial_profit_arr.size + 1))
            if (trial_profit_arr.size and np.isfinite(sig_profit))
            else float("nan")
        )
        p_profit_net = (
            float((np.sum(trial_profit_net_arr >= sig_profit_net) + 1) / (trial_profit_net_arr.size + 1))
            if (trial_profit_net_arr.size and np.isfinite(sig_profit_net))
            else float("nan")
        )
        odds = (
            float(sig_profit / (1.0 - sig_profit))
            if (np.isfinite(sig_profit) and sig_profit > 0.0 and sig_profit < 1.0)
            else float("nan")
        )

        sig_bucket_profiles = _bucket_profiles(sig)
        sig_buckets = dict(sig_bucket_profiles.get("5pct", sig_buckets))
        windows[f"{h2}d"] = {
            "signal": {
                "n": sig_n,
                "mean": sig_mean,
                "std": float(np.std(sig, ddof=1)) if sig_n >= 2 else float("nan"),
                "win_rate": float(np.mean(sig > 0.0)) if sig_n else float("nan"),
                "profit_frequency": sig_profit,
                "profit_frequency_net_cost": sig_profit_net,
                "bucket_probabilities": sig_buckets,
                "bucket_profiles": sig_bucket_profiles,
                "tail_probabilities": {
                    "loss_ge_2pct": sig_tail_loss_2,
                    "gain_ge_2pct": sig_tail_gain_2,
                    "loss_ge_5pct": sig_tail_loss_5,
                    "gain_ge_5pct": sig_tail_gain_5,
                },
                "quantiles": _quantiles(sig),
                "ci95_mean": [sig_ci_lo, sig_ci_hi],
            },
            "random_baseline": {
                "n_trials": int(trial_arr.size),
                "mean_of_trial_means": rnd_mean,
                "ci95_trial_mean": [rnd_ci_lo, rnd_ci_hi],
                "std_of_trial_means": float(np.std(trial_arr, ddof=1)) if trial_arr.size >= 2 else float("nan"),
                "profit_frequency_mean": rnd_profit,
                "profit_frequency_ci95": rnd_profit_ci,
                "profit_frequency_net_cost_mean": rnd_profit_net,
                "profit_frequency_net_cost_ci95": rnd_profit_net_ci,
                "bucket_probabilities_mean": rnd_bucket_mean,
                "bucket_probabilities_ci95": rnd_bucket_ci95,
                "bucket_profiles_mean": rnd_bucket_profiles_mean,
                "bucket_profiles_ci95": rnd_bucket_profiles_ci95,
            },
            "comparison": {
                "mean_diff_vs_random": (sig_mean - rnd_mean) if (np.isfinite(sig_mean) and np.isfinite(rnd_mean)) else float("nan"),
                "p_value_outperform_random": p_out,
                "significant_5pct": bool(np.isfinite(p_out) and p_out < 0.05),
                "delta_profit_frequency": (
                    sig_profit - rnd_profit
                    if (np.isfinite(sig_profit) and np.isfinite(rnd_profit))
                    else float("nan")
                ),
                "delta_profit_frequency_net_cost": (
                    sig_profit_net - rnd_profit_net
                    if (np.isfinite(sig_profit_net) and np.isfinite(rnd_profit_net))
                    else float("nan")
                ),
                "odds_ratio_profit_vs_loss": odds,
                "p_value_profit_frequency_outperform_random": p_profit,
                "p_value_profit_frequency_net_cost_outperform_random": p_profit_net,
                "delta_bucket_probabilities": {
                    bk: (
                        float(sig_buckets.get(bk, float("nan")))
                        - float(rnd_bucket_mean.get(bk, float("nan")))
                        if (
                            np.isfinite(float(sig_buckets.get(bk, float("nan"))))
                            and np.isfinite(float(rnd_bucket_mean.get(bk, float("nan"))))
                        )
                        else float("nan")
                    )
                    for bk in tuple(BUCKET_PROFILE_SPECS["5pct"]["labels"])
                },
                "delta_bucket_profiles": {
                    pname: {
                        bk: (
                            float((sig_bucket_profiles.get(pname) or {}).get(bk, float("nan")))
                            - float((rnd_bucket_profiles_mean.get(pname) or {}).get(bk, float("nan")))
                            if (
                                np.isfinite(float((sig_bucket_profiles.get(pname) or {}).get(bk, float("nan"))))
                                and np.isfinite(float((rnd_bucket_profiles_mean.get(pname) or {}).get(bk, float("nan"))))
                            )
                            else float("nan")
                        )
                        for bk in tuple(spec["labels"])
                    }
                    for pname, spec in BUCKET_PROFILE_SPECS.items()
                },
            },
        }

    return {
        "entry_count": int(len(entry_idx_all)),
        "horizons": [int(h) for h in horizons],
        "windows": windows,
    }

