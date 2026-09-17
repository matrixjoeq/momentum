from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_VERSION = "off_fund_replication_v2"
MODEL_CALIBRATION_STATUS = "independent_walk_forward_v1"
MODEL_CALIBRATION_DATASET = "off_fund_replication_v2_passive_index_2024_2025"
WEIGHT_TOLERANCE = 1e-8
KKT_TOLERANCE = 2e-5
RETURN_SCALE_FLOOR = 1e-6
NEAR_ZERO_VOL_ABS = 1e-10
NEAR_ZERO_VOL_REL = 1e-3
DEFAULT_LAMBDA_SUBSTITUTION = 2e-3
DEFAULT_LAMBDA_TEMPORAL = 2e-1
MAX_WINDOWS_PER_TARGET = 3200
MAX_TARGETS = 50
MAX_FACTORS = 40
MAX_EXTRA_SOLVES = 30
DEFAULT_MAX_START_SHIFT = 20
IDENTIFIABILITY_SSE_RATIO = 0.01
MAX_INTERVAL_GROUPS = 4
MAX_SENSITIVITY_FACTORS = 8
STABILITY_GROUP_DISTANCE_THRESHOLD = 0.02
STABILITY_TE_IMPROVEMENT_THRESHOLD = 0.001


@dataclass(frozen=True)
class ReplicationFactorMeta:
    key: str
    label: str
    reporting_group: str
    substitution_group: str
    asset_class: str


@dataclass(frozen=True)
class _FitResult:
    weights: np.ndarray
    objective: float
    iterations: int
    primal_violation: float
    kkt_residual: float
    active_keys: tuple[str, ...]
    dropped_near_zero_vol: tuple[str, ...]
    reporting_groups: tuple[str, ...]
    group_weights: dict[str, float]
    active_group_weights: dict[str, float]
    fit_tracking_error_annualized: float
    temporal_smoothing_used: bool
    gradient_check_error: float


class ReplicationConfigError(ValueError):
    """Raised when factor metadata makes replication non-identifiable."""


def nav_to_returns(nav: pd.Series) -> pd.Series:
    """Normalize a NAV/price series and calculate daily simple returns."""
    series = pd.to_numeric(nav, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if series.empty or not bool(series.notna().any()):
        return pd.Series(dtype=float)
    index = pd.to_datetime(series.index, errors="coerce")
    valid = ~pd.isna(index)
    if not bool(np.all(valid)):
        series = series.iloc[np.asarray(valid, dtype=bool)]
        index = index[valid]
    if series.empty:
        return pd.Series(dtype=float)
    date_index = pd.DatetimeIndex(index)
    if date_index.tz is not None:
        date_index = date_index.tz_localize(None)
    series.index = pd.DatetimeIndex(date_index.date)
    if not series.index.is_unique:
        series = series.groupby(level=0).last()
    series = series.sort_index()
    # Prices must be strictly positive. Replace <= 0 with NaN to avoid -1.0 returns.
    series = series.where(series > 0)
    # Preserve missing observations until after pct_change. Dropping them
    # first would turn a D0→D2 move into a fake D1→D2 daily return.
    return (
        series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    )


def select_replication_factor_series(
    *,
    close_df: pd.DataFrame,
    factor_rows: list[Mapping[str, Any]],
    target_nav: pd.Series,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Select aliases jointly to maximize target/factor common coverage."""
    target_index = nav_to_returns(target_nav).index
    candidates: dict[str, list[tuple[int, str, pd.Series, pd.DatetimeIndex]]] = {}
    row_by_key: dict[str, Mapping[str, Any]] = {}
    warnings: list[str] = []
    for raw_row in factor_rows:
        key = str(raw_row.get("key") or "").strip().upper()
        if not key:
            continue
        row_by_key[key] = raw_row
        options: list[tuple[int, str, pd.Series, pd.DatetimeIndex]] = []
        for priority, raw_alias in enumerate(raw_row.get("aliases") or []):
            alias = str(raw_alias).strip()
            if not alias or alias not in close_df.columns:
                continue
            close = pd.to_numeric(close_df[alias], errors="coerce")
            return_index = nav_to_returns(close).index
            if len(return_index) < 2:
                continue
            options.append((priority, alias, close, return_index))
        if options:
            candidates[key] = options
        else:
            warnings.append(f"{key}: no usable alias")
    keys = sorted(candidates)
    if not keys or target_index.empty:
        return pd.DataFrame(), [], warnings

    beam_width = 256

    def coverage_rank(
        common: pd.DatetimeIndex,
        priority_sum: int,
        aliases: tuple[str, ...],
    ) -> tuple[int, int, int, tuple[str, ...]]:
        common = common.sort_values()
        if common.empty:
            return (len(target_index), 0, priority_sum, aliases)
        first = pd.Timestamp(common[0])
        shift = sum(pd.Timestamp(date) < first for date in target_index)
        return (int(shift), -int(len(common)), priority_sum, aliases)

    beam: list[
        tuple[
            dict[str, tuple[int, str, pd.Series, pd.DatetimeIndex]],
            pd.DatetimeIndex,
            int,
        ]
    ] = [({}, target_index, 0)]
    for key in keys:
        expanded: list[
            tuple[
                dict[str, tuple[int, str, pd.Series, pd.DatetimeIndex]],
                pd.DatetimeIndex,
                int,
            ]
        ] = []
        for selected, common, priority_sum in beam:
            for option in candidates[key]:
                priority, _, _, return_index = option
                next_selected = dict(selected)
                next_selected[key] = option
                expanded.append(
                    (
                        next_selected,
                        common.intersection(return_index),
                        priority_sum + priority,
                    )
                )
        expanded.sort(
            key=lambda state: coverage_rank(
                state[1],
                state[2],
                tuple(
                    state[0][selected_key][1]
                    for selected_key in keys
                    if selected_key in state[0]
                ),
            )
        )
        deduplicated: list[
            tuple[
                dict[str, tuple[int, str, pd.Series, pd.DatetimeIndex]],
                pd.DatetimeIndex,
                int,
            ]
        ] = []
        seen_common: set[tuple[int, ...]] = set()
        for state in expanded:
            signature = tuple(pd.DatetimeIndex(state[1]).asi8.tolist())
            if signature in seen_common:
                continue
            seen_common.add(signature)
            deduplicated.append(state)
            if len(deduplicated) >= beam_width:
                break
        beam = deduplicated
    choice = beam[0][0]

    selected_columns: dict[str, pd.Series] = {}
    selected_meta: list[dict[str, Any]] = []
    for key in keys:
        _, alias, close, _ = choice[key]
        selected_columns[key] = close
        row = row_by_key[key]
        selected_meta.append(
            {
                "key": key,
                "label": str(row.get("label") or key),
                "aliases": [str(item) for item in row.get("aliases") or []],
                "selected_code": alias,
                "reporting_group": str(row.get("reporting_group") or key),
                "substitution_group": str(row.get("substitution_group") or key),
                "asset_class": str(row.get("asset_class") or "unknown"),
            }
        )
    return pd.DataFrame(selected_columns), selected_meta, warnings


def _project_capped_simplex(values: np.ndarray) -> np.ndarray:
    """Project onto {w >= 0, sum(w) <= 1}."""
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    if float(positive.sum()) <= 1.0:
        return positive
    ordered = np.sort(positive)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    idx = np.arange(1, ordered.size + 1)
    valid = ordered - cssv / idx > 0
    rho = int(np.flatnonzero(valid)[-1])
    theta = float(cssv[rho] / (rho + 1))
    return np.maximum(positive - theta, 0.0)


def _group_matrix(
    keys: list[str],
    factor_meta: Mapping[str, ReplicationFactorMeta],
) -> tuple[np.ndarray, tuple[str, ...]]:
    groups = tuple(sorted({factor_meta[key].reporting_group for key in keys}))
    matrix = np.zeros((len(groups), len(keys)), dtype=float)
    group_idx = {group: i for i, group in enumerate(groups)}
    for j, key in enumerate(keys):
        matrix[group_idx[factor_meta[key].reporting_group], j] = 1.0
    return matrix, groups


def _substitution_edges(
    keys: list[str],
    factor_meta: Mapping[str, ReplicationFactorMeta],
) -> list[tuple[int, int, float]]:
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        group = factor_meta[key].substitution_group
        if group:
            members.setdefault(group, []).append(i)
    edges: list[tuple[int, int, float]] = []
    for indexes in members.values():
        edge_count = len(indexes) * (len(indexes) - 1) // 2
        if edge_count <= 0:
            continue
        edge_weight = 1.0 / float(edge_count)
        for left_pos, left in enumerate(indexes):
            for right in indexes[left_pos + 1 :]:
                edges.append((left, right, edge_weight))
    return edges


def _collapse_exact_duplicates(
    x: np.ndarray,
    keys: list[str],
    factor_meta: Mapping[str, ReplicationFactorMeta],
) -> tuple[np.ndarray, list[str], list[list[int]]]:
    groups: list[list[int]] = []
    reduced_keys: list[str] = []
    for idx, key in enumerate(keys):
        duplicate_group: int | None = None
        for group_idx, indexes in enumerate(groups):
            representative = indexes[0]
            if float(np.max(np.abs(x[:, idx] - x[:, representative]))) <= 1e-12:
                other_key = keys[representative]
                if (
                    factor_meta[key].substitution_group
                    != factor_meta[other_key].substitution_group
                ):
                    raise ReplicationConfigError(
                        "exact duplicate factors must share substitution_group: "
                        f"{other_key}, {key}"
                    )
                if (
                    factor_meta[key].reporting_group
                    != factor_meta[other_key].reporting_group
                ):
                    raise ReplicationConfigError(
                        "exact duplicate factors must share reporting_group: "
                        f"{other_key}, {key}"
                    )
                duplicate_group = group_idx
                break
        if duplicate_group is None:
            groups.append([idx])
            reduced_keys.append(key)
        else:
            groups[duplicate_group].append(idx)
    reduced = np.column_stack([x[:, indexes[0]] for indexes in groups])
    return reduced, reduced_keys, groups


def _validate_exact_duplicate_groups(
    x: np.ndarray,
    keys: list[str],
    factor_meta: Mapping[str, ReplicationFactorMeta],
) -> None:
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            if float(np.max(np.abs(x[:, left] - x[:, right]))) > 1e-12:
                continue
            left_meta = factor_meta[keys[left]]
            right_meta = factor_meta[keys[right]]
            if left_meta.substitution_group != right_meta.substitution_group:
                raise ReplicationConfigError(
                    "exact duplicate factors must share substitution_group: "
                    f"{keys[left]}, {keys[right]}"
                )
            if left_meta.reporting_group != right_meta.reporting_group:
                raise ReplicationConfigError(
                    "exact duplicate factors must share reporting_group: "
                    f"{keys[left]}, {keys[right]}"
                )


def _expand_duplicate_weights(
    reduced_weights: np.ndarray,
    duplicate_groups: list[list[int]],
    size: int,
) -> np.ndarray:
    expanded = np.zeros(size, dtype=float)
    for reduced_idx, indexes in enumerate(duplicate_groups):
        share = float(reduced_weights[reduced_idx]) / float(len(indexes))
        for idx in indexes:
            expanded[idx] = share
    return expanded


def _fit_window(
    *,
    x: np.ndarray,
    y: np.ndarray,
    keys: list[str],
    factor_meta: Mapping[str, ReplicationFactorMeta],
    previous_asset_weights: Mapping[str, float] | None,
    previous_group_weights: Mapping[str, float] | None,
    lambda_substitution: float,
    lambda_temporal: float,
) -> _FitResult | None:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    _validate_exact_duplicate_groups(x, keys, factor_meta)
    y_centered = y - float(np.mean(y))
    target_vol = float(np.std(y_centered, ddof=1))
    scale = max(target_vol, RETURN_SCALE_FLOOR)
    factor_vol = np.std(x - np.mean(x, axis=0), axis=0, ddof=1)
    vol_floor = max(NEAR_ZERO_VOL_ABS, NEAR_ZERO_VOL_REL * scale)
    active_mask = np.isfinite(factor_vol) & (factor_vol > vol_floor)
    dropped = tuple(key for key, active in zip(keys, active_mask) if not active)
    if not bool(np.any(active_mask)):
        return None

    active_keys = [key for key, active in zip(keys, active_mask) if active]
    x_active = x[:, active_mask]
    x_reduced, reduced_keys, duplicate_groups = _collapse_exact_duplicates(
        x_active, active_keys, factor_meta
    )
    x_centered = x_reduced - np.mean(x_reduced, axis=0)
    group_matrix, reporting_groups = _group_matrix(reduced_keys, factor_meta)
    edges = _substitution_edges(reduced_keys, factor_meta)
    previous = np.array(
        [
            float((previous_group_weights or {}).get(group, 0.0))
            for group in reporting_groups
        ],
        dtype=float,
    )
    use_temporal = previous_group_weights is not None and set(
        previous_group_weights
    ) == set(reporting_groups)

    def objective(weights: np.ndarray) -> float:
        residual = (y_centered - x_centered @ weights) / scale
        value = float(np.mean(np.square(residual)))
        if edges and lambda_substitution > 0.0:
            value += float(lambda_substitution) * sum(
                edge_weight * float((weights[left] - weights[right]) ** 2)
                for left, right, edge_weight in edges
            )
        if use_temporal and lambda_temporal > 0.0:
            group_delta = group_matrix @ weights - previous
            value += float(lambda_temporal) * float(group_delta @ group_delta)
        return value

    def gradient(weights: np.ndarray) -> np.ndarray:
        residual = (x_centered @ weights - y_centered) / scale
        grad = 2.0 * (x_centered.T @ residual) / (scale * y.size)
        if edges and lambda_substitution > 0.0:
            for left, right, edge_weight in edges:
                delta = float(weights[left] - weights[right])
                term = 2.0 * float(lambda_substitution) * edge_weight * delta
                grad[left] += term
                grad[right] -= term
        if use_temporal and lambda_temporal > 0.0:
            group_delta = group_matrix @ weights - previous
            grad += 2.0 * float(lambda_temporal) * (group_matrix.T @ group_delta)
        return np.asarray(grad, dtype=float)

    def numerical_gradient(weights: np.ndarray) -> np.ndarray:
        step = 1e-6
        out = np.zeros_like(weights)
        for idx in range(weights.size):
            left = weights.copy()
            right = weights.copy()
            left[idx] -= step
            right[idx] += step
            out[idx] = (objective(right) - objective(left)) / (2.0 * step)
        return out

    initial = np.zeros(len(reduced_keys), dtype=float)
    if use_temporal and previous_asset_weights is not None:
        for reduced_idx, indexes in enumerate(duplicate_groups):
            initial[reduced_idx] = sum(
                float(previous_asset_weights.get(active_keys[index], 0.0))
                for index in indexes
            )
        initial = _project_capped_simplex(initial)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, 1.0)] * len(reduced_keys),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda weights: 1.0 - float(np.sum(weights)),
                "jac": lambda weights: -np.ones_like(weights),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    weights_reduced = np.asarray(result.x, dtype=float)
    primal_violation = max(
        0.0,
        float(-np.min(weights_reduced)),
        float(np.sum(weights_reduced) - 1.0),
    )
    analytic_gradient = gradient(weights_reduced)
    audited_gradient = numerical_gradient(weights_reduced)
    gradient_check_error = float(np.max(np.abs(analytic_gradient - audited_gradient)))
    projected = _project_capped_simplex(weights_reduced - audited_gradient)
    kkt_residual = float(np.max(np.abs(weights_reduced - projected)))
    if (
        not bool(result.success)
        or not np.all(np.isfinite(weights_reduced))
        or primal_violation > WEIGHT_TOLERANCE
        or kkt_residual > KKT_TOLERANCE
        or gradient_check_error > KKT_TOLERANCE
    ):
        return None

    expanded_active = _expand_duplicate_weights(
        weights_reduced, duplicate_groups, len(active_keys)
    )
    full_weights = np.zeros(len(keys), dtype=float)
    full_weights[active_mask] = expanded_active
    active_group_weights = {
        group: float(value)
        for group, value in zip(reporting_groups, group_matrix @ weights_reduced)
    }
    all_reporting_groups = sorted({factor_meta[key].reporting_group for key in keys})
    group_weights = {
        group: float(
            sum(
                full_weights[idx]
                for idx, key in enumerate(keys)
                if factor_meta[key].reporting_group == group
            )
        )
        for group in all_reporting_groups
    }
    fit_residual = y_centered - x_centered @ weights_reduced
    fit_te = (
        float(np.std(fit_residual, ddof=1) * np.sqrt(252.0))
        if fit_residual.size >= 2
        else float("nan")
    )
    return _FitResult(
        weights=full_weights,
        objective=float(result.fun),
        iterations=int(getattr(result, "nit", 0)),
        primal_violation=primal_violation,
        kkt_residual=kkt_residual,
        active_keys=tuple(active_keys),
        dropped_near_zero_vol=dropped,
        reporting_groups=reporting_groups,
        group_weights=group_weights,
        active_group_weights=active_group_weights,
        fit_tracking_error_annualized=fit_te,
        temporal_smoothing_used=use_temporal,
        gradient_check_error=gradient_check_error,
    )


def _oos_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float | int | str | None]:
    valid = [
        row
        for row in rows
        if row.get("target_return") is not None
        and row.get("replication_return") is not None
    ]
    if len(valid) < 2:
        return {
            "sample": "oos_lag1",
            "sample_days": len(valid),
            "tracking_error_annualized": None,
            "residual_drift_annualized": None,
            "geometric_active_return": None,
            "variance_explained": None,
        }
    target = np.array([float(row["target_return"]) for row in valid])
    replicated = np.array([float(row["replication_return"]) for row in valid])
    residual = target - replicated
    target_var = float(np.var(target, ddof=1))
    variance_explained = (
        1.0 - float(np.var(residual, ddof=1)) / target_var
        if target_var > 1e-18
        else None
    )
    target_growth = float(np.prod(1.0 + target))
    replicated_growth = float(np.prod(1.0 + replicated))
    geometric_active = (
        target_growth / replicated_growth - 1.0 if replicated_growth > 0.0 else None
    )
    return {
        "sample": "oos_lag1",
        "sample_days": len(valid),
        "tracking_error_annualized": float(np.std(residual, ddof=1) * np.sqrt(252.0)),
        "residual_drift_annualized": float(np.mean(residual) * 252.0),
        "geometric_active_return": geometric_active,
        "variance_explained": variance_explained,
    }


def _stability_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    distances: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        left = dict(previous.get("group_weights") or {})
        right = dict(current.get("group_weights") or {})
        groups = set(left) | set(right)
        distances.append(
            0.5
            * sum(
                abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
                for key in groups
            )
        )
    return {
        "sample": "rolling_group_weights",
        "transition_count": len(distances),
        "mean_group_weight_distance": (
            float(np.mean(distances)) if distances else None
        ),
        "max_group_weight_distance": max(distances) if distances else None,
    }


def _latest_window_diagnostics(
    *,
    x: np.ndarray,
    y: np.ndarray,
    keys: list[str],
    factor_meta: Mapping[str, ReplicationFactorMeta],
    published_fit: _FitResult,
    previous_asset_weights: Mapping[str, float] | None,
    previous_group_weights: Mapping[str, float] | None,
    lambda_substitution: float,
    lambda_temporal: float,
    extra_solve_budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, bool, bool]:
    budget = max(0, min(int(extra_solve_budget), MAX_EXTRA_SOLVES))
    used = 0
    y_centered = y - float(np.mean(y))
    x_centered = x - np.mean(x, axis=0)
    published = np.asarray(published_fit.weights, dtype=float)
    residual = y_centered - x_centered @ published
    sse_star = float(residual @ residual)
    delta = IDENTIFIABILITY_SSE_RATIO * sse_star
    sse_limit = sse_star + delta
    active = set(published_fit.active_keys)
    bounds = [(0.0, 1.0) if key in active else (0.0, 0.0) for key in keys]
    group_members: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        group_members.setdefault(factor_meta[key].reporting_group, []).append(idx)
    ranked_groups = sorted(
        group_members,
        key=lambda group: (
            -float(published_fit.group_weights.get(group, 0.0)),
            group,
        ),
    )[:MAX_INTERVAL_GROUPS]
    intervals: dict[str, dict[str, float]] = {}

    def sse_constraint(weights: np.ndarray) -> float:
        fit_residual = y_centered - x_centered @ weights
        return sse_limit - float(fit_residual @ fit_residual)

    def sse_constraint_jac(weights: np.ndarray) -> np.ndarray:
        fit_residual = y_centered - x_centered @ weights
        return 2.0 * (x_centered.T @ fit_residual)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda weights: 1.0 - float(np.sum(weights)),
            "jac": lambda weights: -np.ones_like(weights),
        },
        {
            "type": "ineq",
            "fun": sse_constraint,
            "jac": sse_constraint_jac,
        },
    ]
    interval_degraded = False
    for group in ranked_groups:
        if used + 2 > budget:
            interval_degraded = True
            break
        indexes = group_members[group]
        direction = np.zeros(len(keys), dtype=float)
        direction[indexes] = 1.0
        values: list[float] = []
        for sign in (1.0, -1.0):
            result = minimize(
                lambda weights, s=sign, d=direction: s * float(d @ weights),
                published,
                method="SLSQP",
                jac=lambda weights, s=sign, d=direction: s * d,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 300, "disp": False},
            )
            used += 1
            if not bool(result.success) or sse_constraint(result.x) < -1e-8:
                values = []
                break
            values.append(float(direction @ result.x))
        if len(values) != 2:
            interval_degraded = True
            continue
        lower, upper = values[0], values[1]
        published_group = float(published_fit.group_weights.get(group, 0.0))
        if lower > published_group + 1e-8 or upper < published_group - 1e-8:
            interval_degraded = True
            continue
        intervals[group] = {
            "lower": lower,
            "published": published_group,
            "upper": upper,
        }

    sensitivities: list[dict[str, Any]] = []
    sensitivity_degraded = False
    substitution_members: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        substitution_members.setdefault(factor_meta[key].substitution_group, []).append(
            idx
        )
    sensitivity_units: list[tuple[str, str, str, list[str]]] = []
    duplicate_members: set[str] = set()
    for indexes in substitution_members.values():
        for position, idx in enumerate(indexes):
            if any(
                float(np.max(np.abs(x[:, idx] - x[:, previous_idx]))) <= 1e-12
                for previous_idx in indexes[:position]
            ):
                key = keys[idx]
                duplicate_members.add(key)
                sensitivity_units.append(("exact_duplicate", "factor", key, [key]))
    for group, indexes in sorted(substitution_members.items()):
        group_keys = [keys[idx] for idx in indexes]
        nonduplicate_keys = [key for key in group_keys if key not in duplicate_members]
        if len(group_keys) > 1:
            for key in nonduplicate_keys:
                sensitivity_units.append(
                    (
                        "within_group_addition",
                        "factor",
                        key,
                        [key],
                    )
                )
            for kept_key in nonduplicate_keys:
                removed = [key for key in group_keys if key != kept_key]
                if removed:
                    sensitivity_units.append(
                        (
                            "within_group_replacement",
                            "substitution_group",
                            group,
                            removed,
                        )
                    )
    if len(keys) <= MAX_SENSITIVITY_FACTORS:
        for key in keys:
            if len(substitution_members[factor_meta[key].substitution_group]) == 1:
                sensitivity_units.append(("cross_group_addition", "factor", key, [key]))
    else:
        for group, indexes in sorted(group_members.items()):
            sensitivity_units.append(
                (
                    "cross_group_addition",
                    "reporting_group",
                    group,
                    [keys[idx] for idx in indexes],
                )
            )
    unique_units: list[tuple[str, str, str, list[str]]] = []
    seen_units: set[tuple[str, tuple[str, ...]]] = set()
    for unit in sensitivity_units:
        signature = (unit[0], tuple(sorted(unit[3])))
        if signature in seen_units:
            continue
        seen_units.add(signature)
        unique_units.append(unit)
    scenario_order = (
        "exact_duplicate",
        "within_group_replacement",
        "within_group_addition",
        "cross_group_addition",
    )
    buckets = {
        scenario: [unit for unit in unique_units if unit[0] == scenario]
        for scenario in scenario_order
    }
    sensitivity_units = []
    for position in range(max((len(items) for items in buckets.values()), default=0)):
        for scenario in scenario_order:
            if position < len(buckets[scenario]):
                sensitivity_units.append(buckets[scenario][position])
    for scenario, unit_type, unit_name, removed_keys in sensitivity_units[
        :MAX_SENSITIVITY_FACTORS
    ]:
        if used >= budget:
            sensitivity_degraded = True
            break
        removed_set = set(removed_keys)
        keep = [idx for idx, key in enumerate(keys) if key not in removed_set]
        if len(keep) < 2:
            sensitivities.append(
                {
                    "removed_unit_type": unit_type,
                    "removed_unit": unit_name,
                    "removed_factors": removed_keys,
                    "scenario": scenario,
                    "status": "insufficient_remaining_factors",
                }
            )
            continue
        variant_keys = [keys[idx] for idx in keep]
        variant_fit = _fit_window(
            x=x[:, keep],
            y=y,
            keys=variant_keys,
            factor_meta=factor_meta,
            previous_asset_weights=previous_asset_weights,
            previous_group_weights=previous_group_weights,
            lambda_substitution=lambda_substitution,
            lambda_temporal=lambda_temporal,
        )
        used += 1
        if variant_fit is None:
            sensitivities.append(
                {
                    "removed_unit_type": unit_type,
                    "removed_unit": unit_name,
                    "removed_factors": removed_keys,
                    "scenario": scenario,
                    "status": "solver_failed",
                }
            )
            continue
        groups = set(published_fit.group_weights) | set(variant_fit.group_weights)
        distance = 0.5 * sum(
            abs(
                float(published_fit.group_weights.get(group, 0.0))
                - float(variant_fit.group_weights.get(group, 0.0))
            )
            for group in groups
        )
        te_improvement = float(
            variant_fit.fit_tracking_error_annualized
            - published_fit.fit_tracking_error_annualized
        )
        unstable = (
            te_improvement < STABILITY_TE_IMPROVEMENT_THRESHOLD
            and distance > STABILITY_GROUP_DISTANCE_THRESHOLD
        )
        sensitivities.append(
            {
                "removed_unit_type": unit_type,
                "removed_unit": unit_name,
                "removed_factors": removed_keys,
                "scenario": scenario,
                "removed_factor": (unit_name if unit_type == "factor" else None),
                "status": "unstable_redundancy" if unstable else "ok",
                "group_weight_distance": float(distance),
                "base_fit_tracking_error_annualized": (
                    published_fit.fit_tracking_error_annualized
                ),
                "variant_fit_tracking_error_annualized": (
                    variant_fit.fit_tracking_error_annualized
                ),
                "fit_tracking_error_improvement": te_improvement,
            }
        )
    if len(sensitivity_units) > MAX_SENSITIVITY_FACTORS:
        sensitivity_degraded = True
    identifiability = {
        "sample": "latest_window_in_sample",
        "sse_star": sse_star,
        "delta": delta,
        "sse_limit": sse_limit,
        "group_weight_intervals": intervals,
        "interval_degraded": interval_degraded,
    }
    return (
        identifiability,
        sensitivities,
        used,
        interval_degraded,
        sensitivity_degraded,
    )


def _coverage_shift(
    target_index: pd.Index,
    factor_returns: pd.DataFrame,
    keys: list[str],
) -> tuple[pd.Timestamp | None, int]:
    common = target_index
    for key in keys:
        common = common.intersection(factor_returns[key].dropna().index)
    common = common.sort_values()
    if common.empty:
        return None, len(target_index)
    start = pd.Timestamp(common[0])
    return start, int(sum(pd.Timestamp(date) < start for date in target_index))


def _coverage_selection(
    *,
    target_index: pd.Index,
    factor_returns: pd.DataFrame,
    keys: list[str],
    max_start_shift: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    full_start, full_shift = _coverage_shift(target_index, factor_returns, keys)
    coverage = {
        "reference_start": (
            pd.Timestamp(target_index[0]).date().isoformat()
            if len(target_index)
            else None
        ),
        "full_factor_start": (
            full_start.date().isoformat() if full_start is not None else None
        ),
        "start_shift_days": full_shift,
        "max_start_shift_days": int(max_start_shift),
    }
    if full_shift <= max_start_shift:
        return keys, [], coverage

    remaining = list(keys)
    culprits: list[str] = []
    current_shift = full_shift
    while current_shift > max_start_shift and len(remaining) > 2:
        candidates: list[tuple[int, pd.Timestamp, str, list[str]]] = []
        for key in remaining:
            candidate_keys = [item for item in remaining if item != key]
            start, shift = _coverage_shift(target_index, factor_returns, candidate_keys)
            key_start = factor_returns[key].dropna().index.min()
            candidates.append(
                (
                    shift,
                    pd.Timestamp(key_start)
                    if key_start is not None
                    else pd.Timestamp.max,
                    key,
                    candidate_keys,
                )
            )
        candidates.sort(key=lambda item: (item[0], -item[1].value, item[2]))
        best_shift, _, removed, candidate_keys = candidates[0]
        culprits.append(removed)
        remaining = candidate_keys
        current_shift = best_shift
    selected_start, selected_shift = _coverage_shift(
        target_index, factor_returns, remaining
    )
    coverage.update(
        {
            "culprit_factors": culprits,
            "selected_factor_start": (
                selected_start.date().isoformat()
                if selected_start is not None
                else None
            ),
            "selected_start_shift_days": selected_shift,
        }
    )
    return remaining, culprits, coverage


def replicate_fund_by_constrained_weights(
    *,
    fund_nav: pd.Series,
    factor_close_df: pd.DataFrame,
    factor_meta: Mapping[str, ReplicationFactorMeta],
    rolling_window: int,
    min_samples: int,
    include_series: bool,
    max_series_points: int,
    lambda_substitution: float = DEFAULT_LAMBDA_SUBSTITUTION,
    lambda_temporal: float = DEFAULT_LAMBDA_TEMPORAL,
    drop_short_history_factors: bool = False,
    max_start_shift: int = DEFAULT_MAX_START_SHIFT,
    compute_latest_diagnostics: bool = False,
    extra_solve_budget: int = 0,
) -> dict[str, Any]:
    """Build a long-only, cash-residual, lag-one replication portfolio."""
    contract = {
        "model_version": MODEL_VERSION,
        "solver_used": "slsqp_constrained_centered",
        "model_parameters": {
            "calibration_status": MODEL_CALIBRATION_STATUS,
            "calibration_dataset": MODEL_CALIBRATION_DATASET,
            "lambda_substitution": float(lambda_substitution),
            "lambda_temporal": float(lambda_temporal),
            "return_scale_floor": RETURN_SCALE_FLOOR,
            "near_zero_vol_abs": NEAR_ZERO_VOL_ABS,
            "near_zero_vol_rel": NEAR_ZERO_VOL_REL,
            "kkt_tolerance": KKT_TOLERANCE,
            "max_start_shift": int(max_start_shift),
            "initialization": "zero_then_previous_solution",
            "identifiability_sse_ratio": IDENTIFIABILITY_SSE_RATIO,
            "max_extra_solves": MAX_EXTRA_SOLVES,
            "max_targets": MAX_TARGETS,
            "max_factors": MAX_FACTORS,
            "max_windows_per_target": MAX_WINDOWS_PER_TARGET,
            "stability_group_distance_threshold": (STABILITY_GROUP_DISTANCE_THRESHOLD),
            "stability_te_improvement_threshold": (STABILITY_TE_IMPROVEMENT_THRESHOLD),
        },
    }
    ret_fund = nav_to_returns(fund_nav)
    ordered_keys = sorted(str(column) for column in factor_close_df.columns)
    missing_meta = [key for key in ordered_keys if key not in factor_meta]
    if missing_meta:
        raise ReplicationConfigError(f"missing factor metadata: {missing_meta}")
    fac_ret = factor_close_df.loc[:, ordered_keys].apply(nav_to_returns)
    retained_keys, short_history_factors, coverage = _coverage_selection(
        target_index=ret_fund.index,
        factor_returns=fac_ret,
        keys=ordered_keys,
        max_start_shift=int(max_start_shift),
    )
    if (
        int(coverage.get("start_shift_days") or 0) > int(max_start_shift)
        and not drop_short_history_factors
    ):
        return {
            **contract,
            "status": "coverage_conflict",
            "sample_days": 0,
            "effective_start": None,
            "effective_end": None,
            "coverage": coverage,
            "dropped_short_history_factors": [],
            "warnings": ["coverage_conflict"],
        }
    if drop_short_history_factors and int(
        coverage.get("selected_start_shift_days") or 0
    ) > int(max_start_shift):
        return {
            **contract,
            "status": "coverage_conflict",
            "sample_days": 0,
            "effective_start": None,
            "effective_end": None,
            "coverage": coverage,
            "dropped_short_history_factors": [],
            "warnings": ["coverage_conflict_cannot_restore"],
        }
    if drop_short_history_factors and short_history_factors:
        ordered_keys = retained_keys
        fac_ret = fac_ret.loc[:, ordered_keys]
    else:
        short_history_factors = []
    common = ret_fund.dropna().index
    for key in ordered_keys:
        common = common.intersection(fac_ret[key].dropna().index)
    common = common.sort_values()
    required = max(
        int(min_samples),
        int(rolling_window),
        len(ordered_keys) + 6,
    )
    if len(common) < required:
        return {
            **contract,
            "status": "insufficient_samples",
            "sample_days": int(len(common)),
            "effective_start": None,
            "effective_end": None,
            "warnings": [f"sample days={len(common)} < required days={required}"],
        }

    y_all = ret_fund.reindex(common).to_numpy(dtype=float)
    x_all = fac_ret.reindex(common).to_numpy(dtype=float)
    window = int(rolling_window)
    window_count = len(common) - window + 1
    if window_count > MAX_WINDOWS_PER_TARGET:
        return {
            **contract,
            "status": "computation_budget_exceeded",
            "sample_days": int(len(common)),
            "warnings": [f"window_count={window_count} > max={MAX_WINDOWS_PER_TARGET}"],
        }

    rows: list[dict[str, Any]] = []
    previous_group_weights: dict[str, float] | None = None
    previous_asset_weights: dict[str, float] | None = None
    nav_rep = 1.0
    latest_fit: _FitResult | None = None
    latest_weights: dict[str, float] = {}
    latest_cash = 1.0
    fit_failures = 0
    pending_smooth_break = False
    latest_context: dict[str, Any] | None = None
    for end_idx in range(window - 1, len(common)):
        start_idx = end_idx - window + 1
        previous_asset_for_fit = (
            dict(previous_asset_weights) if previous_asset_weights is not None else None
        )
        previous_group_for_fit = (
            dict(previous_group_weights) if previous_group_weights is not None else None
        )
        fit = _fit_window(
            x=x_all[start_idx : end_idx + 1],
            y=y_all[start_idx : end_idx + 1],
            keys=ordered_keys,
            factor_meta=factor_meta,
            previous_asset_weights=previous_asset_weights,
            previous_group_weights=previous_group_weights,
            lambda_substitution=float(lambda_substitution),
            lambda_temporal=float(lambda_temporal),
        )
        if fit is None:
            previous_group_weights = None
            previous_asset_weights = None
            pending_smooth_break = True
            fit_failures += 1
            continue
        latest_fit = fit
        latest_context = {
            "x": x_all[start_idx : end_idx + 1],
            "y": y_all[start_idx : end_idx + 1],
            "previous_asset_weights": previous_asset_for_fit,
            "previous_group_weights": previous_group_for_fit,
        }
        latest_weights = {
            key: float(weight) for key, weight in zip(ordered_keys, fit.weights)
        }
        latest_cash = float(1.0 - float(np.sum(fit.weights)))
        weight_sum_error = abs(float(np.sum(fit.weights)) + latest_cash - 1.0)
        if latest_cash < -WEIGHT_TOLERANCE or weight_sum_error > WEIGHT_TOLERANCE:
            previous_group_weights = None
            previous_asset_weights = None
            pending_smooth_break = True
            fit_failures += 1
            continue
        smooth_break = pending_smooth_break or (
            previous_group_weights is not None and not fit.temporal_smoothing_used
        )
        pending_smooth_break = False
        previous_group_weights = fit.active_group_weights
        previous_asset_weights = latest_weights.copy()
        effective_idx = end_idx + 1
        effective_date = (
            pd.Timestamp(common[effective_idx]).date().isoformat()
            if effective_idx < len(common)
            else None
        )
        target_return: float | None = None
        replication_return: float | None = None
        residual: float | None = None
        if effective_idx < len(common):
            target_return = float(y_all[effective_idx])
            replication_return = float(x_all[effective_idx] @ fit.weights)
            residual = target_return - replication_return
            nav_rep *= 1.0 + replication_return
        rows.append(
            {
                "estimation_date": pd.Timestamp(common[end_idx]).date().isoformat(),
                "effective_date": effective_date,
                "asset_weights": latest_weights.copy(),
                "cash_weight": latest_cash,
                "group_weights": fit.group_weights,
                "target_return": target_return,
                "replication_return": replication_return,
                "residual": residual,
                "replication_nav": nav_rep if effective_date is not None else None,
                "fit_tracking_error_annualized": fit.fit_tracking_error_annualized,
                "dropped_near_zero_vol": list(fit.dropped_near_zero_vol),
                "smooth_break": smooth_break,
                "solver": {
                    "status": "optimal",
                    "objective": fit.objective,
                    "iterations": fit.iterations,
                    "primal_violation": fit.primal_violation,
                    "kkt_residual": fit.kkt_residual,
                    "gradient_check_error": fit.gradient_check_error,
                },
            }
        )

    if latest_fit is None:
        return {
            **contract,
            "status": "solver_failed",
            "sample_days": int(len(common)),
            "effective_start": None,
            "effective_end": None,
            "warnings": ["no valid rolling solve"],
        }
    metrics = _oos_metrics(rows)
    min_oos = max(20, int(min_samples) // 4)
    warnings: list[str] = []
    if int(metrics["sample_days"] or 0) < min_oos:
        warnings.append("insufficient_oos")
        metrics = {
            "sample": "oos_lag1",
            "sample_days": int(metrics["sample_days"] or 0),
            "tracking_error_annualized": None,
            "residual_drift_annualized": None,
            "geometric_active_return": None,
            "variance_explained": None,
        }
    if fit_failures:
        warnings.append(f"solver_failed_windows={fit_failures}")
    identifiability: dict[str, Any] = {}
    factor_sensitivity: list[dict[str, Any]] = []
    extra_solves_used = 0
    diagnostics_degraded = False
    sensitivity_degraded = False
    if compute_latest_diagnostics and latest_context is not None:
        (
            identifiability,
            factor_sensitivity,
            extra_solves_used,
            interval_degraded,
            sensitivity_degraded,
        ) = _latest_window_diagnostics(
            x=latest_context["x"],
            y=latest_context["y"],
            keys=ordered_keys,
            factor_meta=factor_meta,
            published_fit=latest_fit,
            previous_asset_weights=latest_context["previous_asset_weights"],
            previous_group_weights=latest_context["previous_group_weights"],
            lambda_substitution=float(lambda_substitution),
            lambda_temporal=float(lambda_temporal),
            extra_solve_budget=extra_solve_budget,
        )
        diagnostics_degraded = interval_degraded or sensitivity_degraded
        if diagnostics_degraded:
            warnings.append("latest_diagnostics_degraded")
        if any(
            item.get("status") == "unstable_redundancy" for item in factor_sensitivity
        ):
            warnings.append("factor_sensitivity_unstable")
    series = rows
    if not include_series:
        series = []
    elif max_series_points > 0 and len(series) > max_series_points:
        series = series[-max_series_points:]
    effective_dates = [row["effective_date"] for row in rows if row["effective_date"]]
    return {
        **contract,
        "status": "ok",
        "sample_days": int(len(common)),
        "training_days": window,
        "oos_days": int(metrics["sample_days"] or 0),
        "estimation_windows": len(rows),
        "effective_windows": len(effective_dates),
        "effective_start": effective_dates[0] if effective_dates else None,
        "effective_end": effective_dates[-1] if effective_dates else None,
        "asset_weights": latest_weights,
        "cash_weight": latest_cash,
        "group_weights": latest_fit.group_weights,
        "oos_metrics": metrics,
        "stability": _stability_metrics(rows),
        "identifiability": identifiability,
        "factor_sensitivity": factor_sensitivity,
        "extra_solves_used": extra_solves_used,
        "diagnostics_degraded": diagnostics_degraded,
        "sensitivity_degraded": sensitivity_degraded,
        "diagnostics_computed": bool(
            compute_latest_diagnostics and latest_context is not None
        ),
        "fit_tracking_error_annualized": latest_fit.fit_tracking_error_annualized,
        "dropped_near_zero_vol": list(latest_fit.dropped_near_zero_vol),
        "latest_solver": rows[-1]["solver"],
        "warnings": warnings,
        "coverage": coverage,
        "dropped_short_history_factors": short_history_factors,
        "series": series,
    }
