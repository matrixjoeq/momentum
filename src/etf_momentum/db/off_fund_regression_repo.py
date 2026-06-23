from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .models import OffFundRegressionFactorConfig


@dataclass(frozen=True)
class OffFundRegressionFactorConfigData:
    name: str
    is_active: bool
    benchmark_profile: str
    benchmark_factors: list[dict]


_DEFAULT_CONFIG_NAME = "默认A股模板"
_DEFAULT_PROFILE = "cn_stock_core"


def _decode_factors(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(v, list):
        return []
    out: list[dict] = []
    for x in v:
        if not isinstance(x, dict):
            continue
        key = str(x.get("key") or "").strip().upper()
        aliases = [
            str(a).strip()
            for a in list(x.get("aliases") or [])
            if str(a).strip()
        ]
        if not key or not aliases:
            continue
        out.append(
            {
                "key": key,
                "label": str(x.get("label") or key).strip() or key,
                "aliases": aliases,
            }
        )
    return out


def _encode_factors(factors: list[dict] | None) -> str | None:
    if not factors:
        return None
    norm = _decode_factors(json.dumps(factors, ensure_ascii=False))
    if not norm:
        return None
    return json.dumps(norm, ensure_ascii=False, separators=(",", ":"))


def _to_data(x: OffFundRegressionFactorConfig) -> OffFundRegressionFactorConfigData:
    return OffFundRegressionFactorConfigData(
        name=str(x.name),
        is_active=bool(x.is_active),
        benchmark_profile=str(x.benchmark_profile or _DEFAULT_PROFILE),
        benchmark_factors=_decode_factors(x.benchmark_factors_json),
    )


def ensure_default_off_fund_factor_config(
    db: Session,
) -> OffFundRegressionFactorConfigData:
    rows = list(
        db.execute(
            select(OffFundRegressionFactorConfig).order_by(
                OffFundRegressionFactorConfig.name.asc()
            )
        )
        .scalars()
        .all()
    )
    if rows:
        if any(bool(x.is_active) for x in rows):
            active = next(x for x in rows if bool(x.is_active))
            return _to_data(active)
        rows[0].is_active = True
        db.flush()
        return _to_data(rows[0])
    obj = OffFundRegressionFactorConfig(
        name=_DEFAULT_CONFIG_NAME,
        is_active=True,
        benchmark_profile=_DEFAULT_PROFILE,
        benchmark_factors_json=None,
    )
    db.add(obj)
    db.flush()
    return _to_data(obj)


def list_off_fund_factor_configs(db: Session) -> list[OffFundRegressionFactorConfigData]:
    ensure_default_off_fund_factor_config(db)
    rows = list(
        db.execute(
            select(OffFundRegressionFactorConfig).order_by(
                OffFundRegressionFactorConfig.name.asc()
            )
        )
        .scalars()
        .all()
    )
    return [_to_data(x) for x in rows]


def get_off_fund_factor_config(
    db: Session, *, name: str
) -> OffFundRegressionFactorConfig | None:
    return db.execute(
        select(OffFundRegressionFactorConfig).where(
            OffFundRegressionFactorConfig.name == str(name)
        )
    ).scalar_one_or_none()


def set_active_off_fund_factor_config(db: Session, *, name: str) -> bool:
    obj = get_off_fund_factor_config(db, name=name)
    if obj is None:
        return False
    db.execute(update(OffFundRegressionFactorConfig).values(is_active=False))
    obj.is_active = True
    db.flush()
    return True


def upsert_off_fund_factor_config(
    db: Session,
    *,
    name: str,
    benchmark_profile: str,
    benchmark_factors: list[dict] | None,
    set_active: bool,
) -> OffFundRegressionFactorConfigData:
    obj = get_off_fund_factor_config(db, name=name)
    if obj is None:
        obj = OffFundRegressionFactorConfig(name=str(name), is_active=False)
        db.add(obj)
        db.flush()
    obj.benchmark_profile = str(benchmark_profile or _DEFAULT_PROFILE)
    obj.benchmark_factors_json = _encode_factors(benchmark_factors)
    if set_active:
        db.execute(update(OffFundRegressionFactorConfig).values(is_active=False))
        obj.is_active = True
    db.flush()
    return _to_data(obj)


def delete_off_fund_factor_config(db: Session, *, name: str) -> bool:
    obj = get_off_fund_factor_config(db, name=name)
    if obj is None:
        return False
    was_active = bool(obj.is_active)
    db.delete(obj)
    db.flush()
    if was_active:
        rows = list(
            db.execute(
                select(OffFundRegressionFactorConfig).order_by(
                    OffFundRegressionFactorConfig.name.asc()
                )
            )
            .scalars()
            .all()
        )
        if rows:
            rows[0].is_active = True
            db.flush()
    return True
