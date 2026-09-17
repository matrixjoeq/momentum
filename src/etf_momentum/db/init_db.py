from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .base import Base
from . import models as _models  # noqa: F401
from .off_fund_regression_repo import ensure_system_off_fund_factor_configs
from .schema import ensure_runtime_schema


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    with Session(engine) as db:
        try:
            ensure_system_off_fund_factor_configs(db)
            db.commit()
        except IntegrityError:
            # A concurrent starter may have inserted the same immutable
            # template/default row after this transaction's preflight query.
            db.rollback()
            ensure_system_off_fund_factor_configs(db)
            db.commit()
