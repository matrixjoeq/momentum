from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from etf_momentum.db.seed import DEFAULT_POLICIES, ensure_default_policies  # pylint: disable=import-error
from etf_momentum.db.models import ValidationPolicy  # pylint: disable=import-error


def test_seed_default_policies(session_factory: sessionmaker) -> None:
    with session_factory() as db:
        ensure_default_policies(db)
        db.commit()

    with session_factory() as db:
        names = {p.name for p in db.query(ValidationPolicy).all()}
        assert {p["name"] for p in DEFAULT_POLICIES}.issubset(names)


def test_seed_updates_existing_policy_fields(session_factory: sessionmaker) -> None:
    # Create an old policy with a different max_gap_days and ensure it gets updated.
    with session_factory() as db:
        db.add(
            ValidationPolicy(
                name="cn_stock_etf_10",
                description="old",
                max_abs_return=0.12,
                max_hl_spread=0.30,
                max_gap_days=10,
            )
        )
        db.commit()

    with session_factory() as db:
        ensure_default_policies(db)
        db.commit()

    with session_factory() as db:
        p = db.query(ValidationPolicy).filter(ValidationPolicy.name == "cn_stock_etf_10").one()
        assert p.max_gap_days == 12

