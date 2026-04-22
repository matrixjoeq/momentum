from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    FuturesPool,
    FuturesResearchGroup,
    FuturesResearchGroupItem,
    FuturesResearchState,
)


@dataclass(frozen=True)
class FuturesGroupData:
    name: str
    codes: list[str]
    is_active: bool


def get_futures_research_state(db: Session) -> FuturesResearchState:
    obj = db.execute(
        select(FuturesResearchState).where(FuturesResearchState.id == 1)
    ).scalar_one_or_none()
    if obj is None:
        # Lazy-init with concurrency safety:
        # concurrent requests may race to create the singleton row (id=1).
        try:
            with db.begin_nested():
                db.add(FuturesResearchState(id=1))
                db.flush()
        except IntegrityError:
            # Another transaction has inserted the row; read it below.
            pass
        obj = db.execute(
            select(FuturesResearchState).where(FuturesResearchState.id == 1)
        ).scalar_one_or_none()
        if obj is None:
            raise RuntimeError("failed to initialize futures_research_state")
    return obj


def upsert_futures_research_state(
    db: Session,
    *,
    start_date: str | None,
    end_date: str | None,
    dynamic_universe: bool,
    quick_range_key: str | None,
) -> FuturesResearchState:
    obj = get_futures_research_state(db)
    obj.start_date = start_date
    obj.end_date = end_date
    obj.dynamic_universe = bool(dynamic_universe)
    obj.quick_range_key = quick_range_key
    db.flush()
    return obj


def get_futures_group(db: Session, *, name: str) -> FuturesResearchGroup | None:
    return db.execute(
        select(FuturesResearchGroup).where(FuturesResearchGroup.name == name)
    ).scalar_one_or_none()


def list_futures_groups(db: Session) -> list[FuturesGroupData]:
    groups = list(
        db.execute(
            select(FuturesResearchGroup).order_by(FuturesResearchGroup.name.asc())
        )
        .scalars()
        .all()
    )
    if not groups:
        return []
    gid_to_codes: dict[int, list[str]] = {}
    rows = list(
        db.execute(
            select(FuturesResearchGroupItem)
            .where(FuturesResearchGroupItem.group_id.in_([g.id for g in groups]))
            .order_by(
                FuturesResearchGroupItem.group_id.asc(),
                FuturesResearchGroupItem.sort_order.asc(),
                FuturesResearchGroupItem.code.asc(),
            )
        )
        .scalars()
        .all()
    )
    for r in rows:
        gid_to_codes.setdefault(int(r.group_id), []).append(str(r.code))
    return [
        FuturesGroupData(
            name=str(g.name),
            codes=list(gid_to_codes.get(int(g.id), [])),
            is_active=bool(g.is_active),
        )
        for g in groups
    ]


def set_active_futures_group(db: Session, *, name: str) -> bool:
    g = get_futures_group(db, name=name)
    if g is None:
        return False
    db.execute(update(FuturesResearchGroup).values(is_active=False))
    g.is_active = True
    db.flush()
    return True


def get_active_futures_group(db: Session) -> FuturesGroupData | None:
    g = db.execute(
        select(FuturesResearchGroup).where(FuturesResearchGroup.is_active.is_(True))
    ).scalar_one_or_none()
    if g is None:
        return None
    rows = list(
        db.execute(
            select(FuturesResearchGroupItem)
            .where(FuturesResearchGroupItem.group_id == g.id)
            .order_by(
                FuturesResearchGroupItem.sort_order.asc(),
                FuturesResearchGroupItem.code.asc(),
            )
        )
        .scalars()
        .all()
    )
    return FuturesGroupData(
        name=str(g.name), codes=[str(x.code) for x in rows], is_active=True
    )


def _valid_futures_codes(db: Session) -> set[str]:
    rows = db.execute(select(FuturesPool.code)).all()
    return {str(r[0]) for r in rows if r and r[0]}


def upsert_futures_group(
    db: Session,
    *,
    name: str,
    codes: list[str],
    set_active: bool,
) -> tuple[FuturesGroupData, list[str]]:
    existing = get_futures_group(db, name=name)
    if existing is None:
        existing = FuturesResearchGroup(name=name, is_active=False)
        db.add(existing)
        db.flush()
    valid = _valid_futures_codes(db)
    input_codes = [str(c).strip() for c in codes if str(c).strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    skipped: list[str] = []
    for c in input_codes:
        if c in seen:
            continue
        seen.add(c)
        if c in valid:
            deduped.append(c)
        else:
            skipped.append(c)
    db.execute(
        delete(FuturesResearchGroupItem).where(
            FuturesResearchGroupItem.group_id == existing.id
        )
    )
    if deduped:
        db.add_all(
            [
                FuturesResearchGroupItem(
                    group_id=int(existing.id), code=code, sort_order=i
                )
                for i, code in enumerate(deduped)
            ]
        )
    if set_active:
        db.execute(update(FuturesResearchGroup).values(is_active=False))
        existing.is_active = True
    db.flush()
    return (
        FuturesGroupData(
            name=str(existing.name),
            codes=list(deduped),
            is_active=bool(existing.is_active),
        ),
        skipped,
    )


def delete_futures_group(db: Session, *, name: str) -> bool:
    g = get_futures_group(db, name=name)
    if g is None:
        return False
    db.execute(
        delete(FuturesResearchGroupItem).where(
            FuturesResearchGroupItem.group_id == g.id
        )
    )
    db.delete(g)
    db.flush()
    return True
