from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .models import EtfPool, EtfResearchGroup, EtfResearchGroupItem


@dataclass(frozen=True)
class EtfGroupData:
    name: str
    codes: list[str]
    is_active: bool


def get_etf_group(db: Session, *, name: str) -> EtfResearchGroup | None:
    return db.execute(
        select(EtfResearchGroup).where(EtfResearchGroup.name == str(name))
    ).scalar_one_or_none()


def list_etf_groups(db: Session) -> list[EtfGroupData]:
    groups = list(
        db.execute(select(EtfResearchGroup).order_by(EtfResearchGroup.name.asc()))
        .scalars()
        .all()
    )
    if not groups:
        return []
    gid_to_codes: dict[int, list[str]] = {}
    rows = list(
        db.execute(
            select(EtfResearchGroupItem)
            .where(EtfResearchGroupItem.group_id.in_([g.id for g in groups]))
            .order_by(
                EtfResearchGroupItem.group_id.asc(),
                EtfResearchGroupItem.sort_order.asc(),
                EtfResearchGroupItem.code.asc(),
            )
        )
        .scalars()
        .all()
    )
    for r in rows:
        gid_to_codes.setdefault(int(r.group_id), []).append(str(r.code))
    return [
        EtfGroupData(
            name=str(g.name),
            codes=list(gid_to_codes.get(int(g.id), [])),
            is_active=bool(g.is_active),
        )
        for g in groups
    ]


def set_active_etf_group(db: Session, *, name: str) -> bool:
    g = get_etf_group(db, name=name)
    if g is None:
        return False
    db.execute(update(EtfResearchGroup).values(is_active=False))
    g.is_active = True
    db.flush()
    return True


def get_active_etf_group(db: Session) -> EtfGroupData | None:
    g = db.execute(
        select(EtfResearchGroup).where(EtfResearchGroup.is_active.is_(True))
    ).scalar_one_or_none()
    if g is None:
        return None
    rows = list(
        db.execute(
            select(EtfResearchGroupItem)
            .where(EtfResearchGroupItem.group_id == g.id)
            .order_by(
                EtfResearchGroupItem.sort_order.asc(),
                EtfResearchGroupItem.code.asc(),
            )
        )
        .scalars()
        .all()
    )
    return EtfGroupData(name=str(g.name), codes=[str(x.code) for x in rows], is_active=True)


def _valid_etf_codes(db: Session) -> set[str]:
    rows = db.execute(select(EtfPool.code)).all()
    return {str(r[0]) for r in rows if r and r[0]}


def upsert_etf_group(
    db: Session,
    *,
    name: str,
    codes: list[str],
    set_active: bool,
) -> tuple[EtfGroupData, list[str]]:
    existing = get_etf_group(db, name=name)
    if existing is None:
        existing = EtfResearchGroup(name=name, is_active=False)
        db.add(existing)
        db.flush()
    valid = _valid_etf_codes(db)
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
    db.execute(delete(EtfResearchGroupItem).where(EtfResearchGroupItem.group_id == existing.id))
    if deduped:
        db.add_all(
            [
                EtfResearchGroupItem(group_id=int(existing.id), code=code, sort_order=i)
                for i, code in enumerate(deduped)
            ]
        )
    if set_active:
        db.execute(update(EtfResearchGroup).values(is_active=False))
        existing.is_active = True
    db.flush()
    return (
        EtfGroupData(
            name=str(existing.name), codes=list(deduped), is_active=bool(existing.is_active)
        ),
        skipped,
    )


def delete_etf_group(db: Session, *, name: str) -> bool:
    g = get_etf_group(db, name=name)
    if g is None:
        return False
    db.execute(delete(EtfResearchGroupItem).where(EtfResearchGroupItem.group_id == g.id))
    db.delete(g)
    db.flush()
    return True
