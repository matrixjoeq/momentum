from __future__ import annotations

from sqlalchemy.engine import Engine

from .base import Base
from . import models as _models  # noqa: F401


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


