from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select

from etf_momentum.db.futures_research_repo import get_futures_research_state
from etf_momentum.db.init_db import init_db
from etf_momentum.db.models import FuturesResearchState
from etf_momentum.db.session import make_session_factory, make_sqlite_engine

# pylint: disable=not-callable


def test_futures_research_state_concurrent_init_singleton(tmp_path) -> None:
    """
    Regression test for concurrent lazy-init of futures_research_state(id=1).

    Multiple sessions initialize state at the same time; we must end with:
    - no IntegrityError leaks
    - exactly one row in futures_research_state
    """

    db_path = tmp_path / "futures-research-race.db"
    engine = make_sqlite_engine(f"sqlite+pysqlite:///{db_path}")
    init_db(engine)
    sf = make_session_factory(engine)

    workers = 12
    barrier = threading.Barrier(workers)

    def _worker() -> int:
        with sf() as db:
            barrier.wait(timeout=10)
            st = get_futures_research_state(db)
            db.commit()
            return int(st.id)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ids = list(ex.map(lambda _: _worker(), range(workers)))

    assert ids == [1] * workers

    with sf() as db:
        cnt = db.execute(
            select(func.count()).select_from(FuturesResearchState)
        ).scalar_one()
        assert int(cnt) == 1
