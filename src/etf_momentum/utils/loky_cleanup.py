from __future__ import annotations

import logging


def cleanup_loky_executor(logger: logging.Logger | None = None) -> bool:
    """
    Best-effort cleanup for joblib loky reusable executors.

    This avoids repeated "leaked semaphore objects" warnings on process shutdown
    when long-running or interrupted parallel tasks leave loky resources behind.
    """
    try:
        from joblib.externals.loky.reusable_executor import get_reusable_executor
    except ImportError:
        return False

    try:
        ex = get_reusable_executor()
        ex.shutdown(wait=True, kill_workers=True)
        return True
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        PermissionError,
        OSError,
    ) as exc:  # pragma: no cover - best effort cleanup
        if logger is not None:
            logger.debug("loky cleanup failed: %s", exc)
        return False
