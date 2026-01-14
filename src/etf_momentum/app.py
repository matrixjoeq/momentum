from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .api.routes import router as api_router
from .api.deps import init_app_state
from .settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # pylint: disable=redefined-outer-name
        init_app_state(app)
        yield

    app = FastAPI(title="ETF Momentum - Pool & Data Service", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    def index():
        _ = get_settings()  # ensure data dirs exist
        path = Path(__file__).resolve().parent / "web" / "index.html"
        return FileResponse(path)

    @app.get("/research")
    def research():
        _ = get_settings()
        path = Path(__file__).resolve().parent / "web" / "research.html"
        return FileResponse(path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router, prefix="/api")
    return app


app = create_app()

