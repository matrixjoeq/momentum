from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from .api.routes import router as api_router
from .api.deps import init_app_state
from .scheduler import start_auto_sync, stop_auto_sync
from .settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        init_app_state(fastapi_app)
        start_auto_sync(fastapi_app)
        yield
        await stop_auto_sync(fastapi_app)

    fastapi_app = FastAPI(title="ETF Momentum - Pool & Data Service", version="0.1.0", lifespan=lifespan)

    @fastapi_app.get("/")
    def index():
        _ = get_settings()  # ensure data dirs exist
        path = Path(__file__).resolve().parent / "web" / "index.html"
        # In some cloud packaging/install setups, static html may not be shipped.
        # Avoid 500s for root health checks; fallback to docs.
        if not path.exists():
            return RedirectResponse(url="/docs")
        return FileResponse(path)

    @fastapi_app.get("/research")
    def research():
        _ = get_settings()
        path = Path(__file__).resolve().parent / "web" / "research.html"
        if not path.exists():
            return RedirectResponse(url="/docs")
        return FileResponse(path)

    @fastapi_app.get("/research/gold-gvx")
    def research_gold_gvx():
        _ = get_settings()
        path = Path(__file__).resolve().parent / "web" / "research_gold_gvx.html"
        if not path.exists():
            return RedirectResponse(url="/docs")
        return FileResponse(path)

    @fastapi_app.get("/research/nasdaq-vix")
    def research_nasdaq_vix():
        _ = get_settings()
        path = Path(__file__).resolve().parent / "web" / "research_nasdaq_vix.html"
        if not path.exists():
            return RedirectResponse(url="/docs")
        return FileResponse(path)

    @fastapi_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    fastapi_app.include_router(api_router, prefix="/api")
    return fastapi_app


app = create_app()

