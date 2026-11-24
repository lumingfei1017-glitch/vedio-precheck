from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import AUDIT_FEED_ENABLED, AUDIT_FEED_SYNC_HOURS, STATIC_DIR
from .db import init_db
from .jobs import shutdown_worker, start_worker


def _sync_audit_feed() -> None:
    from .audit_feed import needs_sync, sync_audits

    try:
        if needs_sync(AUDIT_FEED_SYNC_HOURS):
            sync_audits()
    except Exception:  # noqa: BLE001
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    start_worker()
    if AUDIT_FEED_ENABLED:
        threading.Thread(target=_sync_audit_feed, daemon=True).start()
    yield
    shutdown_worker()


app = FastAPI(title="video-precheck", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_local_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
