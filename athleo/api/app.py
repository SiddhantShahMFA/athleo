from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI

from athleo.api.routes import router as jobs_router
from athleo.jobs import JobManager
from athleo.storage import JobStorage


def default_storage_root() -> Path:
    configured = os.environ.get("ATHLEO_STORAGE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".athleo_jobs"


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    manager = job_manager or JobManager(storage=JobStorage(default_storage_root()))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(title="Athleo Backend", version="1.0.0", lifespan=lifespan)
    app.state.job_manager = manager
    app.include_router(jobs_router)

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["system"])
    def readyz() -> dict[str, str]:
        storage_root = manager.storage.root_dir
        status = "ok" if storage_root.exists() and storage_root.is_dir() else "not_ready"
        return {"status": status}

    return app


app = create_app()
