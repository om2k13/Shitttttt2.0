from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.logging import setup_logging
from .db.base import init_db, close_db
from .api.jobs import router as jobs_router
from .api.reports import router as reports_router
from .api.config import router as config_router
from .api.actions import router as actions_router

setup_logging()
app = FastAPI(title="Code Review Agent (Local)")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def _startup():
    await init_db()

@app.on_event("shutdown")
async def _shutdown():
    await close_db()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

app.include_router(jobs_router)
app.include_router(reports_router)
app.include_router(config_router)
app.include_router(actions_router)
