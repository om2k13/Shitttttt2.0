from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from ..db.base import get_session
from ..db.models import Job, Finding
from ..core.vcs import clone_repo
from ..review.autofix.safe_fixes import run_python_formatters, run_js_formatters

router = APIRouter(prefix="/api", tags=["actions"])

@router.post("/actions/apply-fix")
async def apply_fix(payload: dict, session=Depends(get_session)):
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id required")
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    repo = clone_repo(job_id + "-fix", job.repo_url, job.branch)
    py = run_python_formatters(repo)
    js = run_js_formatters(repo)
    return {"applied": py + js}
