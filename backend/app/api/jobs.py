from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import select
from ..db.base import get_session
from ..db.models import Job
from ..core import settings
from ..core.utils import new_id
from ..review.pipeline import run_review

router = APIRouter(prefix="/api", tags=["jobs"])

@router.get("/jobs")
async def list_jobs(page: int = 1, size: int = 20, session=Depends(get_session)):
    from sqlalchemy import select, func
    total_result = await session.execute(select(func.count(Job.id)))
    total_count = total_result.scalar()
    items_result = await session.execute(select(Job).order_by(Job.created_at.desc()).offset((page-1)*size).limit(size))
    items = items_result.scalars().all()
    return {"items": items, "page": page, "size": size, "total": total_count}

@router.post("/jobs")
async def create_job(payload: dict, background: BackgroundTasks, session=Depends(get_session)):
    repo_url = payload.get("repo_url")
    branch = payload.get("branch")
    pr_number = payload.get("pr_number")
    if not repo_url:
        raise HTTPException(400, "repo_url required")

    job = Job(id=new_id(), repo_url=repo_url, branch=branch, pr_number=pr_number, status="queued", progress=0, current_stage="queued")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    background.add_task(run_review, job.id)
    return {"job_id": job.id, "status": job.status}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, session=Depends(get_session)):
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "job not found")
    return job
