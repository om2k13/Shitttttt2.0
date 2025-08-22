from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import select
from typing import List, Dict, Optional
from ..db.base import get_session
from ..db.models import Job
from ..core import settings
from ..core.utils import new_id
from ..review.pipeline import run_review
from ..core.vcs import cleanup_all_repos, get_workspace_stats, cleanup_old_workspaces

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

    job = Job(id=new_id(), repo_url=repo_url, base_branch=branch, pr_number=pr_number, status="queued")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    background.add_task(run_review, job.id)
    return {"job_id": job.id, "status": job.status}

@router.get("/jobs/workspace-stats")
async def get_workspaces_statistics():
    """Get statistics about current workspaces"""
    stats = get_workspace_stats()
    return stats

@router.get("/jobs/current-repo")
async def get_current_repository_info():
    """Get information about the current active repository"""
    from ..core.vcs import get_current_repo_info
    current_repo = get_current_repo_info()
    return current_repo

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, session=Depends(get_session)):
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "job not found")
    return job

@router.post("/jobs/cleanup-workspaces")
async def cleanup_workspaces(max_age_hours: int = 1):
    """Clean up old workspace directories"""
    cleanup_old_workspaces(max_age_hours=max_age_hours)
    return {"message": f"Workspace cleanup completed (max age: {max_age_hours} hours)"}

@router.post("/jobs/cleanup-all-repos")
async def cleanup_all_repositories():
    """Clean up all repositories in .workspaces directory"""
    result = cleanup_all_repos()
    return result

@router.post("/jobs/cleanup-after-review/{job_id}")
async def cleanup_repository_after_review(job_id: str):
    """Clean up a specific repository after review completion"""
    from ..core.vcs import cleanup_repo_after_review
    cleanup_repo_after_review(job_id)
    return {"message": f"Repository {job_id} cleaned up after review"}

@router.post("/jobs/cleanup-current-repo")
async def cleanup_current_repository():
    """Clean up the current active repository"""
    from ..core.vcs import cleanup_current_repo
    result = cleanup_current_repo()
    return result

@router.post("/jobs/cleanup-tracking-files")
async def cleanup_tracking_files():
    """Clean up tracking files (.repo_tracker.json, .current_repo.json)"""
    from ..core.vcs import cleanup_tracking_files
    result = cleanup_tracking_files()
    return result
