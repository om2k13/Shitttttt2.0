from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from ..db.base import get_session
from ..db.models import Finding

router = APIRouter(prefix="/api", tags=["reports"])

@router.get("/reports/{job_id}")
async def get_report(job_id: str, session=Depends(get_session)):
    result = await session.execute(select(Finding).where(Finding.job_id==job_id))
    items = result.scalars().all()
    if items is None:
        raise HTTPException(404, "report not found")
    # aggregate
    summary = {"total": len(items), "by_tool": {}, "by_severity": {}}
    for f in items:
        summary["by_tool"][f.tool] = summary["by_tool"].get(f.tool, 0) + 1
        summary["by_severity"][f.severity] = summary["by_severity"].get(f.severity, 0) + 1
    return {"summary": summary, "findings": items}
