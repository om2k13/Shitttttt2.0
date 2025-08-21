from pathlib import Path
import yaml
from fastapi import APIRouter, Depends
from sqlmodel import select
from ..db.base import get_session
from ..db.models import Config

router = APIRouter(prefix="/api", tags=["config"])

@router.get("/config")
async def get_config(session=Depends(get_session)):
    row = (await session.exec(select(Config).limit(1))).first()
    if row: 
        return row.json_value
    p = Path(".codereview-agent.yml")
    return yaml.safe_load(p.read_text()) if p.exists() else {"version": 1}
