from pathlib import Path
import yaml
from fastapi import APIRouter, Depends
from sqlmodel import select
from ..db.base import get_session
from ..db.models import Config

router = APIRouter(prefix="/api", tags=["config"])

@router.get("/config")
async def get_config(session=Depends(get_session)):
    result = await session.execute(select(Config).limit(1))
    row = result.scalar_one_or_none()
    if row: 
        return row.json_value
    p = Path(".codereview-agent.yml")
    return yaml.safe_load(p.read_text()) if p.exists() else {"version": 1}

@router.get("/config/ml")
async def get_ml_config():
    """Get ML configuration"""
    return {
        "enabled": True,
        "models": ["random_forest", "linear_regression"],
        "auto_training": True,
        "confidence_threshold": 0.8
    }

@router.get("/config/compliance")
async def get_compliance_config():
    """Get compliance configuration"""
    return {
        "standards": ["SOC2", "PCI_DSS", "GDPR", "HIPAA"],
        "auto_scanning": True,
        "reporting": "detailed"
    }

@router.get("/config/business")
async def get_business_config():
    """Get business configuration"""
    return {
        "roi_tracking": True,
        "cost_analysis": True,
        "business_impact": True
    }

@router.get("/config/security")
async def get_security_config():
    """Get security configuration"""
    return {
        "owasp_top10": True,
        "dependency_scanning": True,
        "secret_detection": True
    }

@router.get("/config/notifications")
async def get_notifications_config():
    """Get notifications configuration"""
    return {
        "email": True,
        "slack": False,
        "webhooks": True,
        "critical_findings": True
    }
