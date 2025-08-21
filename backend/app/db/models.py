from datetime import datetime
from typing import Optional
from enum import Enum
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import DateTime, func

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    repo_url: str
    branch: Optional[str] = None
    pr_number: Optional[int] = None
    status: str = "queued"
    current_stage: Optional[str] = None
    progress: int = 0
    created_at: datetime = Field(sa_column=Column(DateTime(timezone=True), server_default=func.now()))
    updated_at: datetime = Field(sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()))

class Finding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str
    tool: str
    severity: Severity
    file: Optional[str] = None
    line: Optional[int] = None
    rule_id: Optional[str] = None
    message: str
    remediation: Optional[str] = None
    autofixable: bool = False

class Config(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    json_value: str = Field(default="{}")  # Store as JSON string
