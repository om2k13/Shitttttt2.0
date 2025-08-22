from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum
from sqlmodel import SQLModel, Field

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    ORG_ADMIN = "org_admin"

class User(SQLModel, table=True):
    """User model for multi-user support"""
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    full_name: Optional[str] = None
    role: UserRole = Field(default=UserRole.USER)
    organization: Optional[str] = None
    github_username: Optional[str] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None

class UserToken(SQLModel, table=True):
    """Secure storage of user GitHub tokens"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    token_hash: str = Field(index=True)  # Encrypted/hashed token
    token_name: str = Field(default="Code Review Agent Token")
    scopes: str = Field(default="[]")  # JSON string of GitHub scopes
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Organization(SQLModel, table=True):
    """Organization model for team management"""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    github_org: Optional[str] = None
    org_token_hash: Optional[str] = None  # Encrypted org-wide token
    org_token_scopes: str = Field(default="[]")
    settings: str = Field(default="{}")  # JSON string of org settings
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Job(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")  # Who created the job
    repo_url: str
    status: JobStatus = Field(default=JobStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    findings_count: int = Field(default=0)
    severity_breakdown: str = Field(default="{}")  # JSON string of severity counts
    tools_used: str = Field(default="[]")  # JSON string of tools used
    pr_number: Optional[int] = None  # For PR-specific analysis
    base_branch: Optional[str] = None  # For PR analysis
    head_branch: Optional[str] = None  # For PR analysis
    is_pr_analysis: bool = Field(default=False)  # Flag for PR vs full repo analysis
    organization_id: Optional[int] = Field(foreign_key="organization.id")  # Which org this belongs to
    current_stage: Optional[str] = Field(default="queued")  # Current stage of the review process
    progress: int = Field(default=0)  # Progress percentage (0-100)
    branch: Optional[str] = Field(default="main")  # Branch to analyze (for backward compatibility)
    results: Optional[str] = Field(default="{}")  # JSON string of analysis results

class Finding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id")
    tool: str
    severity: Severity
    file: str
    line: Optional[int] = None
    rule_id: str
    message: str
    remediation: Optional[str] = None
    autofixable: bool = Field(default=False)
    vulnerability_type: Optional[str] = None  # For OWASP categorization
    code_snippet: Optional[str] = None  # The actual code that triggered the finding
    pr_context: Optional[str] = Field(default="{}")  # JSON string of PR context
    risk_score: Optional[int] = Field(default=None)  # 0-10 risk score
    merge_blocking: bool = Field(default=False)  # Whether this blocks PR merge
    test_coverage: Optional[str] = None  # Test coverage information
    breaking_change: bool = Field(default=False)  # Whether this represents a breaking change
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Config(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True)
    value: str
    json_value: str = Field(default="{}")  # JSON string for complex config
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Report(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id")
    report_type: str  # "full_repo", "pr_diff", "security_analysis", "test_plan"
    content: str  # JSON string of report content
    summary: str  # Human-readable summary
    risk_score: Optional[int] = None  # Overall risk score for the report
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_metadata: str = Field(default="{}")  # JSON string for additional metadata
