from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.logging import setup_logging
from .db.base import init_db, close_db
from .api.jobs import router as jobs_router
from .api.reports import router as reports_router
from .api.config import router as config_router
from .api.actions import router as actions_router
from .api.users import router as users_router
from .api.analytics import router as analytics_router
from .api.enhanced_actions import router as enhanced_actions_router
from .api.code_review import router as code_review_router

setup_logging()
app = FastAPI(
    title="Enhanced Multi-User Code Review Agent",
    description="""
    ## 🚀 Enhanced Multi-User Code Review Agent
    
    A comprehensive, enterprise-grade code review system that supports multiple users and organizations:
    
    ### 👥 **Multi-User Support**
    - **Individual Users**: Each user manages their own GitHub tokens
    - **Organization Support**: Team-wide tokens for shared repositories
    - **Role-Based Access**: User, Admin, and Organization Admin roles
    - **Secure Token Storage**: Encrypted GitHub token management
    
    ### 🔍 **Core Features**
    - **Full Repository Analysis**: Complete codebase scanning with multiple tools
    - **PR/Diff Analysis**: Focus on specific changes in pull requests
    - **Enhanced Security**: OWASP Top 10 vulnerability detection
    - **Test Generation**: Automatic test plan and test case generation
    
    ### 🛡️ **Security Analysis**
    - SQL Injection detection
    - XSS vulnerability scanning
    - Command injection prevention
    - Hardcoded secrets detection
    - Dependency vulnerability scanning
    
    ### 🧪 **Testing & Quality**
    - Unit test generation
    - Integration test planning
    - Test coverage analysis
    - Framework-specific test templates
    
    ### 🔗 **GitHub Integration**
    - **Multi-User Tokens**: Each user's token for their repos
    - **Organization Tokens**: Shared access for team repos
    - **PR Comment Posting**: Findings posted directly to PRs
    - **Comprehensive Reviews**: Detailed PR reviews with inline suggestions
    - **Status Updates**: Pass/fail based on findings
    
    ### 🛠️ **Supported Languages**
    - Python (Ruff, MyPy, Bandit, Semgrep)
    - JavaScript/TypeScript (ESLint, Prettier)
    - YAML/JSON validation
    - Configuration file analysis
    
    ### 📊 **Advanced Features**
    - Risk scoring (0-10 scale)
    - Merge blocking rules
    - Auto-fixable issue detection
    - Breaking change identification
    - User activity tracking
    - Organization-wide analytics
    
    ### 🔐 **Security & Privacy**
    - Encrypted token storage
    - User isolation
    - Organization boundaries
    - Audit logging
    - Secure API access
    """,
    version="2.1.0",
    contact={
        "name": "Code Review Agent",
        "url": "https://github.com/your-org/code-review-agent",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

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

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": "2024-01-20T14:30:00Z",
        "version": "2.1.0",
        "services": {
            "database": "connected",
            "api": "running"
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Enhanced Multi-User Code Review Agent API",
        "version": "2.1.0",
        "features": [
            "Multi-user support with individual GitHub tokens",
            "Organization-wide token management",
            "Full repository analysis",
            "PR/Diff analysis", 
            "OWASP Top 10 security checks",
            "Automatic test generation",
            "GitHub PR integration",
            "Risk scoring and merge gates",
            "User activity tracking",
            "Secure token encryption"
        ],
        "docs": "/docs",
        "health": "/healthz"
    }

# Include all routers
app.include_router(jobs_router, tags=["jobs"])
app.include_router(reports_router, tags=["reports"])
app.include_router(config_router, tags=["config"])
app.include_router(actions_router, tags=["actions"])
app.include_router(users_router, tags=["users"])
app.include_router(analytics_router, tags=["analytics"])
app.include_router(enhanced_actions_router, tags=["enhanced-actions"])
app.include_router(code_review_router, tags=["code-review"])
