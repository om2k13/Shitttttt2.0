from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlmodel import Session, select
from typing import List, Dict, Optional
from pathlib import Path
from ..db.base import get_session
from ..db.models import Job, Finding
from ..review.autofix.master_fixer import MasterAutoFixer
from ..review.enhanced_analyzer import EnhancedAnalyzer
from ..integrations.git_automation import GitAutomation

router = APIRouter(prefix="/api/enhanced", tags=["enhanced-actions"])

@router.post("/analyze-and-fix/{job_id}")
async def analyze_and_fix_job(
    job_id: str,
    auto_apply_fixes: bool = True,
    create_github_pr: bool = False,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Comprehensive analysis and auto-fix workflow for a job"""
    try:
        # Get all findings for the job
        result = await session.execute(select(Finding).where(Finding.job_id == job_id))
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail="No findings found for job")
        
        # Get the workspace path for this job
        job_result = await session.execute(select(Job).where(Job.id == job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Find the workspace directory (optional for analysis)
        workspace_dir = Path(f".workspaces/{job_id}")
        workspace_exists = workspace_dir.exists()
        
        # Step 1: Enhanced Analysis (can work without workspace)
        print("🔍 Starting enhanced analysis...")
        
        if workspace_exists:
            analyzer = EnhancedAnalyzer(workspace_dir)
            analysis_results = await analyzer.analyze_findings_comprehensive([
                {
                    "id": f.id,
                    "tool": f.tool,
                    "autofixable": f.autofixable,
                    "message": f.message,
                    "file": f.file,
                    "line": f.line,
                    "rule_id": f.rule_id,
                    "remediation": f.remediation,
                    "severity": f.severity
                }
                for f in findings
            ])
        else:
            # Fallback analysis without workspace access
            print("⚠️  Workspace not available, performing basic analysis...")
            analysis_results = {
                "summary": {
                    "total_findings": len(findings),
                    "autofixable_findings": len([f for f in findings if f.autofixable]),
                    "security_issues": len([f for f in findings if f.tool in ["bandit", "semgrep", "detect-secrets"]]),
                    "code_quality_issues": len([f for f in findings if f.tool in ["ruff", "mypy", "radon"]]),
                    "dependency_issues": len([f for f in findings if f.tool in ["pip-audit", "npm-audit"]])
                },
                "findings_analysis": [
                    {
                        "id": f.id,
                        "tool": f.tool,
                        "severity": f.severity,
                        "category": f.tool,
                        "message": f.message,
                        "remediation": f.remediation,
                        "autofixable": f.autofixable,
                        "risk_score": 8 if f.severity == "high" else 5 if f.severity == "medium" else 2,
                        "effort": "low" if f.autofixable else "medium"
                    }
                    for f in findings
                ],
                "recommendations": [
                    "Upgrade vulnerable dependencies" if any(f.tool in ["pip-audit", "npm-audit"] for f in findings) else None,
                    "Review security findings" if any(f.tool in ["bandit", "semgrep", "detect-secrets"] for f in findings) else None,
                    "Address code quality issues" if any(f.tool in ["ruff", "mypy", "radon"] for f in findings) else None
                ]
            }
            analysis_results["recommendations"] = [r for r in analysis_results["recommendations"] if r is not None]
        
        # Step 2: Auto-Fix (if requested)
        fix_results = None
        if auto_apply_fixes and workspace_exists:
            print("🔧 Applying automatic fixes...")
            fixer = MasterAutoFixer(workspace_dir)
            fix_results = await fixer.apply_all_fixes([
                {
                    "id": f.id,
                    "tool": f.tool,
                    "autofixable": f.autofixable,
                    "message": f.message,
                    "file": f.file,
                    "line": f.line,
                    "rule_id": f.rule_id,
                    "remediation": f.remediation
                }
                for f in findings
            ])
            
            # Validate fixes
            validation_results = await fixer.validate_fixes()
            fix_results["validation"] = validation_results
        
        # Step 3: GitHub Integration (if requested)
        git_results = None
        if create_github_pr and fix_results:
            print("🚀 Setting up GitHub integration...")
            git_automation = GitAutomation(workspace_dir)
            
            # Check if it's a Git repository
            if await git_automation.is_git_repository():
                # Prepare fixes for Git
                all_fixes = []
                for fix_type, fixes in fix_results.items():
                    if isinstance(fixes, dict) and 'successful_fixes' in fixes:
                        # This is a summary, skip
                        continue
                    if isinstance(fixes, list):
                        all_fixes.extend(fixes)
                
                # Run Git workflow
                git_results = await git_automation.automate_fix_workflow(
                    all_fixes, 
                    create_pr=True
                )
            else:
                git_results = {"error": "Not a Git repository"}
        
        return {
            "success": True,
            "job_id": job_id,
            "analysis": analysis_results,
            "fixes": fix_results,
            "git_integration": git_results,
            "summary": {
                "total_findings": len(findings),
                "autofixable_findings": len([f for f in findings if f.autofixable]),
                "fixes_applied": fix_results.get("summary", {}).get("successful_fixes", 0) if fix_results else 0,
                "github_pr_created": git_results.get("success", False) if git_results else False
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{job_id}")
async def get_enhanced_analysis(job_id: str, session: Session = Depends(get_session)):
    """Get enhanced analysis for a job with code snippets, risk scores, and fix suggestions"""
    try:
        # Get all findings for the job
        result = await session.execute(select(Finding).where(Finding.job_id == job_id))
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail="No findings found for job")
        
        # Get the workspace path for this job
        job_result = await session.execute(select(Job).where(Job.id == job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Find the workspace directory (optional for analysis)
        workspace_dir = Path(f".workspaces/{job_id}")
        workspace_exists = workspace_dir.exists()
        
        # Perform enhanced analysis
        if workspace_exists:
            analyzer = EnhancedAnalyzer(workspace_dir)
            analysis_results = await analyzer.analyze_findings_comprehensive([
                {
                    "id": f.id,
                    "tool": f.tool,
                    "autofixable": f.autofixable,
                    "message": f.message,
                    "file": f.file,
                    "line": f.line,
                    "rule_id": f.rule_id,
                    "remediation": f.remediation,
                    "severity": f.severity
                }
                for f in findings
            ])
        else:
            # Fallback analysis without workspace access
            print("⚠️  Workspace not available, performing basic analysis...")
            analysis_results = {
                "summary": {
                    "total_findings": len(findings),
                    "autofixable_findings": len([f for f in findings if f.autofixable]),
                    "security_issues": len([f for f in findings if f.tool in ["bandit", "semgrep", "detect-secrets"]]),
                    "code_quality_issues": len([f for f in findings if f.tool in ["ruff", "mypy", "radon"]]),
                    "dependency_issues": len([f for f in findings if f.tool in ["pip-audit", "npm-audit"]])
                },
                "findings_analysis": [
                    {
                        "id": f.id,
                        "tool": f.tool,
                        "severity": f.severity,
                        "category": f.tool,
                        "message": f.message,
                        "remediation": f.remediation,
                        "autofixable": f.autofixable,
                        "risk_score": 8 if f.severity == "high" else 5 if f.severity == "medium" else 2,
                        "effort": "low" if f.autofixable else "medium"
                    }
                    for f in findings
                ],
                "recommendations": [
                    "Upgrade vulnerable dependencies" if any(f.tool in ["pip-audit", "npm-audit"] for f in findings) else None,
                    "Review security findings" if any(f.tool in ["bandit", "semgrep", "detect-secrets"] for f in findings) else None,
                    "Address code quality issues" if any(f.tool in ["ruff", "mypy", "radon"] for f in findings) else None
                ]
            }
            analysis_results["recommendations"] = [r for r in analysis_results["recommendations"] if r is not None]
        
        return {
            "success": True,
            "job_id": job_id,
            "analysis": {
                "job_id": job_id,
                "findings": analysis_results.get("findings_analysis", []),
                "summary": {
                    "total_findings": analysis_results.get("summary", {}).get("total_findings", 0),
                    "auto_fixable": analysis_results.get("summary", {}).get("autofixable_findings", 0),
                    "manual_fixes": analysis_results.get("summary", {}).get("total_findings", 0) - analysis_results.get("summary", {}).get("autofixable_findings", 0),
                    "risk_distribution": {
                        "high": len([f for f in analysis_results.get("findings_analysis", []) if f.get("risk_score", 0) >= 8]),
                        "medium": len([f for f in analysis_results.get("findings_analysis", []) if 5 <= f.get("risk_score", 0) < 8]),
                        "low": len([f for f in analysis_results.get("findings_analysis", []) if f.get("risk_score", 0) < 5])
                    },
                    "priority_distribution": {
                        "critical": len([f for f in analysis_results.get("findings_analysis", []) if f.get("severity") == "high"]),
                        "high": len([f for f in analysis_results.get("findings_analysis", []) if f.get("severity") == "medium"]),
                        "medium": len([f for f in analysis_results.get("findings_analysis", []) if f.get("severity") == "low"]),
                        "low": 0
                    }
                },
                "recommendations": analysis_results.get("recommendations", []),
                "estimated_fix_time": "2-4 hours",
                "risk_reduction_potential": 75
            }
        }
        
    except Exception as e:
        print(f"❌ Enhanced analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fix-and-commit/{job_id}")
async def fix_and_commit_job(
    job_id: str,
    create_pr: bool = True,
    session: Session = Depends(get_session)
):
    """Apply fixes and commit them to Git with optional PR creation"""
    try:
        # Get all findings for the job
        result = await session.execute(select(Finding).where(Finding.job_id == job_id))
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail="No findings found for job")
        
        # Get the workspace path for this job
        job_result = await session.execute(select(Job).where(Job.id == job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Find the workspace directory
        workspace_dir = Path(f".workspaces/{job_id}")
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Step 1: Apply fixes
        print("🔧 Applying automatic fixes...")
        fixer = MasterAutoFixer(workspace_dir)
        fix_results = await fixer.apply_all_fixes([
            {
                "id": f.id,
                "tool": f.tool,
                "autofixable": f.autofixable,
                "message": f.message,
                "file": f.file,
                "line": f.line,
                "rule_id": f.rule_id,
                "remediation": f.remediation
            }
            for f in findings
        ])
        
        # Step 2: Git integration
        print("🚀 Setting up Git integration...")
        git_automation = GitAutomation(workspace_dir)
        
        if not await git_automation.is_git_repository():
            return {
                "success": False,
                "error": "Not a Git repository",
                "fixes": fix_results
            }
        
        # Prepare fixes for Git
        all_fixes = []
        for fix_type, fixes in fix_results.items():
            if isinstance(fixes, dict) and 'successful_fixes' in fixes:
                # This is a summary, skip
                continue
            if isinstance(fixes, list):
                all_fixes.extend(fixes)
        
        # Run Git workflow
        git_results = await git_automation.automate_fix_workflow(
            all_fixes, 
            create_pr=create_pr
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "fixes": fix_results,
            "git_integration": git_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fix-summary/{job_id}")
async def get_fix_summary(job_id: str, session: Session = Depends(get_session)):
    """Get a summary of what fixes can be applied for a job"""
    try:
        # Get all findings for the job
        result = await session.execute(select(Finding).where(Finding.job_id == job_id))
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail="No findings found for job")
        
        # Get the workspace path for this job
        job_result = await session.execute(select(Job).where(Job.id == job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Find the workspace directory
        workspace_dir = Path(f".workspaces/{job_id}")
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Get fix summary
        fixer = MasterAutoFixer(workspace_dir)
        summary = await fixer.get_fix_summary([
            {
                "id": f.id,
                "tool": f.tool,
                "autofixable": f.autofixable,
                "message": f.message,
                "file": f.file,
                "line": f.line,
                "rule_id": f.rule_id,
                "remediation": f.remediation
            }
            for f in findings
        ])
        
        # Check Git status
        git_automation = GitAutomation(workspace_dir)
        git_status = await git_automation.get_status()
        git_info = {
            "is_git_repo": await git_automation.is_git_repository(),
            "current_branch": await git_automation.get_current_branch() if await git_automation.is_git_repository() else None,
            "remote_origin": await git_automation.get_remote_origin() if await git_automation.is_git_repository() else None,
            "status": git_status
        }
        
        return {
            "success": True,
            "job_id": job_id,
            "fix_summary": summary,
            "git_info": git_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-fixes/{job_id}")
async def validate_applied_fixes(job_id: str, session: Session = Depends(get_session)):
    """Validate that applied fixes don't break the codebase"""
    try:
        # Get the workspace path for this job
        job_result = await session.execute(select(Job).where(Job.id == job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Find the workspace directory
        workspace_dir = Path(f".workspaces/{job_id}")
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Validate fixes
        fixer = MasterAutoFixer(workspace_dir)
        validation_results = await fixer.validate_fixes()
        
        return {
            "success": True,
            "job_id": job_id,
            "validation": validation_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
