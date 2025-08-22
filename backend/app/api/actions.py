from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlmodel import Session, select
from typing import List, Dict, Optional, Any
import json
from ..db.base import get_session
from ..db.models import Job, Finding, Report
from ..review.pipeline import run_review
from ..review.pr_analyzer import analyze_pr_changes
from ..review.security_analyzer import run_security_analysis
from ..review.test_generator import generate_test_plan_for_changes
from ..review.performance_analyzer import analyze_code_performance
from ..review.api_analyzer import analyze_api_changes_for_branches, generate_api_documentation_for_branch
from ..integrations.github import post_findings_to_pr, create_pr_review
from ..core.settings import settings

router = APIRouter(prefix="/api", tags=["actions"])

@router.post("/apply-fix")
async def apply_safe_fix(
    finding_id: int,
    session: Session = Depends(get_session)
):
    """Apply a safe auto-fix for a specific finding"""
    try:
        # Get the finding
        result = await session.execute(select(Finding).where(Finding.id == finding_id))
        finding = result.scalar_one_or_none()
        
        if not finding:
            raise HTTPException(status_code=404, detail="Finding not found")
        
        if not finding.autofixable:
            raise HTTPException(status_code=400, detail="Finding is not auto-fixable")
        
        # Get all findings for context
        all_findings_result = await session.execute(select(Finding).where(Finding.job_id == finding.job_id))
        all_findings = all_findings_result.scalars().all()
        
        # Convert to dict for the fixer
        findings_dict = []
        for f in all_findings:
            findings_dict.append({
                "id": f.id,
                "tool": f.tool,
                "autofixable": f.autofixable,
                "message": f.message,
                "file": f.file,
                "line": f.line,
                "rule_id": f.rule_id,
                "remediation": f.remediation
            })
        
        # Get the workspace path for this job
        job_result = await session.execute(select(Job).where(Job.id == finding.job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Find the workspace directory
        from pathlib import Path
        workspace_dir = Path(f".workspaces/{finding.job_id}")
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Apply the fix using the master fixer
        from ..review.autofix.master_fixer import MasterAutoFixer
        fixer = MasterAutoFixer(workspace_dir)
        
        fix_result = await fixer.fix_specific_finding(str(finding_id), findings_dict)
        
        if "error" in fix_result:
            raise HTTPException(status_code=400, detail=fix_result["error"])
        
        return {
            "success": True,
            "message": f"Auto-fix applied for finding {finding_id}",
            "finding_id": finding_id,
            "fix_details": fix_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply-all-fixes")
async def apply_all_autofixes(
    job_id: str,
    session: Session = Depends(get_session)
):
    """Apply all available auto-fixes for a job"""
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
        from pathlib import Path
        workspace_dir = Path(f".workspaces/{job_id}")
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Convert findings to dict
        findings_dict = []
        for f in findings:
            findings_dict.append({
                "id": f.id,
                "tool": f.tool,
                "autofixable": f.autofixable,
                "message": f.message,
                "file": f.file,
                "line": f.line,
                "rule_id": f.rule_id,
                "remediation": f.remediation
            })
        
        # Apply all fixes using the master fixer
        from ..review.autofix.master_fixer import MasterAutoFixer
        fixer = MasterAutoFixer(workspace_dir)
        
        fix_results = await fixer.apply_all_fixes(findings_dict)
        
        # Validate fixes
        validation_results = await fixer.validate_fixes()
        fix_results["validation"] = validation_results
        
        return {
            "success": True,
            "message": f"Applied {fix_results['summary']['successful_fixes']} auto-fixes",
            "job_id": job_id,
            "fix_results": fix_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fix-summary/{job_id}")
async def get_fix_summary(
    job_id: str,
    session: Session = Depends(get_session)
):
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
        from pathlib import Path
        workspace_dir = Path(f".workspaces/{job_id}")
        
        if not workspace_dir.exists():
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Convert findings to dict
        findings_dict = []
        for f in findings:
            findings_dict.append({
                "id": f.id,
                "tool": f.tool,
                "autofixable": f.autofixable,
                "message": f.message,
                "file": f.file,
                "line": f.line,
                "rule_id": f.rule_id,
                "remediation": f.remediation
            })
        
        # Get fix summary using the master fixer
        from ..review.autofix.master_fixer import MasterAutoFixer
        fixer = MasterAutoFixer(workspace_dir)
        
        summary = await fixer.get_fix_summary(findings_dict)
        
        return {
            "success": True,
            "job_id": job_id,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-pr")
async def analyze_pull_request(
    repo_url: str,
    pr_number: int,
    base_branch: str = "main",
    head_branch: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Analyze a specific pull request instead of the entire repository"""
    try:
        # Create a new job for PR analysis
        job = Job(
            repo_url=repo_url,
            status="pending",
            pr_number=pr_number,
            base_branch=base_branch,
            head_branch=head_branch or f"pr-{pr_number}",
            is_pr_analysis=True
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start PR analysis in background
        if background_tasks:
            background_tasks.add_task(run_pr_analysis, job.id, repo_url, base_branch, head_branch or f"pr-{pr_number}")
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"PR analysis started for {repo_url}#{pr_number}",
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security-analysis")
async def run_security_analysis_endpoint(
    repo_url: str,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Run enhanced security analysis with OWASP Top 10 checks"""
    try:
        # Create a new job for security analysis
        job = Job(
            repo_url=repo_url,
            status="pending",
            is_pr_analysis=False
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start security analysis in background
        if background_tasks:
            background_tasks.add_task(run_security_analysis_task, job.id, repo_url)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Security analysis started for {repo_url}",
            "status": "pending",
            "analysis_type": "enhanced_security",
            "features": [
                "OWASP Top 10 vulnerability detection",
                "Dependency vulnerability scanning",
                "Code security pattern analysis",
                "Hardcoded secrets detection",
                "Input validation checks"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-test-plan")
async def generate_test_plan_endpoint(
    repo_url: str,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Generate a comprehensive test plan for the repository"""
    try:
        # Create a new job for test plan generation
        job = Job(
            repo_url=repo_url,
            status="pending",
            is_pr_analysis=False
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start test plan generation in background
        if background_tasks:
            background_tasks.add_task(generate_test_plan_task, job.id, repo_url)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Test plan generation started for {repo_url}",
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance-analysis")
async def run_performance_analysis_endpoint(
    repo_url: str,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Run performance analysis to detect optimization opportunities"""
    try:
        # Create a new job for performance analysis
        job = Job(
            repo_url=repo_url,
            status="pending",
            is_pr_analysis=False
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start performance analysis in background
        if background_tasks:
            background_tasks.add_task(run_performance_analysis_task, job.id, repo_url)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Performance analysis started for {repo_url}",
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api-change-analysis")
async def analyze_api_changes_endpoint(
    repo_url: str,
    base_branch: str = "main",
    head_branch: str = "feature",
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Analyze API changes between branches for breaking changes"""
    try:
        # Create a new job for API change analysis
        job = Job(
            repo_url=repo_url,
            status="pending",
            is_pr_analysis=False
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start API change analysis in background
        if background_tasks:
            background_tasks.add_task(run_api_change_analysis_task, job.id, repo_url, base_branch, head_branch)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"API change analysis started for {repo_url} ({base_branch} → {head_branch})",
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/comprehensive-analysis")
async def run_comprehensive_analysis_endpoint(
    repo_url: str,
    include_performance: bool = True,
    include_api_analysis: bool = True,
    include_test_generation: bool = True,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """Run a comprehensive analysis including all available checks"""
    try:
        # Create a new job for comprehensive analysis
        job = Job(
            repo_url=repo_url,
            status="pending",
            is_pr_analysis=False
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start comprehensive analysis in background
        if background_tasks:
            background_tasks.add_task(
                run_comprehensive_analysis_task, 
                job.id, 
                repo_url, 
                include_performance, 
                include_api_analysis, 
                include_test_generation
            )
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Comprehensive analysis started for {repo_url}",
            "status": "pending",
            "analysis_types": {
                "security": True,
                "performance": include_performance,
                "api_changes": include_api_analysis,
                "test_generation": include_test_generation
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/github/pr-comments")
async def post_github_pr_comments(
    repo_url: str,
    pr_number: int,
    session: Session = Depends(get_session)
):
    """Post findings as comments to a GitHub PR"""
    try:
        # Get findings for the repository
        result = await session.execute(select(Finding).where(Finding.job_id == 1))  # TODO: Get actual job ID
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail="No findings found for this repository")
        
        # Convert findings to dict format
        findings_dict = []
        for finding in findings:
            findings_dict.append({
                "tool": finding.tool,
                "severity": finding.severity,
                "file": finding.file,
                "line": finding.line,
                "rule_id": finding.rule_id,
                "message": finding.message,
                "remediation": finding.remediation,
                "autofixable": finding.autofixable,
                "vulnerability_type": finding.vulnerability_type,
                "code_snippet": finding.code_snippet,
                "pr_context": json.loads(finding.pr_context) if finding.pr_context else {}
            })
        
        # Post to GitHub
        result = await post_findings_to_pr(repo_url, pr_number, findings_dict)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": f"Posted {result.get('comments_posted', 0)} comments to PR #{pr_number}",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/github/pr-review")
async def create_github_pr_review(
    repo_url: str,
    pr_number: int,
    session: Session = Depends(get_session)
):
    """Create a comprehensive PR review on GitHub"""
    try:
        # Get findings for the repository
        result = await session.execute(select(Finding).where(Finding.job_id == 1))  # TODO: Get actual job ID
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail="No findings found for this repository")
        
        # Convert findings to dict format
        findings_dict = []
        for finding in findings:
            findings_dict.append({
                "tool": finding.tool,
                "severity": finding.severity,
                "file": finding.file,
                "line": finding.line,
                "rule_id": finding.rule_id,
                "message": finding.message,
                "remediation": finding.remediation,
                "autofixable": finding.autofixable,
                "vulnerability_type": finding.vulnerability_type,
                "code_snippet": finding.code_snippet,
                "pr_context": json.loads(finding.pr_context) if finding.pr_context else {}
            })
        
        # Create PR review
        result = await create_pr_review(repo_url, pr_number, findings_dict)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": f"Created PR review for PR #{pr_number}",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def run_pr_analysis(job_id: int, repo_url: str, base_branch: str, head_branch: str):
    """Run PR analysis in the background"""
    try:
        # TODO: Implement PR analysis logic
        # This would involve:
        # 1. Cloning the repository
        # 2. Analyzing the diff between branches
        # 3. Running security analysis on changed files
        # 4. Generating test plan for changes
        # 5. Updating job status
        
        pass
        
    except Exception as e:
        # Update job status to failed
        pass

@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: int, session: Session = Depends(get_session)):
    """Get the status of a background job"""
    try:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return {
            "job_id": job.id,
            "status": job.status,
            "current_stage": job.current_stage,
            "progress": job.progress,
            "repo_url": job.repo_url,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error_message": getattr(job, 'error_message', None)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: int, session: Session = Depends(get_session)):
    """Get the results of a completed background job"""
    try:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} is not completed. Current status: {job.status}"
            )
        
        return {
            "job_id": job.id,
            "status": job.status,
            "current_stage": job.current_stage,
            "results": job.results,
            "repo_url": job.repo_url,
            "completed_at": job.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_session)
):
    """List all jobs with optional filtering"""
    try:
        query = select(Job)
        
        if status:
            query = query.where(Job.status == status)
        
        query = query.order_by(Job.created_at.desc()).offset(offset).limit(limit)
        result = await session.execute(query)
        jobs = result.scalars().all()
        
        return {
            "jobs": [
                {
                    "id": job.id,
                    "status": job.status,
                    "current_stage": job.current_stage,
                    "progress": job.progress,
                    "repo_url": job.repo_url,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at
                }
                for job in jobs
            ],
            "total": len(jobs),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_security_analysis_task(job_id: int, repo_url: str):
    """Run security analysis in the background"""
    try:
        print(f"🔒 Starting security analysis for job {job_id}")
        
        # Update job status
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "running"
                job.current_stage = "security_analysis"
                job.progress = 20
                await session.commit()
        
        # Clone repository and run security analysis
        from ..core.vcs import clone_repo
        from ..review.enhanced_pipeline import EnhancedPipeline
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            await clone_repo(repo_url, repo_path)
            
            # Initialize enhanced pipeline for security analysis
            pipeline = EnhancedPipeline()
            pipeline.repo_path = repo_path
            
            # Run security scanning only
            security_results = await pipeline._run_security_tools()
            
            # Update job status and store results
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.current_stage = "security_analysis_completed"
                    job.progress = 100
                    job.results = {
                        "security_findings": security_results,
                        "total_findings": len(security_results),
                        "analysis_type": "security_only"
                    }
                    await session.commit()
            
            print(f"✅ Security analysis completed for job {job_id}: {len(security_results)} findings")
            
    except Exception as e:
        print(f"❌ Security analysis failed for job {job_id}: {e}")
        # Update job status to failed
        try:
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.current_stage = "security_analysis_failed"
                    job.error_message = str(e)
                    await session.commit()
        except Exception:
            pass

async def generate_test_plan_task(job_id: int, repo_url: str):
    """Generate test plan in the background"""
    try:
        print(f"🧪 Starting test plan generation for job {job_id}")
        
        # Update job status
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "running"
                job.current_stage = "test_generation"
                job.progress = 60
                await session.commit()
        
        # Clone repository
        from ..core.vcs import clone_repo
        from ..review.test_generator import TestGenerator
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            await clone_repo(repo_url, repo_path)
            
            # Initialize test generator
            test_generator = TestGenerator(repo_path)
            
            # Analyze repository structure
            changed_files = []
            for py_file in repo_path.rglob("*.py"):
                if py_file.is_file():
                    changed_files.append({
                        "path": str(py_file.relative_to(repo_path)),
                        "type": "python",
                        "status": "modified"
                    })
            
            for js_file in repo_path.rglob("*.js"):
                if js_file.is_file():
                    changed_files.append({
                        "path": str(js_file.relative_to(repo_path)),
                        "type": "javascript",
                        "status": "modified"
                    })
            
            # Generate test plan
            test_plan = await test_generator.generate_test_plan(
                changed_files=changed_files,
                findings=[]  # No findings for standalone test generation
            )
            
            # Update job status and store results
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.current_stage = "test_generation_completed"
                    job.progress = 100
                    job.results = {"test_plan": test_plan}
                    await session.commit()
            
            print(f"✅ Test plan generation completed for job {job_id}")
            
    except Exception as e:
        print(f"❌ Test plan generation failed for job {job_id}: {e}")
        # Update job status to failed
        try:
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.current_stage = "test_generation_failed"
                    job.error_message = str(e)
                    await session.commit()
        except Exception:
            pass

async def run_performance_analysis_task(job_id: int, repo_url: str):
    """Run performance analysis in the background"""
    try:
        # TODO: Implement performance analysis logic
        # This would involve:
        # 1. Cloning the repository
        # 2. Running performance pattern analysis
        # 3. Detecting N+1 queries, memory leaks, etc.
        # 4. Updating job status and findings
        
        pass
        
    except Exception as e:
        # Update job status to failed
        pass

async def run_api_change_analysis_task(job_id: int, repo_url: str, base_branch: str, head_branch: str):
    """Run API change analysis in the background"""
    try:
        # TODO: Implement API change analysis logic
        # This would involve:
        # 1. Cloning the repository
        # 2. Analyzing API changes between branches
        # 3. Detecting breaking changes
        # 4. Updating job status and findings
        
        pass
        
    except Exception as e:
        # Update job status to failed
        pass

async def run_comprehensive_analysis_task(
    job_id: int, 
    repo_url: str, 
    include_performance: bool, 
    include_api_analysis: bool, 
    include_test_generation: bool
):
    """Run comprehensive analysis in the background"""
    try:
        print(f"🚀 Starting comprehensive analysis for job {job_id}")
        
        # Update job status
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "running"
                job.current_stage = "comprehensive_analysis"
                job.progress = 10
                await session.commit()
        
        # Clone repository and run enhanced pipeline
        from ..core.vcs import clone_repo
        from ..review.enhanced_pipeline import run_enhanced_review
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            await clone_repo(repo_url, repo_path)
            
            # Run enhanced pipeline with code review
            pipeline_results = await run_enhanced_review(
                job_id=str(job_id),
                include_code_review=True
            )
            
            # Update job status and store results
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.current_stage = "comprehensive_analysis_completed"
                    job.progress = 100
                    job.results = {
                        "pipeline_results": pipeline_results,
                        "analysis_types": {
                            "security": True,
                            "code_review": True,
                            "test_generation": True,
                            "performance": include_performance,
                            "api_analysis": include_api_analysis
                        },
                        "total_findings": pipeline_results.get("total_findings", 0),
                        "test_plan": pipeline_results.get("test_plan", {})
                    }
                    await session.commit()
            
            print(f"✅ Comprehensive analysis completed for job {job_id}")
            
    except Exception as e:
        print(f"❌ Comprehensive analysis failed for job {job_id}: {e}")
        # Update job status to failed
        try:
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.current_stage = "comprehensive_analysis_failed"
                    job.error_message = str(e)
                    await session.commit()
        except Exception:
            pass

@router.post("/test-cicd-integration")
async def test_cicd_integration(
    repo_url: str,
    config: Dict[str, Any] = None,
    session: Session = Depends(get_session)
):
    """Test CI/CD integration by executing workflow locally"""
    try:
        if config is None:
            config = {
                "enable_security_analysis": True,
                "enable_test_generation": True,
                "analysis_type": "comprehensive"
            }
        
        # Create a new job for CI/CD testing
        job = Job(
            repo_url=repo_url,
            status="pending",
            is_pr_analysis=False
        )
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        
        # Start CI/CD testing in background
        if background_tasks:
            background_tasks.add_task(test_cicd_integration_task, job.id, repo_url, config)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"CI/CD integration testing started for {repo_url}",
            "status": "pending",
            "config": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def test_cicd_integration_task(job_id: int, repo_url: str, config: Dict[str, Any]):
    """Test CI/CD integration in the background"""
    try:
        print(f"🚀 Starting CI/CD integration testing for job {job_id}")
        
        # Update job status
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "running"
                job.current_stage = "cicd_testing"
                job.progress = 20
                await session.commit()
        
        # Clone repository and test CI/CD integration
        from ..core.vcs import clone_repo
        from ..integrations.cicd import CICDIntegration
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            await clone_repo(repo_url, repo_path)
            
            # Test CI/CD integration
            cicd = CICDIntegration()
            cicd_results = await cicd.execute_github_action_workflow(repo_path, config)
            
            # Update job status and store results
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.current_stage = "cicd_testing_completed"
                    job.progress = 100
                    job.results = {
                        "cicd_results": cicd_results,
                        "config": config,
                        "workflow_file": str(cicd_results.get("workflow_file", ""))
                    }
                    await session.commit()
            
            print(f"✅ CI/CD integration testing completed for job {job_id}")
            
    except Exception as e:
        print(f"❌ CI/CD integration testing failed for job {job_id}: {e}")
        # Update job status to failed
        try:
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.current_stage = "cicd_testing_failed"
                    job.error_message = str(e)
                    await session.commit()
        except Exception:
            pass
