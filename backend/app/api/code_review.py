"""
Code Review Agent API endpoints

This module provides REST API endpoints for the Code Review Agent,
allowing it to be used both as part of the pipeline and standalone.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
from pathlib import Path
import tempfile
import shutil
import os

from ..review.code_review_agent import CodeReviewAgent
from ..review.enhanced_pipeline import EnhancedPipeline, run_enhanced_review
from ..core.settings import settings
from ..db.models import Job, Finding
from ..db.base import get_session
from sqlmodel import select

router = APIRouter(prefix="/code-review", tags=["Code Review Agent"])

# Pydantic models for API requests/responses
class CodeReviewRequest(BaseModel):
    """Request model for standalone code review"""
    repo_path: Optional[str] = Field(None, description="Path to local repository")
    include_llm: bool = Field(True, description="Whether to include LLM enrichment")
    analysis_options: Optional[Dict[str, Any]] = Field(None, description="Analysis configuration options")

class CodeReviewResponse(BaseModel):
    """Response model for code review results"""
    status: str
    total_findings: int
    findings_by_category: Dict[str, int]
    findings_by_severity: Dict[str, int]
    summary: Dict[str, Any]
    findings: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class PipelineReviewRequest(BaseModel):
    """Request model for pipeline code review"""
    job_id: str = Field(..., description="Job ID for pipeline analysis")
    include_code_review: bool = Field(True, description="Whether to include code review analysis")

class ExportRequest(BaseModel):
    """Request model for exporting results"""
    format: str = Field(..., description="Export format (json, markdown, or html)")
    include_code_snippets: bool = Field(True, description="Whether to include code snippets in export")

class AnalysisOptions(BaseModel):
    """Configuration options for code review analysis"""
    max_file_size: int = Field(1000000, description="Maximum file size to analyze in bytes")
    include_patterns: List[str] = Field(["*.py", "*.js", "*.ts", "*.java"], description="File patterns to include")
    exclude_patterns: List[str] = Field(["__pycache__", "node_modules", ".git"], description="Patterns to exclude")
    complexity_threshold: int = Field(10, description="Cyclomatic complexity threshold")
    function_length_threshold: int = Field(20, description="Function length threshold")
    class_length_threshold: int = Field(50, description="Class length threshold")
    enable_duplication_detection: bool = Field(True, description="Enable code duplication detection")
    enable_efficiency_analysis: bool = Field(True, description="Enable efficiency analysis")
    enable_hardcoded_detection: bool = Field(True, description="Enable hardcoded value detection")


@router.post("/standalone", response_model=CodeReviewResponse)
async def run_standalone_code_review(
    request: CodeReviewRequest,
    background_tasks: BackgroundTasks
):
    """
    Run standalone code review analysis on a local repository
    
    This endpoint allows you to analyze code quality, refactoring opportunities,
    and reusable method suggestions without going through the full pipeline.
    """
    try:
        if not request.repo_path:
            raise HTTPException(status_code=400, detail="Repository path is required")
        
        repo_path = Path(request.repo_path)
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail=f"Repository path does not exist: {request.repo_path}")
        
        if not repo_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.repo_path}")
        
        # Initialize Code Review Agent
        agent = CodeReviewAgent(
            repo_path=repo_path,
            standalone=True
        )
        
        # Run analysis
        results = await agent.run_code_review()
        
        if results.get("status") == "error":
            raise HTTPException(status_code=500, detail=f"Analysis failed: {results.get('error')}")
        
        # Convert to response model
        response = CodeReviewResponse(
            status=results.get("status", "completed"),
            total_findings=results.get("total_findings", 0),
            findings_by_category=results.get("findings_by_category", {}),
            findings_by_severity=results.get("findings_by_severity", {}),
            summary=results.get("summary", {}),
            findings=results.get("findings", []),
            metadata={
                "repo_path": str(repo_path),
                "analysis_type": "standalone",
                "tools_used": ["code_review_agent"],
                "analysis_options": request.analysis_options
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/pipeline", response_model=CodeReviewResponse)
async def run_pipeline_code_review(
    request: PipelineReviewRequest,
    background_tasks: BackgroundTasks
):
    """
    Run enhanced pipeline analysis with code review integration
    
    This endpoint runs the full enhanced pipeline including security scanning
    and code review analysis, then returns comprehensive results.
    """
    try:
        # Validate job exists
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == request.job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found")
        
        # Run enhanced pipeline
        results = await run_enhanced_review(
            request.job_id, 
            include_code_review=request.include_code_review
        )
        
        if results.get("status") == "error":
            raise HTTPException(status_code=500, detail=f"Pipeline analysis failed: {results.get('error')}")
        
        # Convert to response model
        response = CodeReviewResponse(
            status=results.get("status", "completed"),
            total_findings=results.get("total_findings", 0),
            findings_by_category=results.get("findings_by_category", {}),
            findings_by_severity=results.get("findings_by_severity", {}),
            summary=results.get("summary", {}),
            findings=results.get("findings", []),
            metadata={
                "job_id": request.job_id,
                "analysis_type": "pipeline",
                "pipeline_stages": results.get("pipeline_stages", []),
                "tools_used": results.get("metadata", {}).get("tools_used", []),
                "stages_completed": results.get("metadata", {}).get("stages_completed", [])
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/upload-and-analyze", response_model=CodeReviewResponse)
async def upload_and_analyze_code(
    file: UploadFile = File(...),
    analysis_options: Optional[AnalysisOptions] = None
):
    """
    Upload code files and run standalone code review analysis
    
    This endpoint allows you to upload code files (ZIP, tar.gz, or individual files)
    and analyze them without needing a local repository.
    """
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Handle different file types
            if file.filename.endswith('.zip'):
                # Extract ZIP file
                import zipfile
                with zipfile.ZipFile(file.file, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
            elif file.filename.endswith(('.tar.gz', '.tgz')):
                # Extract tar.gz file
                import tarfile
                with tarfile.open(file.file, 'r:gz') as tar_ref:
                    tar_ref.extractall(temp_path)
            else:
                # Single file - create appropriate directory structure
                file_path = temp_path / file.filename
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(file.file, f)
            
            # Initialize Code Review Agent
            agent = CodeReviewAgent(
                repo_path=temp_path,
                standalone=True
            )
            
            # Run analysis
            results = await agent.run_code_review()
            
            if results.get("status") == "error":
                raise HTTPException(status_code=500, detail=f"Analysis failed: {results.get('error')}")
            
            # Convert to response model
            response = CodeReviewResponse(
                status=results.get("status", "completed"),
                total_findings=results.get("total_findings", 0),
                findings_by_category=results.get("findings_by_category", {}),
                findings_by_severity=results.get("findings_by_severity", {}),
                summary=results.get("summary", {}),
                findings=results.get("findings", []),
                metadata={
                    "uploaded_file": file.filename,
                    "analysis_type": "upload",
                    "tools_used": ["code_review_agent"],
                    "analysis_options": analysis_options.dict() if analysis_options else None
                }
            )
            
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/export/{job_id}")
async def export_code_review_results(
    job_id: str,
    format: str = "json",
    include_code_snippets: bool = True
):
    """
    Export code review results for a specific job
    
    Supported formats: json, markdown, html
    """
    try:
        # Validate format
        if format not in ["json", "markdown", "html"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: json, markdown, or html")
        
        # Check if job exists
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Get findings for the job
        result = await session.execute(select(Finding).where(Finding.job_id == job_id))
        findings = result.scalars().all()
        
        if not findings:
            raise HTTPException(status_code=404, detail=f"No findings found for job {job_id}")
        
        # Prepare export data
        export_data = {
            "job_id": job_id,
            "repo_url": job.repo_url,
            "created_at": job.created_at.isoformat(),
            "total_findings": len(findings),
            "findings": []
        }
        
        for finding in findings:
            finding_data = {
                "tool": finding.tool,
                "severity": finding.severity,
                "file": finding.file,
                "line": finding.line,
                "rule_id": finding.rule_id,
                "message": finding.message,
                "remediation": finding.remediation,
                "autofixable": finding.autofixable,
                "vulnerability_type": finding.vulnerability_type
            }
            
            if include_code_snippets and finding.code_snippet:
                finding_data["code_snippet"] = finding.code_snippet
            
            export_data["findings"].append(finding_data)
        
        # Return in requested format
        if format == "json":
            return JSONResponse(content=export_data)
        elif format == "markdown":
            markdown_content = _generate_markdown_export(export_data)
            return JSONResponse(content={"markdown": markdown_content})
        elif format == "html":
            html_content = _generate_html_export(export_data)
            return JSONResponse(content={"html": html_content})
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the current status of a code review job"""
    try:
        async with get_session() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            return {
                "job_id": job.id,
                "status": job.status,
                "current_stage": job.current_stage,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "findings_count": job.findings_count,
                "repo_url": job.repo_url
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/jobs/{job_id}/findings")
async def get_job_findings(
    job_id: str,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get findings for a specific job with optional filtering"""
    try:
        async with get_session() as session:
            # Check if job exists
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            # Build query with filters
            query = select(Finding).where(Finding.job_id == job_id)
            
            if category:
                query = query.where(Finding.vulnerability_type == category)
            
            if severity:
                query = query.where(Finding.severity == severity)
            
            # Add pagination
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            findings = result.scalars().all()
            
            return {
                "job_id": job_id,
                "total_findings": len(findings),
                "limit": limit,
                "offset": offset,
                "findings": [
                    {
                        "id": f.id,
                        "tool": f.tool,
                        "severity": f.severity,
                        "file": f.file,
                        "line": f.line,
                        "rule_id": f.rule_id,
                        "message": f.message,
                        "remediation": f.remediation,
                        "autofixable": f.autofixable,
                        "vulnerability_type": f.vulnerability_type,
                        "code_snippet": f.code_snippet
                    }
                    for f in findings
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/stats")
async def get_code_review_stats():
    """Get overall statistics for code review jobs"""
    try:
        async with get_session() as session:
            # Get total jobs
            result = await session.execute(select(Job))
            all_jobs = result.scalars().all()
            
            # Get total findings
            result = await session.execute(select(Finding))
            all_findings = result.scalars().all()
            
            # Calculate statistics
            total_jobs = len(all_jobs)
            total_findings = len(all_findings)
            completed_jobs = len([j for j in all_jobs if j.status == "completed"])
            failed_jobs = len([j for j in all_jobs if j.status == "failed"])
            
            # Findings by severity
            severity_counts = {}
            for finding in all_findings:
                severity = finding.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Findings by tool
            tool_counts = {}
            for finding in all_findings:
                tool = finding.tool
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
                "total_findings": total_findings,
                "findings_by_severity": severity_counts,
                "findings_by_tool": tool_counts,
                "average_findings_per_job": (total_findings / total_jobs) if total_jobs > 0 else 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def _generate_markdown_export(data: Dict[str, Any]) -> str:
    """Generate Markdown export of code review results"""
    markdown = f"# Code Review Report\n\n"
    markdown += f"**Job ID:** {data['job_id']}\n"
    markdown += f"**Repository:** {data['repo_url']}\n"
    markdown += f"**Date:** {data['created_at']}\n"
    markdown += f"**Total Findings:** {data['total_findings']}\n\n"
    
    markdown += "## Findings\n\n"
    for i, finding in enumerate(data['findings'], 1):
        markdown += f"### {i}. {finding['file']}:{finding['line']}\n"
        markdown += f"**Tool:** {finding['tool']}\n"
        markdown += f"**Severity:** {finding['severity']}\n"
        markdown += f"**Message:** {finding['message']}\n"
        
        if finding.get('remediation'):
            markdown += f"**Remediation:** {finding['remediation']}\n"
        
        if finding.get('code_snippet'):
            markdown += f"**Code:**\n```\n{finding['code_snippet']}\n```\n"
        
        markdown += "\n"
    
    return markdown


def _generate_html_export(data: Dict[str, Any]) -> str:
    """Generate HTML export of code review results"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Code Review Report - {data['job_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
            .finding {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .severity-high {{ border-left: 5px solid #ff4444; }}
            .severity-medium {{ border-left: 5px solid #ffaa00; }}
            .severity-low {{ border-left: 5px solid #44aa44; }}
            .code-snippet {{ background-color: #f8f8f8; padding: 10px; border-radius: 3px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Code Review Report</h1>
            <p><strong>Job ID:</strong> {data['job_id']}</p>
            <p><strong>Repository:</strong> {data['repo_url']}</p>
            <p><strong>Date:</strong> {data['created_at']}</p>
            <p><strong>Total Findings:</strong> {data['total_findings']}</p>
        </div>
        
        <h2>Findings</h2>
    """
    
    for i, finding in enumerate(data['findings'], 1):
        severity_class = f"severity-{finding['severity']}"
        html += f"""
        <div class="finding {severity_class}">
            <h3>{i}. {finding['file']}:{finding['line']}</h3>
            <p><strong>Tool:</strong> {finding['tool']}</p>
            <p><strong>Severity:</strong> {finding['severity']}</p>
            <p><strong>Message:</strong> {finding['message']}</p>
        """
        
        if finding.get('remediation'):
            html += f"<p><strong>Remediation:</strong> {finding['remediation']}</p>"
        
        if finding.get('code_snippet'):
            html += f"<div class='code-snippet'><pre>{finding['code_snippet']}</pre></div>"
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for the Code Review Agent API"""
    return {
        "status": "healthy",
        "service": "Code Review Agent API",
        "version": "1.0.0",
        "llm_provider": settings.LLM_PROVIDER
    }
