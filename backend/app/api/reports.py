from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from ..db.base import get_session
from ..db.models import Finding, Job
import json
import csv
from io import StringIO
from fastapi.responses import Response
from datetime import datetime, timezone

router = APIRouter(prefix="/api", tags=["reports"])

@router.get("/reports")
async def get_all_reports(session=Depends(get_session)):
    """Get all reports summary with real-time data"""
    try:
        # Get all findings
        result = await session.execute(select(Finding))
        items = result.scalars().all()
        
        # Get all jobs for context
        jobs_result = await session.execute(select(Job))
        jobs = jobs_result.scalars().all()
        
        # Group by job_id with real-time data
        reports_by_job = {}
        for finding in items:
            job_id = finding.job_id
            if job_id not in reports_by_job:
                # Find corresponding job
                job = next((j for j in jobs if j.id == job_id), None)
                reports_by_job[job_id] = {
                    "total": 0,
                    "by_tool": {},
                    "by_severity": {},
                    "job_info": {
                        "repo_url": job.repo_url if job else "Unknown",
                        "status": job.status if job else "Unknown",
                        "created_at": job.created_at.isoformat() if job and job.created_at else None,
                        "completed_at": job.completed_at.isoformat() if job and job.completed_at else None
                    } if job else None
                }
            
            reports_by_job[job_id]["total"] += 1
            reports_by_job[job_id]["by_tool"][finding.tool] = reports_by_job[job_id]["by_tool"].get(finding.tool, 0) + 1
            reports_by_job[job_id]["by_severity"][finding.severity] = reports_by_job[job_id]["by_severity"].get(finding.severity, 0) + 1
        
        return {"reports": reports_by_job}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{job_id}")
async def get_report(job_id: str, session=Depends(get_session)):
    print(f"📊 Fetching report for job: {job_id}")
    try:
        result = await session.execute(select(Finding).where(Finding.job_id==job_id))
        items = result.scalars().all()
        print(f"   Found {len(items)} findings for job {job_id}")
        
        if not items:
            raise HTTPException(404, "report not found")
        
        # Get job information
        job_result = await session.execute(select(Job).where(Job.id==job_id))
        job = job_result.scalar_one_or_none()
        
        # aggregate findings
        summary = {"total": len(items), "by_tool": {}, "by_severity": {}}
        for f in items:
            summary["by_tool"][f.tool] = summary["by_tool"].get(f.tool, 0) + 1
            summary["by_severity"][f.severity] = summary["by_severity"].get(f.severity, 0) + 1
        
        report_data = {
            "summary": summary, 
            "findings": items,
            "job_info": {
                "id": job.id if job else job_id,
                "repo_url": job.repo_url if job else "Unknown",
                "status": job.status if job else "Unknown",
                "created_at": job.created_at.isoformat() if job and job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job and job.completed_at else None,
                "branch": job.base_branch if job else None,
                "pr_number": job.pr_number if job else None
            } if job else None
        }
        print(f"   Report summary: {summary}")
        return report_data
    except Exception as e:
        print(f"   Error fetching report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/analytics/download")
async def download_analytics_report(format: str = "json", session=Depends(get_session)):
    """Download comprehensive analytics report"""
    try:
        from ..api.analytics import get_trends, get_ml_insights, get_business_analytics, get_compliance_analytics, get_performance_analytics
        
        # Gather all analytics data
        trends = await get_trends(session)
        ml_insights = await get_ml_insights(session)
        business = await get_business_analytics(session)
        compliance = await get_compliance_analytics(session)
        performance = await get_performance_analytics(session)
        
        # Create comprehensive report
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trends": trends,
            "ml_insights": ml_insights,
            "business": business,
            "compliance": compliance,
            "performance": performance
        }
        
        if format.lower() == "json":
            return Response(
                content=json.dumps(report, indent=2, ensure_ascii=False),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=analytics-report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"}
            )
        
        elif format.lower() == "csv":
            # Create CSV analytics report
            output = StringIO()
            writer = csv.writer(output)
            
            # Write analytics summary
            writer.writerow(["Analytics Report", "Value"])
            writer.writerow(["Generated At", report["generated_at"]])
            writer.writerow(["Total Findings", trends["total_findings"]])
            writer.writerow(["Total Jobs", trends["total_jobs"]])
            writer.writerow(["Risk Level", ml_insights["risk_prediction"]])
            writer.writerow(["Compliance Score", f"{compliance['overall_score']}%"])
            writer.writerow(["Maintainability Index", performance["code_quality"]["maintainability_index"]])
            writer.writerow(["Technical Debt", performance["code_quality"]["technical_debt"]])
            writer.writerow(["Cost Savings", business["roi_metrics"]["cost_savings"]])
            writer.writerow(["Time Savings", business["roi_metrics"]["time_savings"]])
            writer.writerow(["Risk Reduction", business["roi_metrics"]["risk_reduction"]])
            
            # Add findings breakdown
            writer.writerow([])
            writer.writerow(["Findings by Severity"])
            for severity, count in trends["by_severity"].items():
                writer.writerow([severity.title(), count])
            
            writer.writerow([])
            writer.writerow(["Findings by Tool"])
            for tool, count in trends["by_tool"].items():
                writer.writerow([tool, count])
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analytics-report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.csv"}
            )
        
        else:
            raise HTTPException(400, "Unsupported format. Use: json or csv")
            
    except Exception as e:
        print(f"   Error downloading analytics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{job_id}/test-json")
async def test_json_serialization(job_id: str, session=Depends(get_session)):
    """Test endpoint to debug JSON serialization issues"""
    try:
        result = await session.execute(select(Finding).where(Finding.job_id==job_id))
        items = result.scalars().all()
        
        if not items:
            return {"error": "No findings found"}
        
        # Test with just one finding
        test_finding = items[0]
        test_dict = {
            "id": str(test_finding.id) if test_finding.id else None,
            "tool": str(test_finding.tool) if test_finding.tool else None,
            "severity": str(test_finding.severity) if test_finding.severity else None,
            "message": str(test_finding.message) if test_finding.message else None
        }
        
        return {"test_finding": test_dict, "serialization_test": "success"}
        
    except Exception as e:
        return {"error": str(e), "type": str(type(e))}

@router.get("/reports/{job_id}/download")
async def download_report(job_id: str, format: str = "json", session=Depends(get_session)):
    
    print(f"📥 Downloading report for job: {job_id} in {format} format")
    
    try:
        result = await session.execute(select(Finding).where(Finding.job_id==job_id))
        items = result.scalars().all()
        
        if not items:
            raise HTTPException(404, "report not found")
        
        # Get job information
        job_result = await session.execute(select(Job).where(Job.id==job_id))
        job = job_result.scalar_one_or_none()
        
        # Create summary
        summary = {"total": len(items), "by_tool": {}, "by_severity": {}}
        for f in items:
            summary["by_tool"][f.tool] = summary["by_tool"].get(f.tool, 0) + 1
            summary["by_severity"][f.severity] = summary["by_severity"].get(f.severity, 0) + 1
        
        # Convert SQLModel objects to dictionaries for proper JSON serialization
        findings_dicts = []
        for finding in items:
            # Safely convert all fields with proper null handling
            finding_dict = {
                "id": str(finding.id) if finding.id else None,
                "job_id": str(finding.job_id) if finding.job_id else None,
                "tool": str(finding.tool) if finding.tool else None,
                "severity": str(finding.severity) if finding.severity else None,
                "file": str(finding.file) if finding.file else None,
                "line": int(finding.line) if finding.line is not None else None,
                "rule_id": str(finding.rule_id) if finding.rule_id else None,
                "message": str(finding.message) if finding.message else None,
                "remediation": str(finding.remediation) if finding.remediation else None,
                "autofixable": bool(finding.autofixable) if finding.autofixable is not None else False,
                "vulnerability_type": str(finding.vulnerability_type) if finding.vulnerability_type else None,
                "code_snippet": str(finding.code_snippet) if finding.code_snippet else None,
                "pr_context": str(finding.pr_context) if finding.pr_context else None,
                "risk_score": float(finding.risk_score) if finding.risk_score is not None else None,
                "merge_blocking": bool(finding.merge_blocking) if finding.merge_blocking is not None else False,
                "test_coverage": str(finding.test_coverage) if finding.test_coverage else None,
                "breaking_change": bool(finding.breaking_change) if finding.breaking_change is not None else False,
                "created_at": finding.created_at.isoformat() if finding.created_at else None
            }
            findings_dicts.append(finding_dict)
        
        report_data = {"summary": summary, "findings": findings_dicts}
        
        if format.lower() == "json":
            # Debug logging to see what we're trying to serialize
            print(f"🔍 JSON Debug - Summary: {summary}")
            print(f"🔍 JSON Debug - First finding: {findings_dicts[0] if findings_dicts else 'No findings'}")
            
            try:
                json_content = json.dumps(report_data, indent=2, ensure_ascii=False)
                print(f"🔍 JSON Debug - Serialization successful, length: {len(json_content)}")
                return Response(
                    content=json_content,
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=report-{job_id}.json"}
                )
            except Exception as json_error:
                print(f"❌ JSON Serialization Error: {json_error}")
                print(f"❌ Error type: {type(json_error)}")
                # Fallback to a simpler format
                fallback_data = {
                    "summary": summary,
                    "findings_count": len(findings_dicts),
                    "error": "Original data could not be serialized, showing summary only"
                }
                return Response(
                    content=json.dumps(fallback_data, indent=2, ensure_ascii=False),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=report-{job_id}-fallback.json"}
                )
        
        elif format.lower() == "csv":
            # Convert findings to CSV with all fields from the CSV report
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header matching the CSV report format
            writer.writerow(["Tool", "Severity", "File", "Line", "Rule ID", "Message", "Remediation", "Autofixable"])
            
            # Write data with all fields
            for finding in items:
                writer.writerow([
                    finding.tool,
                    finding.severity,
                    finding.file,
                    finding.line,
                    finding.rule_id,
                    finding.message,
                    finding.remediation or "",
                    finding.autofixable
                ])
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=report-{job_id}.csv"}
            )
        
        elif format.lower() == "txt":
            # Create a comprehensive text report
            text_content = f"""
Code Review Report - Job {job_id}
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

Repository: {job.repo_url if job else 'Unknown'}
Status: {job.status if job else 'Unknown'}
Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S UTC') if job and job.created_at else 'Unknown'}

Summary:
- Total Issues: {summary['total']}
- By Tool: {summary['by_tool']}
- By Severity: {summary['by_severity']}

Detailed Findings:
"""
            
            for finding in items:
                text_content += f"""
Tool: {finding.tool}
Severity: {finding.severity}
File: {finding.file}
Line: {finding.line}
Rule: {finding.rule_id}
Message: {finding.message}
Remediation: {finding.remediation or 'None provided'}
Auto-fixable: {'Yes' if finding.autofixable else 'No'}
---
"""
            
            return Response(
                content=text_content,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=report-{job_id}.txt"}
            )
        
        else:
            raise HTTPException(400, "Unsupported format. Use: json, csv, or txt")
            
    except Exception as e:
        print(f"   Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{job_id}/findings")
async def get_job_findings(job_id: str, session=Depends(get_session)):
    """Get all findings for a specific job"""
    try:
        result = await session.execute(select(Finding).where(Finding.job_id==job_id))
        findings = result.scalars().all()
        
        if not findings:
            return {"findings": [], "total": 0}
        
        return {
            "findings": findings,
            "total": len(findings),
            "job_id": job_id
        }
        
    except Exception as e:
        print(f"   Error fetching findings for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{job_id}/detailed")
async def get_detailed_report(job_id: str, session=Depends(get_session)):
    """Get detailed report for a specific job including job info and findings"""
    try:
        # Get job information
        job_result = await session.execute(select(Job).where(Job.id==job_id))
        job = job_result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(404, "Job not found")
        
        # Get findings
        findings_result = await session.execute(select(Finding).where(Finding.job_id==job_id))
        findings = findings_result.scalars().all()
        
        # Create summary
        summary = {"total": len(findings), "by_tool": {}, "by_severity": {}}
        for f in findings:
            summary["by_tool"][f.tool] = summary["by_tool"].get(f.tool, 0) + 1
            summary["by_severity"][f.severity] = summary["by_severity"].get(f.severity, 0) + 1
        
        return {
            "job": {
                "id": job.id,
                "repo_url": job.repo_url,
                "status": job.status,
                "created_at": job.created_at,
                "completed_at": job.completed_at
            },
            "summary": summary,
            "findings": findings
        }
        
    except Exception as e:
        print(f"   Error fetching detailed report for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/analytics/download")
async def download_analytics_report(format: str = "json", session=Depends(get_session)):
    """Download comprehensive analytics report"""
    try:
        from ..api.analytics import get_trends, get_ml_insights, get_business_analytics, get_compliance_analytics, get_performance_analytics
        
        # Gather all analytics data
        trends = await get_trends(session)
        ml_insights = await get_ml_insights(session)
        business = await get_business_analytics(session)
        compliance = await get_compliance_analytics(session)
        performance = await get_performance_analytics(session)
        
        # Create comprehensive report
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trends": trends,
            "ml_insights": ml_insights,
            "business": business,
            "compliance": compliance,
            "performance": performance
        }
        
        if format.lower() == "json":
            return Response(
                content=json.dumps(report, indent=2, ensure_ascii=False),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=analytics-report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"}
            )
        
        elif format.lower() == "csv":
            # Create CSV analytics report
            output = StringIO()
            writer = csv.writer(output)
            
            # Write analytics summary
            writer.writerow(["Analytics Report", "Value"])
            writer.writerow(["Generated At", report["generated_at"]])
            writer.writerow(["Total Findings", trends["total_findings"]])
            writer.writerow(["Total Jobs", trends["total_jobs"]])
            writer.writerow(["Risk Level", ml_insights["risk_prediction"]])
            writer.writerow(["Compliance Score", f"{compliance['overall_score']}%"])
            writer.writerow(["Maintainability Index", performance["code_quality"]["maintainability_index"]])
            writer.writerow(["Technical Debt", performance["code_quality"]["technical_debt"]])
            writer.writerow(["Cost Savings", business["roi_metrics"]["cost_savings"]])
            writer.writerow(["Time Savings", business["roi_metrics"]["time_savings"]])
            writer.writerow(["Risk Reduction", business["roi_metrics"]["risk_reduction"]])
            
            # Add findings breakdown
            writer.writerow([])
            writer.writerow(["Findings by Severity"])
            for severity, count in trends["by_severity"].items():
                writer.writerow([severity.title(), count])
            
            writer.writerow([])
            writer.writerow(["Findings by Tool"])
            for tool, count in trends["by_tool"].items():
                writer.writerow([tool, count])
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analytics-report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.csv"}
            )
        
        else:
            raise HTTPException(400, "Unsupported format. Use: json or csv")
            
    except Exception as e:
        print(f"   Error downloading analytics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
