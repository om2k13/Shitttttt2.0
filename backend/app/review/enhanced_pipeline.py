import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from sqlmodel import select
from ..core.vcs import clone_repo, run_cmd, cleanup_repo_after_review
from ..core.llm import enrich_findings_with_llm
from ..db.models import Job, Finding
from ..db.base import get_session
from .code_review_agent import CodeReviewAgent
from .pipeline import run_review as run_basic_review
from .test_generator import TestGenerator

class EnhancedPipeline:
    """
    Enhanced pipeline that integrates Code Review Agent with the existing pipeline.
    This ensures proper flow: Code Scanning -> Code Review -> Testing phase.
    """
    
    def __init__(self):
        self.code_review_agent = None
        self.current_job_id = None
        self.repo_path = None
    
    async def run_enhanced_review(self, job_id: str, include_code_review: bool = True) -> Dict[str, Any]:
        """
        Run enhanced review pipeline with optional code review integration
        
        Args:
            job_id: The job ID to process
            include_code_review: Whether to include code review analysis
            
        Returns:
            Dictionary containing comprehensive review results
        """
        print(f"🚀 Starting Enhanced Review Pipeline for job: {job_id}")
        self.current_job_id = job_id
        
        try:
            # Step 1: Run basic security and vulnerability scanning
            print("🔍 Step 1: Running security and vulnerability scanning...")
            basic_results = await self._run_basic_scanning(job_id)
            
            if not basic_results.get("success", False):
                return {
                    "status": "error",
                    "error": "Basic scanning failed",
                    "details": basic_results
                }
            
            # Step 2: Run Code Review Agent (if enabled)
            code_review_results = None
            if include_code_review:
                print("🔍 Step 2: Running Code Review Agent...")
                code_review_results = await self._run_code_review_agent(job_id)
            
            # Step 3: Generate comprehensive report
            print("📊 Step 3: Generating comprehensive report...")
            comprehensive_report = await self._generate_comprehensive_report(
                basic_results, 
                code_review_results
            )
            
            # Step 4: Generate test plan based on findings
            print("🧪 Step 4: Generating test plan...")
            test_plan = await self._generate_test_plan(comprehensive_report)
            comprehensive_report["test_plan"] = test_plan
            
            # Step 5: Store enhanced findings in database
            print("💾 Step 5: Storing enhanced findings...")
            await self._store_enhanced_findings(job_id, comprehensive_report)
            
            print("✅ Enhanced Review Pipeline completed successfully!")
            return comprehensive_report
            
        except Exception as e:
            print(f"❌ Error in enhanced review pipeline: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "findings": []
            }

async def run_enhanced_review(job_id: str, include_code_review: bool = True) -> Dict[str, Any]:
    """
    Standalone function to run enhanced review pipeline
    
    Args:
        job_id: The job ID to process
        include_code_review: Whether to include code review analysis
        
    Returns:
        Dictionary containing comprehensive review results
    """
    pipeline = EnhancedPipeline()
    return await pipeline.run_enhanced_review(job_id, include_code_review)

async def run_standalone_code_review(repo_path: str) -> Dict[str, Any]:
    """
    Run standalone code review analysis on a local repository
    
    Args:
        repo_path: Path to the repository to analyze
        
    Returns:
        Dictionary containing code review results
    """
    pipeline = EnhancedPipeline()
    pipeline.repo_path = Path(repo_path)
    
    # Run only code review analysis
    code_review_results = await pipeline._run_code_review_agent("standalone")
    
    if code_review_results:
        return {
            "status": "completed",
            "findings": code_review_results.get("findings", []),
            "summary": code_review_results.get("summary", {}),
            "metadata": {
                "analysis_type": "standalone_code_review",
                "repo_path": repo_path
            }
        }
    else:
        return {
            "status": "error",
            "error": "Code review analysis failed",
            "findings": []
        }
    
    async def _run_basic_scanning(self, job_id: str) -> Dict[str, Any]:
        """Run the basic security and vulnerability scanning"""
        try:
            # Get job details
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if not job:
                    return {"success": False, "error": f"Job {job_id} not found"}
                
                # Update job status
                job.status = "running"
                job.current_stage = "security_scanning"
                job.progress = 10
                await session.commit()
            
            # Clone repository if not already done
            if not self.repo_path:
                self.repo_path = await self._clone_repository(job)
                if not self.repo_path:
                    return {"success": False, "error": "Failed to clone repository"}
            
            # Run basic security scanning
            basic_findings = await self._run_security_tools()
            
            # Store results for Code Review Agent to use
            self._last_security_results = {
                "success": True,
                "findings": basic_findings,
                "repo_path": str(self.repo_path)
            }
            
            return self._last_security_results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_code_review_agent(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Run the Code Review Agent"""
        try:
            # Update job status
            async with get_session() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.current_stage = "code_review"
                    job.progress = 50
                    await session.commit()
            
            # Initialize Code Review Agent
            self.code_review_agent = CodeReviewAgent(
                repo_path=self.repo_path,
                standalone=False  # Integrated mode
            )
            
            # Get security findings from previous stage to pass to Code Review Agent
            security_findings = []
            if hasattr(self, '_last_security_results') and self._last_security_results:
                security_findings = self._last_security_results.get("findings", [])
            
            # Run code review analysis with security findings as input
            code_review_results = await self.code_review_agent.run_code_review(
                input_data={"security_findings": security_findings}
            )
            
            return code_review_results
            
        except Exception as e:
            print(f"Warning: Code Review Agent failed: {e}")
            return None
    
    async def _run_security_tools(self) -> List[Dict]:
        """Run security and vulnerability scanning tools"""
        findings = []
        
        try:
            # Run Bandit (Python security)
            print("   Running Bandit security analysis...")
            rc, out, err = run_cmd(["bandit", "-q", "-r", ".", "-f", "json"], self.repo_path)
            if rc == 0 and out.strip():
                try:
                    data = json.loads(out)
                    for issue in data.get("results", []):
                        findings.append({
                            "tool": "bandit",
                            "severity": self._map_severity(issue.get("issue_severity", "LOW")),
                            "file": issue.get("filename"),
                            "line": issue.get("line_number"),
                            "rule_id": issue.get("test_id"),
                            "message": issue.get("issue_text"),
                            "category": "security",
                            "stage": "security_scanning"
                        })
                except Exception as e:
                    print(f"   Warning: Could not parse Bandit output: {e}")
            
            # Run Semgrep
            print("   Running Semgrep analysis...")
            rc, out, err = run_cmd(["semgrep", "--quiet", "--json", "--error", "--timeout", "0", "--config", "p/ci"], self.repo_path)
            if out.strip():
                try:
                    data = json.loads(out)
                    for r in data.get("results", []):
                        findings.append({
                            "tool": "semgrep",
                            "severity": self._map_severity(r.get("extra", {}).get("severity", "LOW")),
                            "file": r.get("path"),
                            "line": r.get("start", {}).get("line"),
                            "rule_id": r.get("check_id"),
                            "message": r.get("extra", {}).get("message"),
                            "category": "security",
                            "stage": "security_scanning"
                        })
                except Exception:
                    pass
            
            # Run dependency vulnerability checks
            await self._run_dependency_checks(findings)
            
            # Run code quality tools
            await self._run_quality_tools(findings)
            
        except Exception as e:
            print(f"Warning: Error running security tools: {e}")
        
        return findings
    
    async def _run_dependency_checks(self, findings: List[Dict]):
        """Run dependency vulnerability checks"""
        try:
            # Python dependencies
            req_file = None
            for cand in ["requirements.txt", "backend/requirements.txt"]:
                if (self.repo_path / cand).exists():
                    req_file = cand
                    break
            
            if req_file:
                print("   Running pip-audit...")
                rc, out, err = run_cmd(["pip-audit", "-r", req_file, "--format", "json"], self.repo_path)
                if out.strip():
                    try:
                        data = json.loads(out)
                        for v in data.get("dependencies", []):
                            name = v.get("name")
                            version = v.get("version")
                            for adv in v.get("vulns", []):
                                findings.append({
                                    "tool": "pip-audit",
                                    "severity": self._map_severity(adv.get("severity", "MEDIUM")),
                                    "file": req_file,
                                    "line": None,
                                    "rule_id": ",".join(adv.get("aliases", [])) or adv.get("id"),
                                    "message": f"{name}=={version}: {adv.get('description', 'vulnerability')}",
                                    "category": "dependency",
                                    "stage": "security_scanning",
                                    "autofixable": True
                                })
                    except Exception:
                        pass
            
            # Node.js dependencies
            if (self.repo_path / "package.json").exists():
                print("   Running npm audit...")
                rc, out, err = run_cmd(["npm", "audit", "--json"], self.repo_path)
                if out.strip():
                    try:
                        data = json.loads(out)
                        advisories = data.get("vulnerabilities", {})
                        for name, v in advisories.items():
                            findings.append({
                                "tool": "npm-audit",
                                "severity": self._map_severity(v.get("severity", "low")),
                                "file": "package.json",
                                "line": None,
                                "rule_id": name,
                                "message": f"{name}: {v.get('title', 'vulnerability')}",
                                "category": "dependency",
                                "stage": "security_scanning"
                            })
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"Warning: Error running dependency checks: {e}")
    
    async def _run_quality_tools(self, findings: List[Dict]):
        """Run code quality analysis tools"""
        try:
            # Python code quality
            if self.repo_path.glob("*.py"):
                print("   Running ruff analysis...")
                rc, out, err = run_cmd(["ruff", "check", "--output-format", "json", "."], self.repo_path)
                if out.strip():
                    try:
                        data = json.loads(out)
                        for item in data:
                            findings.append({
                                "tool": "ruff",
                                "severity": "low",
                                "file": item.get("filename"),
                                "line": item.get("location", {}).get("row"),
                                "rule_id": item.get("code"),
                                "message": item.get("message"),
                                "category": "quality",
                                "stage": "security_scanning",
                                "autofixable": True
                            })
                    except Exception as e:
                        print(f"   Warning: Could not parse ruff output: {e}")
                
                # Run complexity analysis
                print("   Running complexity analysis...")
                rc, out, err = run_cmd(["radon", "cc", "-j", "."], self.repo_path)
                if out.strip():
                    try:
                        data = json.loads(out)
                        for file, entries in data.items():
                            for ent in entries:
                                if ent.get("rank") in ("D", "E", "F"):
                                    findings.append({
                                        "tool": "radon",
                                        "severity": "medium" if ent.get("rank") == "D" else "high",
                                        "file": file,
                                        "line": ent.get("lineno"),
                                        "rule_id": f"complexity-{ent.get('rank')}",
                                        "message": f"Cyclomatic complexity {ent.get('complexity')} (rank {ent.get('rank')})",
                                        "category": "complexity",
                                        "stage": "security_scanning"
                                    })
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"Warning: Error running quality tools: {e}")
    
    async def _generate_test_plan(self, comprehensive_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test plan based on findings from security scanning and code review"""
        try:
            print("🧪 Generating test plan based on findings...")
            
            # Initialize Test Generator
            test_generator = TestGenerator(self.repo_path)
            
            # Get all findings
            all_findings = comprehensive_report.get("findings", [])
            
            # Group findings by file for test generation
            files_with_findings = {}
            for finding in all_findings:
                file_path = finding.get("file")
                if file_path:
                    if file_path not in files_with_findings:
                        files_with_findings[file_path] = []
                    files_with_findings[file_path].append(finding)
            
            # Generate test plan for files with findings
            test_plan = {
                "total_files": len(files_with_findings),
                "test_files": [],
                "coverage_analysis": {},
                "priority_tests": []
            }
            
            for file_path, findings in files_with_findings.items():
                # Create file info for test generation
                file_info = {
                    "path": file_path,
                    "type": self._get_file_type(file_path),
                    "findings": findings
                }
                
                # Generate tests for this file
                file_tests = await test_generator.generate_test_plan(
                    changed_files=[file_info],
                    findings=findings
                )
                
                if file_tests:
                    test_plan["test_files"].append(file_tests)
                    
                    # Identify high-priority tests based on critical findings
                    critical_findings = [f for f in findings if f.get("severity") in ["critical", "high"]]
                    if critical_findings:
                        test_plan["priority_tests"].extend([
                            {
                                "file": file_path,
                                "priority": "high",
                                "reason": f"Critical findings: {len(critical_findings)}",
                                "findings": critical_findings
                            }
                        ])
            
            # Add coverage analysis
            if test_plan["test_files"]:
                test_plan["coverage_analysis"] = await test_generator.analyze_test_coverage(
                    changed_files=list(files_with_findings.keys()),
                    findings=all_findings
                )
            
            print(f"✅ Test plan generated: {len(test_plan['test_files'])} test files")
            return test_plan
            
        except Exception as e:
            print(f"Warning: Test plan generation failed: {e}")
            return {
                "error": str(e),
                "total_files": 0,
                "test_files": [],
                "coverage_analysis": {},
                "priority_tests": []
            }
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type based on extension"""
        if file_path.endswith('.py'):
            return "python"
        elif file_path.endswith(('.js', '.ts')):
            return "javascript"
        elif file_path.endswith('.java'):
            return "java"
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return "yaml"
        elif file_path.endswith('.json'):
            return "json"
        else:
            return "unknown"
    
    async def _generate_comprehensive_report(self, basic_results: Dict, code_review_results: Optional[Dict]) -> Dict[str, Any]:
        """Generate comprehensive report combining all findings"""
        print("📊 Generating comprehensive report...")
        
        # Combine findings from both stages
        all_findings = []
        
        # Add basic security findings
        if basic_results.get("findings"):
            for finding in basic_results["findings"]:
                all_findings.append({
                    "stage": "security_scanning",
                    "tool": finding.get("tool", "unknown"),
                    "severity": finding.get("severity", "low"),
                    "file": finding.get("file"),
                    "line": finding.get("line"),
                    "rule_id": finding.get("rule_id"),
                    "message": finding.get("message"),
                    "category": finding.get("category", "unknown"),
                    "remediation": finding.get("remediation"),
                    "autofixable": finding.get("autofixable", False),
                    "code_snippet": None  # Will be enriched later
                })
        
        # Add code review findings
        if code_review_results and code_review_results.get("findings"):
            for finding in code_review_results["findings"]:
                all_findings.append({
                    "stage": "code_review",
                    "tool": "code_review_agent",
                    "severity": finding.get("severity", "low"),
                    "file": finding.get("file"),
                    "line": finding.get("line"),
                    "rule_id": finding.get("category", "unknown"),
                    "message": finding.get("message"),
                    "category": finding.get("category", "unknown"),
                    "remediation": finding.get("suggestion"),
                    "autofixable": finding.get("autofixable", False),
                    "code_snippet": finding.get("code_snippet")
                })
        
        # Calculate statistics
        total_findings = len(all_findings)
        findings_by_stage = {}
        findings_by_category = {}
        findings_by_severity = {}
        
        for finding in all_findings:
            # Count by stage
            stage = finding.get("stage", "unknown")
            if stage not in findings_by_stage:
                findings_by_stage[stage] = 0
            findings_by_stage[stage] += 1
            
            # Count by category
            category = finding.get("category", "unknown")
            if category not in findings_by_category:
                findings_by_category[category] = 0
            findings_by_category[category] += 1
            
            # Count by severity
            severity = finding.get("severity", "low")
            if severity not in findings_by_severity:
                findings_by_severity[severity] = 0
            findings_by_severity[severity] += 1
        
        # Generate comprehensive report
        comprehensive_report = {
            "status": "completed",
            "pipeline_stages": ["security_scanning", "code_review"],
            "total_findings": total_findings,
            "findings_by_stage": findings_by_stage,
            "findings_by_category": findings_by_category,
            "findings_by_severity": findings_by_severity,
            "findings": all_findings,
            "summary": {
                "security_issues": findings_by_category.get("security", 0),
                "dependency_vulnerabilities": findings_by_category.get("dependency", 0),
                "code_quality_issues": findings_by_category.get("quality", 0),
                "complexity_issues": findings_by_category.get("complexity", 0),
                "refactoring_opportunities": findings_by_category.get("refactoring", 0),
                "reusability_improvements": findings_by_category.get("reusability", 0),
                "efficiency_gains": findings_by_category.get("efficiency", 0),
                "critical_issues": findings_by_severity.get("critical", 0),
                "high_priority": findings_by_severity.get("high", 0),
                "medium_priority": findings_by_severity.get("medium", 0),
                "low_priority": findings_by_severity.get("low", 0)
            },
            "metadata": {
                "repo_path": str(self.repo_path),
                "tools_used": list(set(f.get("tool") for f in all_findings if f.get("tool"))),
                "stages_completed": ["security_scanning"] + (["code_review"] if code_review_results else [])
            }
        }
        
        return comprehensive_report
    
    async def _store_enhanced_findings(self, job_id: str, report: Dict[str, Any]):
        """Store enhanced findings in the database"""
        try:
            async with get_session() as session:
                # Update job status
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.current_stage = "completed"
                    job.progress = 100
                    job.findings_count = report.get("total_findings", 0)
                    
                    # Store severity breakdown
                    severity_breakdown = report.get("findings_by_severity", {})
                    job.severity_breakdown = json.dumps(severity_breakdown)
                    
                    # Store tools used
                    tools_used = report.get("metadata", {}).get("tools_used", [])
                    job.tools_used = json.dumps(tools_used)
                    
                    await session.commit()
                
                # Store individual findings
                for finding_data in report.get("findings", []):
                    finding = Finding(
                        job_id=job_id,
                        tool=finding_data.get("tool", "unknown"),
                        severity=finding_data.get("severity", "low"),
                        file=finding_data.get("file", ""),
                        line=finding_data.get("line"),
                        rule_id=finding_data.get("rule_id", ""),
                        message=finding_data.get("message", ""),
                        remediation=finding_data.get("remediation"),
                        autofixable=finding_data.get("autofixable", False),
                        vulnerability_type=finding_data.get("category"),
                        code_snippet=finding_data.get("code_snippet")
                    )
                    session.add(finding)
                
                await session.commit()
                print(f"💾 Stored {len(report.get('findings', []))} findings in database")
                
        except Exception as e:
            print(f"Warning: Could not store enhanced findings: {e}")
    
    async def _clone_repository(self, job) -> Optional[Path]:
        """Clone the repository for analysis"""
        try:
            print(f"🔗 Cloning repo: {job.repo_url} (branch: {job.base_branch})")
            repo_path = clone_repo(job.id, job.repo_url, job.base_branch)
            print(f"📁 Repo cloned to: {repo_path}")
            return repo_path
        except Exception as e:
            print(f"❌ Failed to clone repository: {e}")
            return None
    
    def _map_severity(self, level: str) -> str:
        """Map tool-specific severity levels to standard levels"""
        s = (level or "").lower()
        if s in ("critical", "high", "medium", "low"):
            return s
        if s in ("error", "warn", "warning"):
            return "medium"
        return "low"
    
    async def export_comprehensive_report(self, job_id: str, output_path: str, format: str = "json"):
        """
        Export comprehensive report to file
        
        Args:
            job_id: Job ID to export
            output_path: Path to output file
            format: Output format ("json" or "markdown")
        """
        try:
            # Get the comprehensive report
            report = await self._get_job_report(job_id)
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
            elif format.lower() == "markdown":
                await self._export_to_markdown(report, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"📄 Comprehensive report exported to {output_path}")
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
    
    async def _get_job_report(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive report for a specific job"""
        # This would typically query the database for stored results
        # For now, return a placeholder
        return {
            "status": "not_found",
            "error": "Job report not found. Run the enhanced review first."
        }
    
    async def _export_to_markdown(self, report: Dict[str, Any], output_path: str):
        """Export report to Markdown format"""
        with open(output_path, 'w') as f:
            f.write("# Enhanced Code Review Report\n\n")
            f.write(f"**Total Findings:** {report.get('total_findings', 0)}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            summary = report.get('summary', {})
            f.write(f"- Security Issues: {summary.get('security_issues', 0)}\n")
            f.write(f"- Dependency Vulnerabilities: {summary.get('dependency_vulnerabilities', 0)}\n")
            f.write(f"- Code Quality Issues: {summary.get('code_quality_issues', 0)}\n")
            f.write(f"- Complexity Issues: {summary.get('complexity_issues', 0)}\n")
            f.write(f"- Refactoring Opportunities: {summary.get('refactoring_opportunities', 0)}\n")
            f.write(f"- Reusability Improvements: {summary.get('reusability_improvements', 0)}\n")
            f.write(f"- Efficiency Gains: {summary.get('efficiency_gains', 0)}\n\n")
            
            # Findings by stage
            f.write("## Findings by Stage\n\n")
            for stage, count in report.get('findings_by_stage', {}).items():
                f.write(f"- **{stage.replace('_', ' ').title()}:** {count} findings\n")
            f.write("\n")
            
            # Detailed findings
            f.write("## Detailed Findings\n\n")
            for finding in report.get('findings', []):
                f.write(f"### {finding.get('file', 'Unknown')}:{finding.get('line', 'N/A')}\n")
                f.write(f"**Stage:** {finding.get('stage', 'unknown').replace('_', ' ').title()}\n")
                f.write(f"**Tool:** {finding.get('tool', 'unknown')}\n")
                f.write(f"**Category:** {finding.get('category', 'unknown')}\n")
                f.write(f"**Severity:** {finding.get('severity', 'unknown')}\n")
                f.write(f"**Message:** {finding.get('message', 'No message')}\n")
                if finding.get('remediation'):
                    f.write(f"**Remediation:** {finding.get('remediation')}\n")
                if finding.get('code_snippet'):
                    f.write(f"**Code:**\n```\n{finding.get('code_snippet')}\n```\n")
                f.write("\n")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.repo_path and self.repo_path.exists():
                # Clean up repository workspace
                cleanup_repo_after_review(str(self.repo_path))
                print(f"🧹 Cleaned up repository workspace: {self.repo_path}")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
