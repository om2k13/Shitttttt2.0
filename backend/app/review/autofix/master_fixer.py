from pathlib import Path
from typing import List, Dict, Optional
from .security_fixes import SecurityFixer
from .code_quality_fixes import CodeQualityFixer
from .type_fixes import TypeCheckingFixer
from .safe_fixes import run_python_formatters, run_js_formatters

class MasterAutoFixer:
    """Orchestrates all types of automatic fixes for code review findings"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.security_fixer = SecurityFixer(repo_path)
        self.code_quality_fixer = CodeQualityFixer(repo_path)
        self.type_fixer = TypeCheckingFixer(repo_path)
    
    async def apply_all_fixes(self, findings: List[Dict]) -> Dict:
        """Apply all available automatic fixes"""
        results = {
            "security_fixes": {},
            "code_quality_fixes": {},
            "type_fixes": {},
            "formatting_fixes": {},
            "summary": {
                "total_findings": len(findings),
                "autofixable_findings": len([f for f in findings if f.get('autofixable', False)]),
                "total_fixes_applied": 0,
                "successful_fixes": 0,
                "failed_fixes": 0
            }
        }
        
        # Apply security fixes
        print("🔒 Applying security fixes...")
        security_results = await self.security_fixer.apply_all_security_fixes(findings)
        results["security_fixes"] = security_results
        
        # Apply code quality fixes
        print("✨ Applying code quality fixes...")
        quality_results = await self.code_quality_fixer.apply_all_code_quality_fixes(findings)
        results["code_quality_fixes"] = quality_results
        
        # Apply type checking fixes
        print("🔍 Applying type checking fixes...")
        type_results = await self.type_fixer.apply_all_type_fixes(findings)
        results["type_fixes"] = type_results
        
        # Apply formatting fixes
        print("🎨 Applying formatting fixes...")
        formatting_results = await self._apply_formatting_fixes()
        results["formatting_fixes"] = formatting_results
        
        # Calculate summary
        total_fixes = (
            security_results.get('successful_fixes', 0) +
            quality_results.get('successful_fixes', 0) +
            type_results.get('successful_fixes', 0) +
            formatting_results.get('successful_fixes', 0)
        )
        
        results["summary"]["total_fixes_applied"] = total_fixes
        results["summary"]["successful_fixes"] = total_fixes
        
        return results
    
    async def _apply_formatting_fixes(self) -> Dict:
        """Apply code formatting fixes"""
        try:
            results = {
                "python_formatting": {},
                "javascript_formatting": {},
                "successful_fixes": 0
            }
            
            # Python formatting
            if any(self.repo_path.rglob("*.py")):
                python_results = run_python_formatters(self.repo_path)
                results["python_formatting"] = {
                    "ruff_fixes": python_results[0] if len(python_results) > 0 else None,
                    "black_fixes": python_results[1] if len(python_results) > 1 else None
                }
                
                # Count successful fixes
                for tool, rc, output in python_results:
                    if rc == 0:
                        results["successful_fixes"] += 1
            
            # JavaScript formatting
            if (self.repo_path / "package.json").exists():
                js_results = run_js_formatters(self.repo_path)
                results["javascript_formatting"] = {
                    "eslint_fixes": js_results[0] if len(js_results) > 0 else None,
                    "prettier_fixes": js_results[1] if len(js_results) > 1 else None
                }
                
                # Count successful fixes
                for tool, rc, output in js_results:
                    if rc == 0:
                        results["successful_fixes"] += 1
            
            return results
            
        except Exception as e:
            return {
                "error": f"Failed to apply formatting fixes: {str(e)}",
                "successful_fixes": 0
            }
    
    async def fix_specific_finding(self, finding_id: str, findings: List[Dict]) -> Dict:
        """Apply fixes for a specific finding"""
        finding = next((f for f in findings if f.get('id') == finding_id), None)
        if not finding:
            return {"error": "Finding not found"}
        
        if not finding.get('autofixable', False):
            return {"error": "Finding is not auto-fixable"}
        
        tool = finding.get('tool', '')
        
        try:
            if tool == 'pip-audit':
                # Security fix
                fixes = await self.security_fixer.fix_python_dependencies([finding])
                return {
                    "finding_id": finding_id,
                    "fix_type": "security",
                    "result": fixes[0] if fixes else {"error": "Fix failed"}
                }
            
            elif tool == 'ruff':
                # Code quality fix
                fixes = await self.code_quality_fixer.fix_python_code_quality([finding])
                return {
                    "finding_id": finding_id,
                    "fix_type": "code_quality",
                    "result": fixes[0] if fixes else {"error": "Fix failed"}
                }
            
            elif tool == 'mypy':
                # Type checking fix
                fixes = await self.type_fixer.fix_mypy_issues([finding])
                return {
                    "finding_id": finding_id,
                    "fix_type": "type_checking",
                    "result": fixes[0] if fixes else {"error": "Fix failed"}
                }
            
            else:
                return {"error": f"Unsupported tool: {tool}"}
                
        except Exception as e:
            return {"error": f"Failed to apply fix: {str(e)}"}
    
    async def get_fix_summary(self, findings: List[Dict]) -> Dict:
        """Get a summary of what fixes can be applied"""
        autofixable_findings = [f for f in findings if f.get('autofixable', False)]
        
        summary = {
            "total_findings": len(findings),
            "autofixable_findings": len(autofixable_findings),
            "fix_categories": {
                "security": {
                    "count": len([f for f in autofixable_findings if f.get('tool') == 'pip-audit']),
                    "description": "Dependency vulnerability fixes"
                },
                "code_quality": {
                    "count": len([f for f in autofixable_findings if f.get('tool') == 'ruff']),
                    "description": "Code style and quality fixes"
                },
                "type_checking": {
                    "count": len([f for f in autofixable_findings if f.get('tool') == 'mypy']),
                    "description": "Type annotation and import fixes"
                }
            },
            "estimated_time": "2-5 minutes",
            "risk_level": "Low (automated fixes only)"
        }
        
        return summary
    
    async def validate_fixes(self) -> Dict:
        """Validate that applied fixes don't break the codebase"""
        results = {
            "python_tests": {},
            "javascript_tests": {},
            "overall_status": "unknown"
        }
        
        try:
            # Test Python code
            if any(self.repo_path.rglob("*.py")):
                # Run basic syntax check
                rc, output, error = await self._run_python_syntax_check()
                results["python_tests"] = {
                    "syntax_check": rc == 0,
                    "output": output or error
                }
            
            # Test JavaScript code
            if (self.repo_path / "package.json").exists():
                # Run basic syntax check
                rc, output, error = await self._run_javascript_syntax_check()
                results["javascript_tests"] = {
                    "syntax_check": rc == 0,
                    "output": output or error
                }
            
            # Determine overall status
            all_passed = all([
                results.get("python_tests", {}).get("syntax_check", True),
                results.get("javascript_tests", {}).get("syntax_check", True)
            ])
            
            results["overall_status"] = "passed" if all_passed else "failed"
            
            return results
            
        except Exception as e:
            return {
                "error": f"Validation failed: {str(e)}",
                "overall_status": "error"
            }
    
    async def _run_python_syntax_check(self) -> tuple:
        """Run Python syntax check"""
        try:
            from ...core.vcs import run_cmd
            rc, output, error = run_cmd(["python", "-m", "py_compile", "."], self.repo_path)
            return rc, output, error
        except Exception as e:
            return -1, "", str(e)
    
    async def _run_javascript_syntax_check(self) -> tuple:
        """Run JavaScript syntax check"""
        try:
            from ...core.vcs import run_cmd
            rc, output, error = run_cmd(["npx", "eslint", "--no-eslintrc", "--env", "browser", "."], self.repo_path)
            return rc, output, error
        except Exception as e:
            return -1, "", str(e)
