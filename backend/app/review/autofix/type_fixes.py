import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ...core.vcs import run_cmd

class TypeCheckingFixer:
    """Automatically fixes type checking issues like missing stubs and import errors"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.python_files = list(self.repo_path.rglob("*.py"))
    
    async def fix_mypy_issues(self, findings: List[Dict]) -> List[Dict]:
        """Fix mypy type checking issues"""
        fixes_applied = []
        
        for finding in findings:
            if finding.get('tool') == 'mypy' and finding.get('autofixable'):
                fix_result = await self._fix_mypy_issue(finding)
                if fix_result:
                    fixes_applied.append(fix_result)
        
        return fixes_applied
    
    async def _fix_mypy_issue(self, finding: Dict) -> Optional[Dict]:
        """Fix a specific mypy issue"""
        try:
            message = finding.get('message', '')
            file_path = finding.get('file', '')
            
            if not file_path:
                return None
            
            # Convert relative path to absolute
            if file_path.startswith('/'):
                abs_file_path = Path(file_path)
            else:
                abs_file_path = self.repo_path / file_path
            
            if not abs_file_path.exists():
                return None
            
            # Handle different types of mypy errors
            if "Cannot find implementation or library stub" in message:
                fix_result = await self._fix_missing_stub(abs_file_path, message)
            elif "error: Cannot find module" in message:
                fix_result = await self._fix_import_error(abs_file_path, message)
            elif "error: Incompatible types" in message:
                fix_result = await self._fix_type_annotation(abs_file_path, message)
            else:
                # Try generic mypy fix
                fix_result = await self._run_mypy_fix(abs_file_path)
            
            if fix_result:
                return {
                    "finding_id": finding.get('id'),
                    "message": message,
                    "file": str(file_path),
                    "fix_applied": True,
                    "fix_type": "mypy",
                    "details": fix_result
                }
            
        except Exception as e:
            return {
                "finding_id": finding.get('id'),
                "error": str(e),
                "fix_applied": False
            }
        
        return None
    
    async def _fix_missing_stub(self, file_path: Path, message: str) -> Dict:
        """Fix missing library stubs by installing type stubs"""
        try:
            # Extract module name from error message
            module_match = re.search(r'Cannot find implementation or library stub for module ([a-zA-Z0-9_.]+)', message)
            if not module_match:
                return {"error": "Could not extract module name"}
            
            module_name = module_match.group(1)
            
            # Try to install type stubs
            stub_package = f"types-{module_name}"
            
            # Check if virtual environment exists
            venv_path = self.repo_path / ".venv"
            if venv_path.exists():
                if self.repo_path.name == "windows":
                    pip_path = venv_path / "Scripts" / "pip.exe"
                else:
                    pip_path = venv_path / "bin" / "pip"
                
                # Try to install type stubs
                rc, output, error = run_cmd([str(pip_path), "install", stub_package], self.repo_path)
                
                if rc == 0:
                    return {
                        "action": "installed_type_stubs",
                        "package": stub_package,
                        "success": True,
                        "output": output
                    }
                else:
                    # Try alternative stub package names
                    alt_stubs = [
                        f"types-{module_name.replace('_', '-')}",
                        f"{module_name}-stubs",
                        f"stubs-{module_name}"
                    ]
                    
                    for alt_stub in alt_stubs:
                        rc, output, error = run_cmd([str(pip_path), "install", alt_stub], self.repo_path)
                        if rc == 0:
                            return {
                                "action": "installed_alternative_type_stubs",
                                "package": alt_stub,
                                "success": True,
                                "output": output
                            }
            
            return {
                "action": "type_stub_installation_failed",
                "module": module_name,
                "success": False,
                "error": "Could not install type stubs"
            }
            
        except Exception as e:
            return {"error": f"Failed to fix missing stub: {str(e)}"}
    
    async def _fix_import_error(self, file_path: Path, message: str) -> Dict:
        """Fix import errors by adding proper import statements"""
        try:
            # Extract module name from error message
            module_match = re.search(r'Cannot find module ([a-zA-Z0-9_.]+)', message)
            if not module_match:
                return {"error": "Could not extract module name"}
            
            module_name = module_match.group(1)
            
            # Read file content
            content = file_path.read_text()
            lines = content.split('\n')
            
            # Check if it's a relative import issue
            if module_name.startswith('.'):
                # Try to fix relative import
                fixed_content = await self._fix_relative_import(content, module_name)
                if fixed_content != content:
                    file_path.write_text(fixed_content)
                    return {
                        "action": "fixed_relative_import",
                        "module": module_name,
                        "success": True
                    }
            
            # Check if it's a missing __init__.py issue
            if await self._create_missing_init_files(module_name):
                return {
                    "action": "created_missing_init_files",
                    "module": module_name,
                    "success": True
                }
            
            return {
                "action": "import_error_fix_failed",
                "module": module_name,
                "success": False,
                "error": "Could not resolve import"
            }
            
        except Exception as e:
            return {"error": f"Failed to fix import error: {str(e)}"}
    
    async def _fix_relative_import(self, content: str, module_name: str) -> str:
        """Fix relative import issues"""
        try:
            # Simple relative import fix - convert to absolute if possible
            if module_name.startswith('.'):
                # Try to find the actual module in the project
                # This is a simplified approach
                return content
            
            return content
            
        except Exception:
            return content
    
    async def _create_missing_init_files(self, module_name: str) -> bool:
        """Create missing __init__.py files for packages"""
        try:
            # Split module name to find package structure
            parts = module_name.split('.')
            
            for i in range(len(parts)):
                package_path = self.repo_path / '/'.join(parts[:i+1])
                init_file = package_path / "__init__.py"
                
                if package_path.exists() and not init_file.exists():
                    init_file.write_text("# Auto-generated __init__.py file\n")
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _fix_type_annotation(self, file_path: Path, message: str) -> Dict:
        """Fix type annotation issues"""
        try:
            # Read file content
            content = file_path.read_text()
            
            # Try to use mypy's auto-fix capabilities
            # For now, return a generic fix attempt
            return {
                "action": "type_annotation_fix_attempted",
                "success": False,
                "note": "Type annotation fixes require manual review"
            }
            
        except Exception as e:
            return {"error": f"Failed to fix type annotation: {str(e)}"}
    
    async def _run_mypy_fix(self, file_path: Path) -> Dict:
        """Run mypy with auto-fix options"""
        try:
            # Try to run mypy with --fix-errors (if available in newer versions)
            rc, output, error = run_cmd(["mypy", "--fix-errors", str(file_path)], self.repo_path)
            
            if rc == 0:
                return {
                    "action": "mypy_auto_fix",
                    "success": True,
                    "output": output
                }
            else:
                # Try alternative mypy options
                rc, output, error = run_cmd(["mypy", "--show-error-codes", str(file_path)], self.repo_path)
                
                return {
                    "action": "mypy_error_analysis",
                    "success": True,
                    "output": output,
                    "note": "Errors analyzed for manual fixing"
                }
                
        except Exception as e:
            return {"error": f"Failed to run mypy: {str(e)}"}
    
    async def install_common_type_stubs(self) -> Dict:
        """Install common type stubs for popular packages"""
        try:
            common_stubs = [
                "types-requests",
                "types-PyYAML",
                "types-setuptools",
                "types-six",
                "types-urllib3",
                "types-jsonschema"
            ]
            
            venv_path = self.repo_path / ".venv"
            if not venv_path.exists():
                return {"error": "No virtual environment found"}
            
            if self.repo_path.name == "windows":
                pip_path = venv_path / "Scripts" / "pip.exe"
            else:
                pip_path = venv_path / "bin" / "pip"
            
            installed_stubs = []
            failed_stubs = []
            
            for stub in common_stubs:
                rc, output, error = run_cmd([str(pip_path), "install", stub], self.repo_path)
                if rc == 0:
                    installed_stubs.append(stub)
                else:
                    failed_stubs.append(stub)
            
            return {
                "action": "installed_common_type_stubs",
                "installed": installed_stubs,
                "failed": failed_stubs,
                "success": len(installed_stubs) > 0
            }
            
        except Exception as e:
            return {"error": f"Failed to install common stubs: {str(e)}"}
    
    async def apply_all_type_fixes(self, findings: List[Dict]) -> Dict:
        """Apply all available type checking fixes"""
        results = {
            "mypy_fixes": [],
            "stubs_installed": False,
            "total_fixes": 0,
            "successful_fixes": 0
        }
        
        # Install common type stubs first
        stub_result = await self.install_common_type_stubs()
        results["stubs_installed"] = stub_result.get("success", False)
        
        # Apply mypy fixes
        mypy_fixes = await self.fix_mypy_issues(findings)
        results["mypy_fixes"] = mypy_fixes
        
        # Calculate totals
        results["total_fixes"] = len(mypy_fixes)
        results["successful_fixes"] = len([f for f in mypy_fixes if f.get('fix_applied', False)])
        
        return results
