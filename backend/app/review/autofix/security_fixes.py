import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ...core.vcs import run_cmd

class SecurityFixer:
    """Automatically fixes security vulnerabilities in dependencies"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.requirements_file = repo_path / "requirements.txt"
        self.package_json = repo_path / "package.json"
        self.gemfile = repo_path / "Gemfile"
        self.composer_json = repo_path / "composer.json"
    
    async def fix_python_dependencies(self, findings: List[Dict]) -> List[Dict]:
        """Fix Python dependency vulnerabilities"""
        if not self.requirements_file.exists():
            return []
        
        fixes_applied = []
        
        for finding in findings:
            if finding.get('tool') == 'pip-audit' and finding.get('autofixable'):
                fix_result = await self._fix_python_dependency(finding)
                if fix_result:
                    fixes_applied.append(fix_result)
        
        return fixes_applied
    
    async def _fix_python_dependency(self, finding: Dict) -> Optional[Dict]:
        """Fix a specific Python dependency vulnerability"""
        try:
            message = finding.get('message', '')
            remediation = finding.get('remediation', '')
            
            # Extract package name and version from message
            package_match = re.search(r'([a-zA-Z0-9_-]+)==([0-9.]+)', message)
            if not package_match:
                return None
            
            package_name = package_match.group(1)
            current_version = package_match.group(2)
            
            # Extract target version from remediation
            target_version_match = re.search(r'Upgrade [a-zA-Z0-9_-]+ to ([0-9.]+)', remediation)
            if not target_version_match:
                return None
            
            target_version = target_version_match.group(1)
            
            # Update requirements.txt
            await self._update_requirements_txt(package_name, current_version, target_version)
            
            # Try to upgrade the package
            upgrade_result = await self._upgrade_package(package_name, target_version)
            
            return {
                "finding_id": finding.get('id'),
                "package": package_name,
                "old_version": current_version,
                "new_version": target_version,
                "cve": finding.get('rule_id', ''),
                "fix_applied": True,
                "upgrade_success": upgrade_result.get('success', False),
                "upgrade_output": upgrade_result.get('output', '')
            }
            
        except Exception as e:
            return {
                "finding_id": finding.get('id'),
                "error": str(e),
                "fix_applied": False
            }
    
    async def _update_requirements_txt(self, package: str, old_version: str, new_version: str):
        """Update requirements.txt with new version"""
        try:
            # Read current requirements
            content = self.requirements_file.read_text()
            
            # Replace old version with new version
            old_spec = f"{package}=={old_version}"
            new_spec = f"{package}=={new_version}"
            
            if old_spec in content:
                content = content.replace(old_spec, new_spec)
                self.requirements_file.write_text(content)
                return True
            else:
                # Try to find package with different format
                pattern = rf"{package}[=<>!~]+{old_version}"
                if re.search(pattern, content):
                    content = re.sub(pattern, f"{package}=={new_version}", content)
                    self.requirements_file.write_text(content)
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error updating requirements.txt: {e}")
            return False
    
    async def _upgrade_package(self, package: str, version: str) -> Dict:
        """Attempt to upgrade the package using pip"""
        try:
            # Create virtual environment if it doesn't exist
            venv_path = self.repo_path / ".venv"
            if not venv_path.exists():
                run_cmd(["python", "-m", "venv", ".venv"], self.repo_path)
            
            # Activate venv and upgrade package
            if self.repo_path.name == "windows":
                pip_path = venv_path / "Scripts" / "pip.exe"
                activate_cmd = [str(venv_path / "Scripts" / "activate.bat")]
            else:
                pip_path = venv_path / "bin" / "pip"
                activate_cmd = [str(venv_path / "bin" / "activate")]
            
            # Upgrade package
            upgrade_cmd = [str(pip_path), "install", "--upgrade", f"{package}=={version}"]
            rc, output, error = run_cmd(upgrade_cmd, self.repo_path)
            
            return {
                "success": rc == 0,
                "output": output or error,
                "return_code": rc
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": str(e),
                "return_code": -1
            }
    
    async def fix_node_dependencies(self, findings: List[Dict]) -> List[Dict]:
        """Fix Node.js dependency vulnerabilities"""
        if not self.package_json.exists():
            return []
        
        fixes_applied = []
        
        for finding in findings:
            if finding.get('tool') == 'npm-audit' and finding.get('autofixable'):
                fix_result = await self._fix_node_dependency(finding)
                if fix_result:
                    fixes_applied.append(fix_result)
        
        return fixes_applied
    
    async def _fix_node_dependency(self, finding: Dict) -> Optional[Dict]:
        """Fix a specific Node.js dependency vulnerability"""
        try:
            # Run npm audit fix
            rc, output, error = run_cmd(["npm", "audit", "fix"], self.repo_path)
            
            return {
                "finding_id": finding.get('id'),
                "fix_applied": rc == 0,
                "output": output or error,
                "return_code": rc
            }
            
        except Exception as e:
            return {
                "finding_id": finding.get('id'),
                "error": str(e),
                "fix_applied": False
            }
    
    async def apply_all_security_fixes(self, findings: List[Dict]) -> Dict:
        """Apply all available security fixes"""
        results = {
            "python_fixes": [],
            "node_fixes": [],
            "total_fixes": 0,
            "successful_fixes": 0
        }
        
        # Apply Python fixes
        python_fixes = await self.fix_python_dependencies(findings)
        results["python_fixes"] = python_fixes
        
        # Apply Node.js fixes
        node_fixes = await self.fix_node_dependencies(findings)
        results["node_fixes"] = node_fixes
        
        # Calculate totals
        all_fixes = python_fixes + node_fixes
        results["total_fixes"] = len(all_fixes)
        results["successful_fixes"] = len([f for f in all_fixes if f.get('fix_applied', False)])
        
        return results
