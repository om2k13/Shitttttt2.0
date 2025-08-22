import re
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ...core.vcs import run_cmd

class CodeQualityFixer:
    """Automatically fixes code quality issues like unused imports and variables"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def fix_python_code_quality(self, findings: List[Dict]) -> List[Dict]:
        """Fix Python code quality issues"""
        fixes_applied = []
        
        for finding in findings:
            if finding.get('tool') == 'ruff' and finding.get('autofixable'):
                fix_result = await self._fix_python_issue(finding)
                if fix_result:
                    fixes_applied.append(fix_result)
        
        return fixes_applied
    
    async def _fix_python_issue(self, finding: Dict) -> Optional[Dict]:
        """Fix a specific Python code quality issue"""
        try:
            rule_id = finding.get('rule_id', '')
            file_path = finding.get('file', '')
            line_number = finding.get('line')
            
            if not file_path or not line_number:
                return None
            
            # Convert relative path to absolute
            if file_path.startswith('/'):
                abs_file_path = Path(file_path)
            else:
                abs_file_path = self.repo_path / file_path
            
            if not abs_file_path.exists():
                return None
            
            # Apply specific fixes based on rule ID
            if rule_id == 'F401':  # Unused imports
                fix_result = await self._fix_unused_imports(abs_file_path, line_number)
            elif rule_id == 'F841':  # Unused variables
                fix_result = await self._fix_unused_variables(abs_file_path, line_number)
            elif rule_id == 'E501':  # Line too long
                fix_result = await self._fix_line_length(abs_file_path, line_number)
            else:
                # Try to use ruff's auto-fix
                fix_result = await self._run_ruff_fix(abs_file_path)
            
            if fix_result:
                return {
                    "finding_id": finding.get('id'),
                    "rule_id": rule_id,
                    "file": str(file_path),
                    "line": line_number,
                    "fix_applied": True,
                    "fix_type": "code_quality",
                    "details": fix_result
                }
            
        except Exception as e:
            return {
                "finding_id": finding.get('id'),
                "error": str(e),
                "fix_applied": False
            }
        
        return None
    
    async def _fix_unused_imports(self, file_path: Path, line_number: int) -> Dict:
        """Remove unused imports from Python file"""
        try:
            # Read file content
            content = file_path.read_text()
            lines = content.split('\n')
            
            if line_number > len(lines):
                return {"error": "Line number out of range"}
            
            # Parse the file to find unused imports
            tree = ast.parse(content)
            
            # Find all import statements
            import_lines = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_lines.append(node.lineno)
            
            # Check if the line contains an import
            if line_number in import_lines:
                # Remove the specific line
                lines.pop(line_number - 1)  # Convert to 0-based index
                
                # Write back to file
                file_path.write_text('\n'.join(lines))
                
                return {
                    "action": "removed_unused_import",
                    "line_removed": line_number,
                    "success": True
                }
            
            return {"error": "No import found on specified line"}
            
        except Exception as e:
            return {"error": f"Failed to fix unused import: {str(e)}"}
    
    async def _fix_unused_variables(self, file_path: Path, line_number: int) -> Dict:
        """Fix unused variables in Python file"""
        try:
            # Read file content
            content = file_path.read_text()
            lines = content.split('\n')
            
            if line_number > len(lines):
                return {"error": "Line number out of range"}
            
            target_line = lines[line_number - 1]
            
            # Check if it's a variable assignment
            if '=' in target_line and not target_line.strip().startswith('#'):
                # Replace with underscore to indicate intentionally unused
                original_line = target_line
                new_line = re.sub(r'(\w+)\s*=', r'_\1 =', target_line)
                
                if new_line != original_line:
                    lines[line_number - 1] = new_line
                    file_path.write_text('\n'.join(lines))
                    
                    return {
                        "action": "renamed_unused_variable",
                        "line_modified": line_number,
                        "original": original_line.strip(),
                        "modified": new_line.strip(),
                        "success": True
                    }
            
            return {"error": "No variable assignment found on specified line"}
            
        except Exception as e:
            return {"error": f"Failed to fix unused variable: {str(e)}"}
    
    async def _fix_line_length(self, file_path: Path, line_number: int) -> Dict:
        """Fix lines that are too long"""
        try:
            # Use black to format the file
            rc, output, error = run_cmd(["black", "-q", str(file_path)], self.repo_path)
            
            if rc == 0:
                return {
                    "action": "formatted_with_black",
                    "success": True,
                    "output": output
                }
            else:
                return {
                    "action": "black_formatting_failed",
                    "success": False,
                    "error": error
                }
                
        except Exception as e:
            return {"error": f"Failed to run black: {str(e)}"}
    
    async def _run_ruff_fix(self, file_path: Path) -> Dict:
        """Run ruff's auto-fix on the file"""
        try:
            rc, output, error = run_cmd(["ruff", "check", "--fix", str(file_path)], self.repo_path)
            
            if rc == 0:
                return {
                    "action": "ruff_auto_fix",
                    "success": True,
                    "output": output
                }
            else:
                return {
                    "action": "ruff_fix_failed",
                    "success": False,
                    "error": error
                }
                
        except Exception as e:
            return {"error": f"Failed to run ruff fix: {str(e)}"}
    
    async def fix_javascript_code_quality(self, findings: List[Dict]) -> List[Dict]:
        """Fix JavaScript code quality issues"""
        fixes_applied = []
        
        for finding in findings:
            if finding.get('tool') == 'eslint' and finding.get('autofixable'):
                fix_result = await self._fix_javascript_issue(finding)
                if fix_result:
                    fixes_applied.append(fix_result)
        
        return fixes_applied
    
    async def _fix_javascript_issue(self, finding: Dict) -> Optional[Dict]:
        """Fix a specific JavaScript code quality issue"""
        try:
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
            
            # Use ESLint auto-fix
            rc, output, error = run_cmd(["npx", "eslint", "--fix", str(abs_file_path)], self.repo_path)
            
            return {
                "finding_id": finding.get('id'),
                "rule_id": finding.get('rule_id', ''),
                "file": str(file_path),
                "fix_applied": rc == 0,
                "fix_type": "javascript_quality",
                "output": output or error,
                "return_code": rc
            }
            
        except Exception as e:
            return {
                "finding_id": finding.get('id'),
                "error": str(e),
                "fix_applied": False
            }
    
    async def apply_all_code_quality_fixes(self, findings: List[Dict]) -> Dict:
        """Apply all available code quality fixes"""
        results = {
            "python_fixes": [],
            "javascript_fixes": [],
            "total_fixes": 0,
            "successful_fixes": 0
        }
        
        # Apply Python fixes
        python_fixes = await self.fix_python_code_quality(findings)
        results["python_fixes"] = python_fixes
        
        # Apply JavaScript fixes
        javascript_fixes = await self.fix_javascript_code_quality(findings)
        results["javascript_fixes"] = javascript_fixes
        
        # Calculate totals
        all_fixes = python_fixes + javascript_fixes
        results["total_fixes"] = len(all_fixes)
        results["successful_fixes"] = len([f for f in all_fixes if f.get('fix_applied', False)])
        
        return results
