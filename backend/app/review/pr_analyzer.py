import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ..core.vcs import run_cmd, clone_repo
from ..db.models import Finding

class PRAnalyzer:
    """Analyzes specific PR diffs instead of entire repositories"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.changed_files = []
        self.diff_stats = {}
        
    async def analyze_pr_diff(self, base_branch: str, head_branch: str) -> Dict:
        """Analyze changes between two branches (PR diff)"""
        try:
            # Get the diff between branches
            diff_output = await self._get_diff(base_branch, head_branch)
            
            # Parse changed files
            self.changed_files = self._parse_changed_files(diff_output)
            
            # Get diff statistics
            self.diff_stats = self._parse_diff_stats(diff_output)
            
            # Analyze only changed files
            findings = await self._analyze_changed_files()
            
            return {
                "changed_files": self.changed_files,
                "diff_stats": self.diff_stats,
                "findings": findings,
                "summary": {
                    "total_files_changed": len(self.changed_files),
                    "lines_added": self.diff_stats.get("lines_added", 0),
                    "lines_deleted": self.diff_stats.get("lines_deleted", 0),
                    "total_findings": len(findings)
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to analyze PR diff: {str(e)}",
                "changed_files": [],
                "findings": []
            }
    
    async def _get_diff(self, base_branch: str, head_branch: str) -> str:
        """Get git diff between two branches"""
        # Fetch latest changes
        await run_cmd(["git", "fetch", "origin"], self.repo_path)
        
        # Get diff output
        rc, out, err = await run_cmd([
            "git", "diff", "--stat", "--name-status", 
            f"origin/{base_branch}..origin/{head_branch}"
        ], self.repo_path)
        
        if rc != 0:
            raise Exception(f"Git diff failed: {err}")
            
        return out
    
    def _parse_changed_files(self, diff_output: str) -> List[Dict]:
        """Parse changed files from git diff output"""
        files = []
        for line in diff_output.splitlines():
            if line.strip() and not line.startswith(' '):
                parts = line.split('\t')
                if len(parts) >= 2:
                    status = parts[0]
                    file_path = parts[1]
                    
                    # Determine file type
                    file_type = self._get_file_type(file_path)
                    
                    files.append({
                        "path": file_path,
                        "status": status,  # M=modified, A=added, D=deleted
                        "type": file_type,
                        "language": self._get_language(file_path)
                    })
        
        return files
    
    def _parse_diff_stats(self, diff_output: str) -> Dict:
        """Parse diff statistics from git output"""
        stats = {}
        
        # Look for the summary line at the end
        lines = diff_output.splitlines()
        for line in reversed(lines):
            if 'files changed' in line:
                # Parse: " 5 files changed, 123 insertions(+), 45 deletions(-)"
                parts = line.split(',')
                if len(parts) >= 3:
                    stats["files_changed"] = int(parts[0].split()[0])
                    
                    # Parse insertions
                    insertions = parts[1].strip().split()[0]
                    stats["lines_added"] = int(insertions)
                    
                    # Parse deletions
                    deletions = parts[2].strip().split()[0]
                    stats["lines_deleted"] = int(deletions)
                break
        
        return stats
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine the type of file change"""
        if file_path.endswith('.py'):
            return "python"
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            return "javascript"
        elif file_path.endswith(('.go')):
            return "go"
        elif file_path.endswith(('.rs')):
            return "rust"
        elif file_path.endswith(('.java')):
            return "java"
        elif file_path.endswith(('.cpp', '.c', '.h', '.hpp')):
            return "cpp"
        elif file_path.endswith(('.yml', '.yaml')):
            return "yaml"
        elif file_path.endswith(('.json')):
            return "json"
        elif file_path.endswith(('.md', '.txt')):
            return "documentation"
        else:
            return "other"
    
    def _get_language(self, file_path: str) -> str:
        """Get programming language from file extension"""
        ext_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React JSX',
            '.tsx': 'React TSX',
            '.go': 'Go',
            '.rs': 'Rust',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.yml': 'YAML',
            '.yaml': 'YAML',
            '.json': 'JSON',
            '.md': 'Markdown',
            '.txt': 'Text'
        }
        
        for ext, lang in ext_map.items():
            if file_path.endswith(ext):
                return lang
        
        return "Unknown"
    
    async def _analyze_changed_files(self) -> List[Dict]:
        """Analyze only the changed files for issues"""
        findings = []
        
        for file_info in self.changed_files:
            if file_info["status"] == "D":  # Deleted files
                continue
                
            file_path = self.repo_path / file_info["path"]
            if not file_path.exists():
                continue
            
            # Run appropriate analysis based on file type
            if file_info["type"] == "python":
                findings.extend(await self._analyze_python_file(file_path, file_info))
            elif file_info["type"] == "javascript":
                findings.extend(await self._analyze_javascript_file(file_path, file_info))
            elif file_info["type"] == "yaml":
                findings.extend(await self._analyze_yaml_file(file_path, file_info))
            elif file_info["type"] == "json":
                findings.extend(await self._analyze_json_file(file_path, file_info))
        
        return findings
    
    async def _analyze_python_file(self, file_path: Path, file_info: Dict) -> List[Dict]:
        """Analyze a Python file for issues"""
        findings = []
        
        # Run Ruff on the specific file
        rc, out, err = await run_cmd([
            "ruff", "check", "--output-format", "json", str(file_path)
        ], self.repo_path)
        
        if out.strip():
            try:
                data = json.loads(out)
                for item in data:
                    findings.append({
                        "tool": "ruff",
                        "severity": "low",
                        "file": str(file_path.relative_to(self.repo_path)),
                        "line": item.get("location", {}).get("row"),
                        "rule_id": item.get("code"),
                        "message": item.get("message"),
                        "autofixable": True,
                        "pr_context": {
                            "file_status": file_info["status"],
                            "file_type": file_info["type"],
                            "language": file_info["language"]
                        }
                    })
            except Exception:
                pass
        
        # Run MyPy on the specific file
        rc, out, err = await run_cmd([
            "mypy", "--hide-error-codes", "--no-error-summary", 
            "--pretty", "--no-color-output", str(file_path)
        ], self.repo_path)
        
        if out:
            for line in out.splitlines():
                if ":" in line and "error:" in line:
                    try:
                        file_part, ln, _ = line.split(":", 2)
                        findings.append({
                            "tool": "mypy",
                            "severity": "medium",
                            "file": str(file_path.relative_to(self.repo_path)),
                            "line": int(ln.strip()),
                            "rule_id": "mypy",
                            "message": line.strip(),
                            "pr_context": {
                                "file_status": file_info["status"],
                                "file_type": file_info["type"],
                                "language": file_info["language"]
                            }
                        })
                    except Exception:
                        pass
        
        return findings
    
    async def _analyze_javascript_file(self, file_path: Path, file_info: Dict) -> List[Dict]:
        """Analyze a JavaScript/TypeScript file for issues"""
        findings = []
        
        # Check if package.json exists for ESLint
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            # Run ESLint on the specific file
            rc, out, err = await run_cmd([
                "npx", "eslint", "--format", "json", str(file_path)
            ], self.repo_path)
            
            if out.strip():
                try:
                    data = json.loads(out)
                    for item in data:
                        for message in item.get("messages", []):
                            findings.append({
                                "tool": "eslint",
                                "severity": "low" if message.get("severity") == 1 else "medium",
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": message.get("line"),
                                "rule_id": message.get("ruleId", "eslint"),
                                "message": message.get("message"),
                                "autofixable": message.get("fix", False),
                                "pr_context": {
                                    "file_status": file_info["status"],
                                    "file_type": file_info["type"],
                                    "language": file_info["language"]
                                }
                            })
                except Exception:
                    pass
        
        return findings
    
    async def _analyze_yaml_file(self, file_path: Path, file_info: Dict) -> List[Dict]:
        """Analyze a YAML file for issues"""
        findings = []
        
        # Basic YAML validation
        try:
            import yaml
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
        except Exception as e:
            findings.append({
                "tool": "yaml-validator",
                "severity": "high",
                "file": str(file_path.relative_to(self.repo_path)),
                "line": None,
                "rule_id": "yaml-syntax-error",
                "message": f"YAML syntax error: {str(e)}",
                "autofixable": False,
                "pr_context": {
                    "file_status": file_info["status"],
                    "file_type": file_info["type"],
                    "language": file_info["language"]
                }
            })
        
        return findings
    
    async def _analyze_json_file(self, file_path: Path, file_info: Dict) -> List[Dict]:
        """Analyze a JSON file for issues"""
        findings = []
        
        # Basic JSON validation
        try:
            with open(file_path, 'r') as f:
                json.load(f)
        except Exception as e:
            findings.append({
                "tool": "json-validator",
                "severity": "high",
                "file": str(file_path.relative_to(self.repo_path)),
                "line": None,
                "rule_id": "json-syntax-error",
                "message": f"JSON syntax error: {str(e)}",
                "autofixable": False,
                "pr_context": {
                    "file_status": file_info["status"],
                    "file_type": file_info["type"],
                    "language": file_info["language"]
                }
            })
        
        return findings

async def analyze_pr_changes(repo_path: Path, base_branch: str, head_branch: str) -> Dict:
    """Convenience function to analyze PR changes"""
    analyzer = PRAnalyzer(repo_path)
    return await analyzer.analyze_pr_diff(base_branch, head_branch)
