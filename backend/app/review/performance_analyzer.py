import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Optional
from ..core.vcs import run_cmd

class PerformanceAnalyzer:
    """Analyzes code for performance issues and optimization opportunities"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.performance_patterns = self._load_performance_patterns()
    
    def _load_performance_patterns(self) -> Dict:
        """Load performance patterns for various issues"""
        return {
            "n_plus_one_queries": {
                "patterns": [
                    r"for\s+\w+\s+in\s+\w+:\s*\n\s*\w+\.query\.",
                    r"for\s+\w+\s+in\s+\w+:\s*\n\s*\w+\.filter\.",
                    r"for\s+\w+\s+in\s+\w+:\s*\n\s*\w+\.get\(",
                    r"for\s+\w+\s+in\s+\w+:\s*\n\s*db\.session\.query\.",
                ],
                "severity": "high",
                "description": "Potential N+1 query problem",
                "remediation": "Use eager loading, select_related, or prefetch_related to batch queries"
            },
            "memory_leaks": {
                "patterns": [
                    r"global\s+\w+",
                    r"class\s+\w+.*\n.*\w+\s*=\s*\[\]",
                    r"class\s+\w+.*\n.*\w+\s*=\s*\{\}",
                    r"@staticmethod\s*\n\s*def\s+\w+\([^)]*\):\s*\n\s*\w+\s*=\s*\[\]",
                ],
                "severity": "medium",
                "description": "Potential memory leak or global state issue",
                "remediation": "Avoid global variables and mutable class variables, use instance variables instead"
            },
            "inefficient_algorithms": {
                "patterns": [
                    r"\.sort\(\)\s*\n\s*for\s+\w+\s+in\s+\w+",
                    r"for\s+\w+\s+in\s+\w+:\s*\n\s*for\s+\w+\s+in\s+\w+:",
                    r"list\(set\(.*\)\)",
                    r"\[.*for.*in.*for.*in.*\]",
                ],
                "severity": "medium",
                "description": "Inefficient algorithm or data structure usage",
                "remediation": "Consider using more efficient data structures or algorithms (e.g., dict for lookups, set for uniqueness)"
            },
            "resource_management": {
                "patterns": [
                    r"open\([^)]*\)\s*\n\s*#\s*no\s*close",
                    r"requests\.get\([^)]*\)\s*\n\s*#\s*no\s*close",
                    r"subprocess\.Popen\([^)]*\)\s*\n\s*#\s*no\s*terminate",
                    r"threading\.Thread\([^)]*\)\s*\n\s*#\s*no\s*join",
                ],
                "severity": "medium",
                "description": "Resource not properly managed or closed",
                "remediation": "Use context managers (with statements) or ensure proper cleanup in finally blocks"
            },
            "blocking_operations": {
                "patterns": [
                    r"time\.sleep\(",
                    r"requests\.get\([^)]*\)\s*#\s*synchronous",
                    r"urllib\.request\.urlopen\(",
                    r"subprocess\.call\(",
                ],
                "severity": "low",
                "description": "Blocking operation that could be made asynchronous",
                "remediation": "Consider using async/await, asyncio, or background tasks for I/O operations"
            },
            "string_concatenation": {
                "patterns": [
                    r"\w+\s*\+\s*[\"'][^\"']*[\"']\s*\+\s*\w+",
                    r"[\"'][^\"']*[\"']\s*\+\s*\w+\s*\+\s*[\"'][^\"']*[\"']",
                ],
                "severity": "low",
                "description": "Inefficient string concatenation in loops",
                "remediation": "Use str.join() or f-strings for better performance"
            },
            "unnecessary_computations": {
                "patterns": [
                    r"len\(list\([^)]*\)\)",
                    r"len\(\[.*for.*in.*\]\)",
                    r"bool\(list\([^)]*\)\)",
                    r"bool\(\[.*for.*in.*\]\)",
                ],
                "severity": "low",
                "description": "Unnecessary computation or type conversion",
                "remediation": "Use more efficient alternatives (e.g., any() instead of bool(list(...)))"
            }
        }
    
    async def analyze_performance(self) -> List[Dict]:
        """Analyze code for performance issues"""
        findings = []
        
        # Check Python files
        python_files = list(self.repo_path.rglob("*.py"))
        for py_file in python_files:
            findings.extend(await self._analyze_python_performance(py_file))
        
        # Check JavaScript/TypeScript files
        js_files = list(self.repo_path.rglob("*.js")) + list(self.repo_path.rglob("*.ts"))
        for js_file in js_files:
            findings.extend(await self._analyze_javascript_performance(js_file))
        
        # Check for dependency performance issues
        findings.extend(await self._check_dependency_performance())
        
        return findings
    
    async def _analyze_python_performance(self, file_path: Path) -> List[Dict]:
        """Analyze Python file for performance issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Pattern-based analysis
            for line_num, line in enumerate(lines, 1):
                for issue_type, pattern_info in self.performance_patterns.items():
                    for pattern in pattern_info["patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                "tool": "performance-analyzer",
                                "severity": pattern_info["severity"],
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": line_num,
                                "rule_id": f"perf-{issue_type}",
                                "message": pattern_info["description"],
                                "remediation": pattern_info["remediation"],
                                "autofixable": False,
                                "code_snippet": line.strip(),
                                "issue_type": issue_type
                            })
            
            # AST-based analysis for more complex patterns
            findings.extend(await self._analyze_python_ast(content, file_path))
            
        except Exception as e:
            # Skip files that can't be read
            pass
        
        return findings
    
    async def _analyze_python_ast(self, content: str, file_path: Path) -> List[Dict]:
        """Analyze Python AST for performance issues"""
        findings = []
        
        try:
            tree = ast.parse(content)
            
            # Look for nested loops (O(n²) complexity)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # Check if there's a nested loop
                    for child in ast.walk(node):
                        if isinstance(child, ast.For) and child != node:
                            findings.append({
                                "tool": "performance-analyzer",
                                "severity": "medium",
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": node.lineno,
                                "rule_id": "perf-nested-loops",
                                "message": "Nested loops detected - potential O(n²) complexity",
                                "remediation": "Consider using itertools.product, list comprehensions, or vectorized operations",
                                "autofixable": False,
                                "code_snippet": f"for loop at line {node.lineno}",
                                "issue_type": "nested_loops"
                            })
                            break
            
            # Look for list comprehensions that could be generators
            for node in ast.walk(tree):
                if isinstance(node, ast.ListComp):
                    # Check if the list is only used for iteration
                    parent = getattr(node, 'parent', None)
                    if parent and isinstance(parent, ast.For):
                        findings.append({
                            "tool": "performance-analyzer",
                            "severity": "low",
                            "file": str(file_path.relative_to(self.repo_path)),
                            "line": node.lineno,
                            "rule_id": "perf-list-comp-to-generator",
                                "message": "List comprehension could be replaced with generator expression",
                                "remediation": "Use (x for x in ...) instead of [x for x in ...] for memory efficiency",
                                "autofixable": True,
                                "code_snippet": f"List comprehension at line {node.lineno}",
                                "issue_type": "list_comprehension"
                            })
            
        except Exception as e:
            # If AST parsing fails, skip this analysis
            pass
        
        return findings
    
    async def _analyze_javascript_performance(self, file_path: Path) -> List[Dict]:
        """Analyze JavaScript/TypeScript file for performance issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # JavaScript-specific performance patterns
            js_patterns = {
                "dom_manipulation": {
                    "patterns": [
                        r"for\s*\(\s*let\s+\w+\s*=\s*0;\s*\w+\s*<\s*\w+\.length;\s*\w\+\+\)\s*\{\s*\w+\.innerHTML\s*=",
                        r"for\s*\(\s*let\s+\w+\s+in\s+\w+\)\s*\{\s*\w+\.innerHTML\s*=",
                        r"forEach\s*\(\s*[^)]*\)\s*=>\s*\{\s*\w+\.innerHTML\s*=",
                    ],
                    "severity": "medium",
                    "description": "DOM manipulation in loops can cause performance issues",
                    "remediation": "Batch DOM updates or use DocumentFragment for better performance"
                },
                "memory_leaks": {
                    "patterns": [
                        r"addEventListener\s*\(\s*[^,]+,\s*[^)]+\)\s*#\s*no\s*removeEventListener",
                        r"setInterval\s*\(\s*[^,]+,\s*[^)]+\)\s*#\s*no\s*clearInterval",
                        r"setTimeout\s*\(\s*[^,]+,\s*[^)]+\)\s*#\s*no\s*clearTimeout",
                    ],
                    "severity": "medium",
                    "description": "Event listeners or timers not properly cleaned up",
                    "remediation": "Always remove event listeners and clear timers to prevent memory leaks"
                },
                "inefficient_selectors": {
                    "patterns": [
                        r"document\.getElementsByTagName\s*\(\s*[\"']\*[\"']\s*\)",
                        r"document\.querySelectorAll\s*\(\s*[\"']\*[\"']\s*\)",
                        r"document\.getElementsByClassName\s*\(\s*[\"']\w+[\"']\s*\)\s*\[\s*\d+\s*\]",
                    ],
                    "severity": "low",
                    "description": "Inefficient DOM selector usage",
                    "remediation": "Use more specific selectors or cache DOM queries"
                }
            }
            
            for line_num, line in enumerate(lines, 1):
                for issue_type, pattern_info in js_patterns.items():
                    for pattern in pattern_info["patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                "tool": "performance-analyzer",
                                "severity": pattern_info["severity"],
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": line_num,
                                "rule_id": f"perf-js-{issue_type}",
                                "message": pattern_info["description"],
                                "remediation": pattern_info["remediation"],
                                "autofixable": False,
                                "code_snippet": line.strip(),
                                "issue_type": issue_type
                            })
            
        except Exception as e:
            # Skip files that can't be read
            pass
        
        return findings
    
    async def _check_dependency_performance(self) -> List[Dict]:
        """Check for performance-related dependency issues"""
        findings = []
        
        # Check Python dependencies
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    content = f.read()
                
                # Look for known slow packages
                slow_packages = {
                    "pandas": "Consider using polars or pyarrow for better performance",
                    "numpy": "Ensure you're using vectorized operations, not loops",
                    "requests": "Consider using httpx or aiohttp for async operations",
                    "sqlalchemy": "Use bulk operations and proper indexing for database performance",
                    "matplotlib": "Consider using plotly or bokeh for interactive plots",
                }
                
                for package, advice in slow_packages.items():
                    if package in content:
                        findings.append({
                            "tool": "performance-analyzer",
                            "severity": "low",
                            "file": "requirements.txt",
                            "line": None,
                            "rule_id": f"perf-dependency-{package}",
                            "message": f"Performance consideration for {package}",
                            "remediation": advice,
                            "autofixable": False,
                            "issue_type": "dependency_performance"
                        })
                        
            except Exception:
                pass
        
        # Check Node.js dependencies
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    content = f.read()
                
                # Look for known performance considerations
                js_perf_packages = {
                    "lodash": "Consider using native JavaScript methods for simple operations",
                    "moment": "Consider using date-fns or native Date for better performance",
                    "jquery": "Consider using native DOM APIs for better performance",
                    "express": "Use compression middleware and proper caching for API performance",
                }
                
                for package, advice in js_perf_packages.items():
                    if package in content:
                        findings.append({
                            "tool": "performance-analyzer",
                            "severity": "low",
                            "file": "package.json",
                            "line": None,
                            "rule_id": f"perf-dependency-{package}",
                            "message": f"Performance consideration for {package}",
                            "remediation": advice,
                            "autofixable": False,
                            "issue_type": "dependency_performance"
                        })
                        
            except Exception:
                pass
        
        return findings
    
    async def generate_performance_report(self) -> Dict:
        """Generate a comprehensive performance analysis report"""
        findings = await self.analyze_performance()
        
        # Categorize findings
        by_severity = {}
        by_type = {}
        
        for finding in findings:
            severity = finding.get("severity", "low")
            issue_type = finding.get("issue_type", "unknown")
            
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(finding)
            
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(finding)
        
        # Calculate performance score (0-100)
        total_issues = len(findings)
        critical_issues = len(by_severity.get("critical", []))
        high_issues = len(by_severity.get("high", []))
        medium_issues = len(by_severity.get("medium", []))
        low_issues = len(by_severity.get("low", []))
        
        # Weighted scoring
        performance_score = max(0, 100 - (critical_issues * 20 + high_issues * 10 + medium_issues * 5 + low_issues * 2))
        
        return {
            "summary": {
                "total_issues": total_issues,
                "performance_score": performance_score,
                "critical_issues": critical_issues,
                "high_issues": high_issues,
                "medium_issues": medium_issues,
                "low_issues": low_issues
            },
            "findings_by_severity": by_severity,
            "findings_by_type": by_type,
            "recommendations": self._generate_performance_recommendations(findings),
            "all_findings": findings
        }
    
    def _generate_performance_recommendations(self, findings: List[Dict]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Count issue types
        issue_counts = {}
        for finding in findings:
            issue_type = finding.get("issue_type", "unknown")
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Generate specific recommendations
        if issue_counts.get("n_plus_one_queries", 0) > 0:
            recommendations.append(f"Fix {issue_counts['n_plus_one_queries']} N+1 query issues to improve database performance")
        
        if issue_counts.get("memory_leaks", 0) > 0:
            recommendations.append(f"Address {issue_counts['memory_leaks']} memory leak issues to prevent resource exhaustion")
        
        if issue_counts.get("inefficient_algorithms", 0) > 0:
            recommendations.append(f"Optimize {issue_counts['inefficient_algorithms']} inefficient algorithms for better scalability")
        
        if issue_counts.get("nested_loops", 0) > 0:
            recommendations.append(f"Refactor {issue_counts['nested_loops']} nested loops to reduce time complexity")
        
        # General recommendations
        if len(findings) > 10:
            recommendations.append("Consider a comprehensive performance audit and optimization sprint")
        
        if issue_counts.get("dependency_performance", 0) > 0:
            recommendations.append("Review and update dependencies for performance improvements")
        
        return recommendations

async def analyze_code_performance(repo_path: Path) -> Dict:
    """Analyze code performance and generate report"""
    analyzer = PerformanceAnalyzer(repo_path)
    return await analyzer.generate_performance_report()
