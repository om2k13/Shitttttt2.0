import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from ..core.vcs import run_cmd

class SecurityAnalyzer:
    """Enhanced security analysis with OWASP Top 10 checks"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict:
        """Load security patterns for various vulnerability types"""
        return {
            "sql_injection": {
                "patterns": [
                    r"execute\s*\(\s*[\"'].*\{\s*\w+\s*\}.*[\"']",  # f-string SQL
                    r"execute\s*\(\s*[\"'].*\+\s*\w+.*[\"']",      # String concatenation
                    r"cursor\.execute\s*\(\s*f[\"'].*\{.*\}.*[\"']", # f-string with cursor
                    r"\.execute\s*\(\s*[\"'].*%s.*[\"']\s*,\s*\(\s*\w+\s*\)", # Parameterized but wrong
                ],
                "severity": "high",
                "description": "Potential SQL injection vulnerability",
                "remediation": "Use parameterized queries with proper placeholders"
            },
            "xss": {
                "patterns": [
                    r"innerHTML\s*=\s*\w+",  # Direct innerHTML assignment
                    r"document\.write\s*\(\s*\w+\)",  # document.write
                    r"eval\s*\(\s*\w+\)",  # eval function
                    r"setTimeout\s*\(\s*\w+",  # setTimeout with user input
                    r"setInterval\s*\(\s*\w+",  # setInterval with user input
                ],
                "severity": "high",
                "description": "Potential Cross-Site Scripting (XSS) vulnerability",
                "remediation": "Sanitize user input and use safe DOM manipulation methods"
            },
            "command_injection": {
                "patterns": [
                    r"os\.system\s*\(\s*\w+\)",  # os.system
                    r"subprocess\.call\s*\(\s*\[\s*\w+",  # subprocess.call
                    r"subprocess\.Popen\s*\(\s*\[\s*\w+",  # subprocess.Popen
                    r"exec\s*\(\s*\w+\)",  # exec function
                ],
                "severity": "critical",
                "description": "Potential command injection vulnerability",
                "remediation": "Avoid executing user input as commands, use safe alternatives"
            },
            "path_traversal": {
                "patterns": [
                    r"open\s*\(\s*\w+",  # open() function
                    r"file\s*\(\s*\w+",  # file() function
                    r"Path\s*\(\s*\w+",  # Path constructor
                    r"os\.path\.join\s*\(\s*[\"']\.\.",  # Path traversal attempts
                ],
                "severity": "high",
                "description": "Potential path traversal vulnerability",
                "remediation": "Validate and sanitize file paths, use safe path operations"
            },
            "hardcoded_secrets": {
                "patterns": [
                    r"password\s*=\s*[\"'][^\"']{8,}[\"']",  # Hardcoded passwords
                    r"api_key\s*=\s*[\"'][^\"']{16,}[\"']",  # Hardcoded API keys
                    r"secret\s*=\s*[\"'][^\"']{8,}[\"']",    # Hardcoded secrets
                    r"token\s*=\s*[\"'][^\"']{16,}[\"']",    # Hardcoded tokens
                ],
                "severity": "critical",
                "description": "Hardcoded secrets in source code",
                "remediation": "Move secrets to environment variables or secure secret management"
            },
            "weak_crypto": {
                "patterns": [
                    r"hashlib\.md5\s*\(\s*\w+\)",  # MD5 hashing
                    r"hashlib\.sha1\s*\(\s*\w+\)",  # SHA1 hashing
                    r"\.encrypt\s*\(\s*\w+\)",      # Encryption without proper key management
                    r"base64\s*\.\s*b64encode",     # Base64 encoding (not encryption)
                ],
                "severity": "medium",
                "description": "Weak cryptographic implementation",
                "remediation": "Use strong cryptographic algorithms (SHA-256, bcrypt, etc.)"
            },
            "insecure_deserialization": {
                "patterns": [
                    r"pickle\.loads\s*\(\s*\w+\)",  # Pickle deserialization
                    r"yaml\.load\s*\(\s*\w+\)",     # YAML load (not safe_load)
                    r"json\.loads\s*\(\s*\w+\)",    # JSON deserialization (context dependent)
                ],
                "severity": "high",
                "description": "Insecure deserialization vulnerability",
                "remediation": "Use safe deserialization methods and validate input"
            },
            "broken_auth": {
                "patterns": [
                    r"if\s+user\s*==\s*[\"']admin[\"']",  # Hardcoded admin check
                    r"role\s*=\s*[\"']admin[\"']",         # Hardcoded role assignment
                    r"is_admin\s*=\s*True",                # Hardcoded admin flag
                    r"permission\s*=\s*[\"']all[\"']",     # Overly permissive permissions
                ],
                "severity": "high",
                "description": "Broken authentication or authorization",
                "remediation": "Implement proper authentication and role-based access control"
            },
            "sensitive_data_exposure": {
                "patterns": [
                    r"print\s*\(\s*\w+\.password",  # Printing passwords
                    r"log\s*\(\s*\w+\.token",       # Logging tokens
                    r"debug\s*\(\s*\w+\.secret",    # Debug logging secrets
                    r"console\.log\s*\(\s*\w+\.key", # Console logging keys
                ],
                "severity": "medium",
                "description": "Sensitive data exposure in logs or output",
                "remediation": "Avoid logging sensitive information, use redaction"
            },
            "missing_input_validation": {
                "patterns": [
                    r"def\s+\w+\([^)]*\):\s*[^#\n]*\n\s*[^#\n]*\w+\s*=",  # No validation
                    r"@app\.route.*\n\s*def\s+\w+\([^)]*\):\s*[^#\n]*\n\s*[^#\n]*\w+\s*=",  # Flask route
                    r"@.*\.post.*\n\s*def\s+\w+\([^)]*\):\s*[^#\n]*\n\s*[^#\n]*\w+\s*=",    # FastAPI route
                ],
                "severity": "medium",
                "description": "Missing input validation",
                "remediation": "Implement proper input validation and sanitization"
            }
        }
    
    async def run_owasp_checks(self) -> List[Dict]:
        """Run OWASP Top 10 security checks"""
        findings = []
        
        # Check Python files
        python_files = list(self.repo_path.rglob("*.py"))
        for py_file in python_files:
            findings.extend(await self._analyze_python_security(py_file))
        
        # Check JavaScript/TypeScript files
        js_files = list(self.repo_path.rglob("*.js")) + list(self.repo_path.rglob("*.ts"))
        for js_file in js_files:
            findings.extend(await self._analyze_javascript_security(js_file))
        
        # Check configuration files
        config_files = list(self.repo_path.rglob("*.yml")) + list(self.repo_path.rglob("*.yaml"))
        for config_file in config_files:
            findings.extend(await self._analyze_config_security(config_file))
        
        # Check for dependency vulnerabilities
        findings.extend(await self._check_dependency_vulnerabilities())
        
        return findings
    
    async def _analyze_python_security(self, file_path: Path) -> List[Dict]:
        """Analyze Python file for security vulnerabilities"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                for vuln_type, pattern_info in self.security_patterns.items():
                    for pattern in pattern_info["patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                "tool": "owasp-security",
                                "severity": pattern_info["severity"],
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": line_num,
                                "rule_id": f"owasp-{vuln_type}",
                                "message": pattern_info["description"],
                                "remediation": pattern_info["remediation"],
                                "autofixable": False,
                                "code_snippet": line.strip(),
                                "vulnerability_type": vuln_type
                            })
            
            # Additional Python-specific checks
            findings.extend(await self._check_python_specific_security(content, file_path))
            
        except Exception as e:
            # Skip files that can't be read
            pass
        
        return findings
    
    async def _analyze_javascript_security(self, file_path: Path) -> List[Dict]:
        """Analyze JavaScript/TypeScript file for security vulnerabilities"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                for vuln_type, pattern_info in self.security_patterns.items():
                    for pattern in pattern_info["patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                "tool": "owasp-security",
                                "severity": pattern_info["severity"],
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": line_num,
                                "rule_id": f"owasp-{vuln_type}",
                                "message": pattern_info["description"],
                                "remediation": pattern_info["remediation"],
                                "autofixable": False,
                                "code_snippet": line.strip(),
                                "vulnerability_type": vuln_type
                            })
            
            # Additional JavaScript-specific checks
            findings.extend(await self._check_javascript_specific_security(content, file_path))
            
        except Exception as e:
            # Skip files that can't be read
            pass
        
        return findings
    
    async def _analyze_config_security(self, file_path: Path) -> List[Dict]:
        """Analyze configuration files for security issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded secrets in config files
            secret_patterns = [
                r"password:\s*[\"'][^\"']{4,}[\"']",
                r"secret:\s*[\"'][^\"']{4,}[\"']",
                r"api_key:\s*[\"'][^\"']{16,}[\"']",
                r"token:\s*[\"'][^\"']{16,}[\"']",
                r"key:\s*[\"'][^\"']{16,}[\"']",
            ]
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    findings.append({
                        "tool": "owasp-security",
                        "severity": "critical",
                        "file": str(file_path.relative_to(self.repo_path)),
                        "line": content[:match.start()].count('\n') + 1,
                        "rule_id": "owasp-hardcoded-secrets-config",
                        "message": "Hardcoded secrets in configuration file",
                        "remediation": "Move secrets to environment variables or secure secret management",
                        "autofixable": False,
                        "code_snippet": match.group(0),
                        "vulnerability_type": "hardcoded_secrets"
                    })
            
            # Check for overly permissive settings
            permissive_patterns = [
                r"debug:\s*true",
                r"verbose:\s*true",
                r"log_level:\s*debug",
                r"development:\s*true",
            ]
            
            for pattern in permissive_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    findings.append({
                        "tool": "owasp-security",
                        "severity": "medium",
                        "file": str(file_path.relative_to(self.repo_path)),
                        "line": content[:match.start()].count('\n') + 1,
                        "rule_id": "owasp-permissive-config",
                        "message": "Overly permissive configuration setting",
                        "remediation": "Use production-appropriate settings, avoid debug mode in production",
                        "autofixable": False,
                        "code_snippet": match.group(0),
                        "vulnerability_type": "permissive_config"
                    })
            
        except Exception as e:
            # Skip files that can't be read
            pass
        
        return findings
    
    async def _check_python_specific_security(self, content: str, file_path: Path) -> List[Dict]:
        """Check for Python-specific security issues"""
        findings = []
        
        # Check for mutable default arguments
        mutable_default_pattern = r"def\s+\w+\([^)]*=\s*(\[\]|\{\}|None)[^)]*\):"
        if re.search(mutable_default_pattern, content):
            findings.append({
                "tool": "owasp-security",
                "severity": "medium",
                "file": str(file_path.relative_to(self.repo_path)),
                "line": None,
                "rule_id": "owasp-python-mutable-defaults",
                "message": "Mutable default arguments can lead to unexpected behavior",
                "remediation": "Use None as default and initialize mutable objects inside the function",
                "autofixable": False,
                "vulnerability_type": "python_mutable_defaults"
            })
        
        # Check for broad exception handling
        broad_except_pattern = r"except\s*:"
        if re.search(broad_except_pattern, content):
            findings.append({
                "tool": "owasp-security",
                "severity": "low",
                "file": str(file_path.relative_to(self.repo_path)),
                "line": None,
                "rule_id": "owasp-python-broad-except",
                "message": "Broad exception handling can mask security issues",
                "remediation": "Catch specific exceptions instead of using bare except",
                "autofixable": False,
                "vulnerability_type": "python_broad_except"
            })
        
        return findings
    
    async def _check_javascript_specific_security(self, content: str, file_path: Path) -> List[Dict]:
        """Check for JavaScript-specific security issues"""
        findings = []
        
        # Check for unawaited promises
        unawaited_promise_pattern = r"async\s+function\s+\w+\([^)]*\)\s*\{[^}]*\w+\([^)]*\)[^}]*\}"
        if re.search(unawaited_promise_pattern, content):
            findings.append({
                "tool": "owasp-security",
                "severity": "medium",
                "file": str(file_path.relative_to(self.repo_path)),
                "line": None,
                "rule_id": "owasp-js-unawaited-promises",
                "message": "Unawaited promises can lead to race conditions and errors",
                "remediation": "Always await promises or handle them with .then()/.catch()",
                "autofixable": False,
                "vulnerability_type": "js_unawaited_promises"
            })
        
        # Check for prototype pollution
        prototype_pollution_pattern = r"__proto__|prototype"
        if re.search(prototype_pollution_pattern, content):
            findings.append({
                "tool": "owasp-security",
                "severity": "high",
                "file": str(file_path.relative_to(self.repo_path)),
                "line": None,
                "rule_id": "owasp-js-prototype-pollution",
                "message": "Prototype pollution can lead to security vulnerabilities",
                "remediation": "Avoid direct manipulation of __proto__ and prototype properties",
                "autofixable": False,
                "vulnerability_type": "js_prototype_pollution"
            })
        
        return findings
    
    async def _check_dependency_vulnerabilities(self) -> List[Dict]:
        """Check for known dependency vulnerabilities"""
        findings = []
        
        # Check Python dependencies
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                # Run safety check if available
                rc, out, err = await run_cmd(["safety", "check", "--json"], self.repo_path)
                if rc == 0 and out.strip():
                    try:
                        data = json.loads(out)
                        for vuln in data:
                            findings.append({
                                "tool": "owasp-security",
                                "severity": "high" if vuln.get("severity") == "HIGH" else "medium",
                                "file": "requirements.txt",
                                "line": None,
                                "rule_id": f"owasp-dependency-{vuln.get('package', 'unknown')}",
                                "message": f"Vulnerable dependency: {vuln.get('package', 'unknown')} - {vuln.get('advisory', 'No advisory available')}",
                                "remediation": f"Upgrade {vuln.get('package', 'unknown')} to a secure version",
                                "autofixable": True,
                                "vulnerability_type": "dependency_vulnerability"
                            })
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass
        
        # Check Node.js dependencies
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                # Run npm audit if available
                rc, out, err = await run_cmd(["npm", "audit", "--json"], self.repo_path)
                if rc == 0 and out.strip():
                    try:
                        data = json.loads(out)
                        advisories = data.get("vulnerabilities", {})
                        for name, vuln in advisories.items():
                            findings.append({
                                "tool": "owasp-security",
                                "severity": "high" if vuln.get("severity") == "high" else "medium",
                                "file": "package.json",
                                "line": None,
                                "rule_id": f"owasp-dependency-{name}",
                                "message": f"Vulnerable dependency: {name} - {vuln.get('title', 'No title available')}",
                                "remediation": f"Run 'npm audit fix' or manually update {name}",
                                "autofixable": True,
                                "vulnerability_type": "dependency_vulnerability"
                            })
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass
        
        return findings

async def run_security_analysis(repo_path: Path) -> List[Dict]:
    """Run comprehensive security analysis on a repository"""
    analyzer = SecurityAnalyzer(repo_path)
    return await analyzer.run_owasp_checks()
