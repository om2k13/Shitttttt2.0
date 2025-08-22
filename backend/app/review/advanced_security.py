import asyncio
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any
import aiohttp
import yaml
from ..core.vcs import run_cmd

class AdvancedSecurityScanner:
    """Advanced security scanning with SAST, DAST, and infrastructure analysis"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.security_tools = self._load_security_tools()
        self.vulnerability_db = self._load_vulnerability_database()
    
    def _load_security_tools(self) -> Dict:
        """Load configuration for various security tools"""
        return {
            "sast": {
                "semgrep": {
                    "enabled": True,
                    "configs": ["p/security-audit", "p/owasp-top-ten", "p/secrets"],
                    "timeout": 300
                },
                "bandit": {
                    "enabled": True,
                    "config": "bandit.yaml",
                    "timeout": 180
                },
                "safety": {
                    "enabled": True,
                    "timeout": 120
                },
                "npm_audit": {
                    "enabled": True,
                    "timeout": 120
                },
                "gosec": {
                    "enabled": True,
                    "timeout": 180
                }
            },
            "dast": {
                "zap": {
                    "enabled": False,  # Requires running application
                    "timeout": 600
                },
                "nikto": {
                    "enabled": False,  # Requires running application
                    "timeout": 300
                }
            },
            "container": {
                "trivy": {
                    "enabled": True,
                    "timeout": 300
                },
                "docker_scout": {
                    "enabled": True,
                    "timeout": 180
                }
            },
            "iac": {
                "tfsec": {
                    "enabled": True,
                    "timeout": 180
                },
                "checkov": {
                    "enabled": True,
                    "timeout": 180
                }
            }
        }
    
    def _load_vulnerability_database(self) -> Dict:
        """Load vulnerability database for enhanced analysis"""
        return {
            "cwe_mappings": {
                "CWE-79": "Cross-site Scripting (XSS)",
                "CWE-89": "SQL Injection",
                "CWE-78": "OS Command Injection",
                "CWE-200": "Information Exposure",
                "CWE-287": "Improper Authentication",
                "CWE-434": "Unrestricted Upload of File with Dangerous Type",
                "CWE-502": "Deserialization of Untrusted Data",
                "CWE-611": "Improper Restriction of XML External Entity Reference",
                "CWE-918": "Server-Side Request Forgery (SSRF)"
            },
            "severity_weights": {
                "critical": 10.0,
                "high": 7.0,
                "medium": 4.0,
                "low": 1.0
            },
            "exploitability_scores": {
                "remote": 1.0,
                "local": 0.7,
                "physical": 0.3
            }
        }
    
    async def run_comprehensive_security_scan(self) -> Dict:
        """Run comprehensive security analysis"""
        scan_results = {
            "scan_type": "comprehensive_security",
            "timestamp": None,
            "repository": str(self.repo_path),
            "sast_results": {},
            "dependency_results": {},
            "container_results": {},
            "iac_results": {},
            "summary": {},
            "risk_score": 0.0
        }
        
        try:
            # Run SAST analysis
            sast_results = await self._run_sast_analysis()
            scan_results["sast_results"] = sast_results
            
            # Run dependency analysis
            dependency_results = await self._run_dependency_analysis()
            scan_results["dependency_results"] = dependency_results
            
            # Run container security analysis
            container_results = await self._run_container_analysis()
            scan_results["container_results"] = container_results
            
            # Run Infrastructure as Code analysis
            iac_results = await self._run_iac_analysis()
            scan_results["iac_results"] = iac_results
            
            # Generate comprehensive summary
            scan_results["summary"] = self._generate_security_summary(scan_results)
            scan_results["risk_score"] = self._calculate_overall_risk_score(scan_results)
            scan_results["timestamp"] = self._get_timestamp()
            
        except Exception as e:
            scan_results["error"] = str(e)
        
        return scan_results
    
    async def _run_sast_analysis(self) -> Dict:
        """Run Static Application Security Testing"""
        sast_results = {
            "tools_used": [],
            "findings": [],
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0
        }
        
        # Run Semgrep
        if self.security_tools["sast"]["semgrep"]["enabled"]:
            try:
                semgrep_results = await self._run_semgrep()
                sast_results["tools_used"].append("semgrep")
                sast_results["findings"].extend(semgrep_results)
            except Exception as e:
                print(f"Semgrep failed: {e}")
        
        # Run Bandit (Python)
        if self.security_tools["sast"]["bandit"]["enabled"]:
            try:
                bandit_results = await self._run_bandit()
                sast_results["tools_used"].append("bandit")
                sast_results["findings"].extend(bandit_results)
            except Exception as e:
                print(f"Bandit failed: {e}")
        
        # Run Gosec (Go)
        if self.security_tools["sast"]["gosec"]["enabled"]:
            try:
                gosec_results = await self._run_gosec()
                sast_results["tools_used"].append("gosec")
                sast_results["findings"].extend(gosec_results)
            except Exception as e:
                print(f"Gosec failed: {e}")
        
        # Calculate statistics
        sast_results.update(self._calculate_finding_statistics(sast_results["findings"]))
        
        return sast_results
    
    async def _run_semgrep(self) -> List[Dict]:
        """Run Semgrep security analysis"""
        findings = []
        
        try:
            # Check if semgrep is available
            rc, version, _ = await run_cmd(["semgrep", "--version"], self.repo_path)
            if rc != 0:
                return findings
            
            # Run semgrep with security rules
            configs = self.security_tools["sast"]["semgrep"]["configs"]
            for config in configs:
                try:
                    rc, output, _ = await run_cmd([
                        "semgrep", "--config", config, "--json", "--quiet"
                    ], self.repo_path)
                    
                    if rc == 0 and output:
                        results = json.loads(output)
                        findings.extend(self._parse_semgrep_results(results))
                        
                except Exception as e:
                    print(f"Semgrep config {config} failed: {e}")
            
        except Exception as e:
            print(f"Semgrep execution failed: {e}")
        
        return findings
    
    async def _run_bandit(self) -> List[Dict]:
        """Run Bandit Python security analysis"""
        findings = []
        
        try:
            # Check if bandit is available
            rc, version, _ = await run_cmd(["bandit", "--version"], self.repo_path)
            if rc != 0:
                return findings
            
            # Find Python files
            python_files = list(self.repo_path.rglob("*.py"))
            if not python_files:
                return findings
            
            # Run bandit on Python files
            rc, output, _ = await run_cmd([
                "bandit", "-r", ".", "-f", "json", "-q"
            ], self.repo_path)
            
            if rc == 0 and output:
                results = json.loads(output)
                findings.extend(self._parse_bandit_results(results))
            
        except Exception as e:
            print(f"Bandit execution failed: {e}")
        
        return findings
    
    async def _run_gosec(self) -> List[Dict]:
        """Run Gosec Go security analysis"""
        findings = []
        
        try:
            # Check if gosec is available
            rc, version, _ = await run_cmd(["gosec", "--version"], self.repo_path)
            if rc != 0:
                return findings
            
            # Find Go files
            go_files = list(self.repo_path.rglob("*.go"))
            if not go_files:
                return findings
            
            # Run gosec
            rc, output, _ = await run_cmd([
                "gosec", "-fmt", "json", "./..."
            ], self.repo_path)
            
            if rc == 0 and output:
                results = json.loads(output)
                findings.extend(self._parse_gosec_results(results))
            
        except Exception as e:
            print(f"Gosec execution failed: {e}")
        
        return findings
    
    async def _run_dependency_analysis(self) -> Dict:
        """Run dependency vulnerability analysis"""
        dependency_results = {
            "python": {},
            "javascript": {},
            "go": {},
            "java": {},
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0
        }
        
        # Python dependencies
        if (self.repo_path / "requirements.txt").exists():
            try:
                python_results = await self._run_safety_check()
                dependency_results["python"] = python_results
            except Exception as e:
                print(f"Python dependency check failed: {e}")
        
        # JavaScript dependencies
        if (self.repo_path / "package.json").exists():
            try:
                js_results = await self._run_npm_audit()
                dependency_results["javascript"] = js_results
            except Exception as e:
                print(f"JavaScript dependency check failed: {e}")
        
        # Go dependencies
        if (self.repo_path / "go.mod").exists():
            try:
                go_results = await self._run_go_vuln_check()
                dependency_results["go"] = go_results
            except Exception as e:
                print(f"Go dependency check failed: {e}")
        
        # Calculate totals
        for lang_results in dependency_results.values():
            if isinstance(lang_results, dict):
                dependency_results["total_vulnerabilities"] += lang_results.get("total", 0)
                dependency_results["critical_vulnerabilities"] += lang_results.get("critical", 0)
                dependency_results["high_vulnerabilities"] += lang_results.get("high", 0)
        
        return dependency_results
    
    async def _run_safety_check(self) -> Dict:
        """Run Safety check for Python dependencies"""
        try:
            rc, output, _ = await run_cmd(["safety", "check", "--json"], self.repo_path)
            
            if rc == 0 and output:
                results = json.loads(output)
                return self._parse_safety_results(results)
            else:
                return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
                
        except Exception as e:
            print(f"Safety check failed: {e}")
            return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    
    async def _run_npm_audit(self) -> Dict:
        """Run npm audit for JavaScript dependencies"""
        try:
            rc, output, _ = await run_cmd(["npm", "audit", "--json"], self.repo_path)
            
            if rc == 0 and output:
                results = json.loads(output)
                return self._parse_npm_audit_results(results)
            else:
                return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
                
        except Exception as e:
            print(f"npm audit failed: {e}")
            return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    
    async def _run_go_vuln_check(self) -> Dict:
        """Run Go vulnerability check"""
        try:
            rc, output, _ = await run_cmd(["go", "list", "-json", "-m", "all"], self.repo_path)
            
            if rc == 0 and output:
                # Parse go.mod dependencies and check against vulnerability database
                return self._parse_go_dependencies(output)
            else:
                return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
                
        except Exception as e:
            print(f"Go vulnerability check failed: {e}")
            return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    
    async def _run_container_analysis(self) -> Dict:
        """Run container security analysis"""
        container_results = {
            "dockerfiles": [],
            "docker_compose": [],
            "kubernetes": [],
            "total_issues": 0
        }
        
        # Analyze Dockerfiles
        dockerfiles = list(self.repo_path.rglob("Dockerfile*"))
        for dockerfile in dockerfiles:
            try:
                analysis = await self._analyze_dockerfile(dockerfile)
                container_results["dockerfiles"].append(analysis)
            except Exception as e:
                print(f"Dockerfile analysis failed for {dockerfile}: {e}")
        
        # Analyze docker-compose files
        compose_files = list(self.repo_path.rglob("docker-compose*.yml")) + list(self.repo_path.rglob("docker-compose*.yaml"))
        for compose_file in compose_files:
            try:
                analysis = await self._analyze_docker_compose(compose_file)
                container_results["docker_compose"].append(analysis)
            except Exception as e:
                print(f"Docker Compose analysis failed for {compose_file}: {e}")
        
        # Analyze Kubernetes manifests
        k8s_files = list(self.repo_path.rglob("*.yaml")) + list(self.repo_path.rglob("*.yml"))
        for k8s_file in k8s_files:
            if self._is_kubernetes_manifest(k8s_file):
                try:
                    analysis = await self._analyze_kubernetes_manifest(k8s_file)
                    container_results["kubernetes"].append(analysis)
                except Exception as e:
                    print(f"Kubernetes analysis failed for {k8s_file}: {e}")
        
        # Calculate total issues
        for category in ["dockerfiles", "docker_compose", "kubernetes"]:
            for item in container_results[category]:
                container_results["total_issues"] += item.get("issues", 0)
        
        return container_results
    
    async def _analyze_dockerfile(self, dockerfile_path: Path) -> Dict:
        """Analyze Dockerfile for security issues"""
        analysis = {
            "file": str(dockerfile_path.relative_to(self.repo_path)),
            "issues": [],
            "risk_score": 0
        }
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check for common security issues
            security_patterns = {
                "root_user": {
                    "pattern": r"USER\s+root",
                    "severity": "high",
                    "description": "Container running as root user",
                    "remediation": "Use non-root user or USER instruction"
                },
                "latest_tag": {
                    "pattern": r"FROM\s+.*:latest",
                    "severity": "medium",
                    "description": "Using 'latest' tag which can be unpredictable",
                    "remediation": "Use specific version tags"
                },
                "sensitive_data": {
                    "pattern": r"(COPY|ADD)\s+.*\.(key|pem|p12|pfx)",
                    "severity": "critical",
                    "description": "Copying sensitive files into container",
                    "remediation": "Use secrets management instead"
                },
                "unnecessary_packages": {
                    "pattern": r"RUN\s+apt-get\s+install\s+.*\s+&&\s+apt-get\s+clean",
                    "severity": "low",
                    "description": "Not cleaning up package cache",
                    "remediation": "Clean package cache after installation"
                }
            }
            
            for issue_type, pattern_info in security_patterns.items():
                if pattern_info["pattern"] in content:
                    analysis["issues"].append({
                        "type": issue_type,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "remediation": pattern_info["remediation"],
                        "line": self._find_pattern_line(content, pattern_info["pattern"])
                    })
            
            # Calculate risk score
            analysis["risk_score"] = self._calculate_container_risk_score(analysis["issues"])
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    async def _analyze_docker_compose(self, compose_file_path: Path) -> Dict:
        """Analyze docker-compose file for security issues"""
        analysis = {
            "file": str(compose_file_path.relative_to(self.repo_path)),
            "issues": [],
            "risk_score": 0
        }
        
        try:
            with open(compose_file_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # Check for security issues
            if "services" in content:
                for service_name, service_config in content["services"].items():
                    # Check for privileged containers
                    if service_config.get("privileged", False):
                        analysis["issues"].append({
                            "type": "privileged_container",
                            "service": service_name,
                            "severity": "critical",
                            "description": "Container running in privileged mode",
                            "remediation": "Avoid privileged mode unless absolutely necessary"
                        })
                    
                    # Check for host network mode
                    if service_config.get("network_mode") == "host":
                        analysis["issues"].append({
                            "type": "host_network",
                            "service": service_name,
                            "severity": "high",
                            "description": "Container using host network mode",
                            "remediation": "Use bridge networking instead"
                        })
                    
                    # Check for volume mounts
                    volumes = service_config.get("volumes", [])
                    for volume in volumes:
                        if isinstance(volume, str) and ":" in volume:
                            host_path = volume.split(":")[0]
                            if host_path.startswith("/"):
                                analysis["issues"].append({
                                    "type": "host_path_mount",
                                    "service": service_name,
                                    "severity": "medium",
                                    "description": f"Mounting host path: {host_path}",
                                    "remediation": "Use named volumes instead of host paths"
                                })
            
            # Calculate risk score
            analysis["risk_score"] = self._calculate_container_risk_score(analysis["issues"])
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    async def _analyze_kubernetes_manifest(self, manifest_path: Path) -> Dict:
        """Analyze Kubernetes manifest for security issues"""
        analysis = {
            "file": str(manifest_path.relative_to(self.repo_path)),
            "issues": [],
            "risk_score": 0
        }
        
        try:
            with open(manifest_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # Check for security issues
            if content.get("kind") == "Pod":
                spec = content.get("spec", {})
                
                # Check for privileged containers
                containers = spec.get("containers", [])
                for container in containers:
                    security_context = container.get("securityContext", {})
                    if security_context.get("privileged", False):
                        analysis["issues"].append({
                            "type": "privileged_container",
                            "container": container.get("name", "unknown"),
                            "severity": "critical",
                            "description": "Container running in privileged mode",
                            "remediation": "Avoid privileged mode unless absolutely necessary"
                        })
                
                # Check for host path volumes
                volumes = spec.get("volumes", [])
                for volume in volumes:
                    if "hostPath" in volume:
                        analysis["issues"].append({
                            "type": "host_path_volume",
                            "volume": volume.get("name", "unknown"),
                            "severity": "medium",
                            "description": "Using hostPath volume",
                            "remediation": "Use persistent volumes instead"
                        })
            
            # Calculate risk score
            analysis["risk_score"] = self._calculate_container_risk_score(analysis["issues"])
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    async def _run_iac_analysis(self) -> Dict:
        """Run Infrastructure as Code security analysis"""
        iac_results = {
            "terraform": [],
            "cloudformation": [],
            "kubernetes": [],
            "total_issues": 0
        }
        
        # Analyze Terraform files
        tf_files = list(self.repo_path.rglob("*.tf")) + list(self.repo_path.rglob("*.tfvars"))
        for tf_file in tf_files:
            try:
                analysis = await self._analyze_terraform_file(tf_file)
                iac_results["terraform"].append(analysis)
            except Exception as e:
                print(f"Terraform analysis failed for {tf_file}: {e}")
        
        # Analyze CloudFormation files
        cf_files = list(self.repo_path.rglob("*.yaml")) + list(self.repo_path.rglob("*.yml"))
        for cf_file in cf_files:
            if self._is_cloudformation_template(cf_file):
                try:
                    analysis = await self._analyze_cloudformation_file(cf_file)
                    iac_results["cloudformation"].append(analysis)
                except Exception as e:
                    print(f"CloudFormation analysis failed for {cf_file}: {e}")
    
    async def _analyze_cloudformation_file(self, cf_file_path: Path) -> Dict:
        """Analyze CloudFormation file for security issues"""
        analysis = {
            "file": str(cf_file_path.relative_to(self.repo_path)),
            "issues": [],
            "risk_score": 0
        }
        
        try:
            with open(cf_file_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # Check for security issues
            security_patterns = {
                "public_access": {
                    "pattern": "0.0.0.0/0",
                    "severity": "critical",
                    "description": "Public access allowed (0.0.0.0/0)",
                    "remediation": "Restrict access to specific IP ranges"
                },
                "encryption_disabled": {
                    "pattern": "Encryption: false",
                    "severity": "high",
                    "description": "Encryption explicitly disabled",
                    "remediation": "Enable encryption for sensitive resources"
                },
                "logging_disabled": {
                    "pattern": "Logging: false",
                    "severity": "medium",
                    "description": "Logging explicitly disabled",
                    "remediation": "Enable logging for audit purposes"
                }
            }
            
            # Simple pattern matching for CloudFormation
            content_str = str(content)
            for issue_type, pattern_info in security_patterns.items():
                if pattern_info["pattern"] in content_str:
                    analysis["issues"].append({
                        "type": issue_type,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "remediation": pattern_info["remediation"]
                    })
            
            # Calculate risk score
            analysis["risk_score"] = self._calculate_container_risk_score(analysis["issues"])
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
        
        # Calculate total issues
        for category in ["terraform", "cloudformation", "kubernetes"]:
            for item in iac_results[category]:
                iac_results["total_issues"] += item.get("issues", 0)
        
        return iac_results
    
    async def _analyze_terraform_file(self, tf_file_path: Path) -> Dict:
        """Analyze Terraform file for security issues"""
        analysis = {
            "file": str(tf_file_path.relative_to(self.repo_path)),
            "issues": [],
            "risk_score": 0
        }
        
        try:
            with open(tf_file_path, 'r') as f:
                content = f.read()
            
            # Check for security issues
            security_patterns = {
                "public_access": {
                    "pattern": r'cidr_blocks\s*=\s*\[.*"0\.0\.0\.0/0"',
                    "severity": "critical",
                    "description": "Public access allowed (0.0.0.0/0)",
                    "remediation": "Restrict access to specific IP ranges"
                },
                "encryption_disabled": {
                    "pattern": r'encryption\s*=\s*false',
                    "severity": "high",
                    "description": "Encryption explicitly disabled",
                    "remediation": "Enable encryption for sensitive resources"
                },
                "logging_disabled": {
                    "pattern": r'logging\s*=\s*false',
                    "severity": "medium",
                    "description": "Logging explicitly disabled",
                    "remediation": "Enable logging for audit purposes"
                }
            }
            
            for issue_type, pattern_info in security_patterns.items():
                if pattern_info["pattern"] in content:
                    analysis["issues"].append({
                        "type": issue_type,
                        "severity": pattern_info["severity"],
                        "description": pattern_info["description"],
                        "remediation": pattern_info["remediation"]
                    })
            
            # Calculate risk score
            analysis["risk_score"] = self._calculate_container_risk_score(analysis["issues"])
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _is_kubernetes_manifest(self, file_path: Path) -> bool:
        """Check if file is a Kubernetes manifest"""
        try:
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f)
            
            return content.get("apiVersion") is not None and content.get("kind") is not None
        except:
            return False
    
    def _is_cloudformation_template(self, file_path: Path) -> bool:
        """Check if file is a CloudFormation template"""
        try:
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f)
            
            return content.get("AWSTemplateFormatVersion") is not None
        except:
            return False
    
    def _parse_semgrep_results(self, results: Dict) -> List[Dict]:
        """Parse Semgrep results into standardized format"""
        findings = []
        
        for result in results.get("results", []):
            finding = {
                "tool": "semgrep",
                "severity": self._map_semgrep_severity(result.get("extra", {}).get("severity", "medium")),
                "file": result.get("path", "Unknown"),
                "line": result.get("start", {}).get("line", 0),
                "message": result.get("extra", {}).get("message", "No message"),
                "rule_id": result.get("check_id", "Unknown"),
                "code_snippet": result.get("extra", {}).get("lines", ""),
                "cwe": result.get("extra", {}).get("metadata", {}).get("cwe", []),
                "autofixable": False
            }
            findings.append(finding)
        
        return findings
    
    def _parse_bandit_results(self, results: Dict) -> List[Dict]:
        """Parse Bandit results into standardized format"""
        findings = []
        
        for result in results.get("results", []):
            finding = {
                "tool": "bandit",
                "severity": self._map_bandit_severity(result.get("issue_severity", "medium")),
                "file": result.get("filename", "Unknown"),
                "line": result.get("line_number", 0),
                "message": result.get("issue_text", "No message"),
                "rule_id": result.get("test_id", "Unknown"),
                "code_snippet": result.get("code", ""),
                "cwe": [result.get("issue_cwe", {}).get("id", "Unknown")],
                "autofixable": False
            }
            findings.append(finding)
        
        return findings
    
    def _parse_gosec_results(self, results: Dict) -> List[Dict]:
        """Parse Gosec results into standardized format"""
        findings = []
        
        for result in results.get("Issues", []):
            finding = {
                "tool": "gosec",
                "severity": self._map_gosec_severity(result.get("severity", "medium")),
                "file": result.get("file", "Unknown"),
                "line": result.get("line", 0),
                "message": result.get("details", "No message"),
                "rule_id": result.get("rule_id", "Unknown"),
                "code_snippet": result.get("code", ""),
                "cwe": [result.get("cwe", {}).get("ID", "Unknown")],
                "autofixable": False
            }
            findings.append(finding)
        
        return findings
    
    def _parse_safety_results(self, results: List[Dict]) -> Dict:
        """Parse Safety results"""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for vuln in results:
            severity = vuln.get("severity", "medium").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return {
            "total": len(results),
            **severity_counts
        }
    
    def _parse_npm_audit_results(self, results: Dict) -> Dict:
        """Parse npm audit results"""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for severity, vulns in results.get("metadata", {}).get("vulnerabilities", {}).items():
            if severity in severity_counts:
                severity_counts[severity] = vulns
        
        return {
            "total": sum(severity_counts.values()),
            **severity_counts
        }
    
    def _parse_go_dependencies(self, output: str) -> Dict:
        """Parse Go dependencies for vulnerability analysis"""
        # This is a simplified parser - in practice, you'd want to check against a vulnerability database
        return {
            "total": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
    
    def _map_semgrep_severity(self, severity: str) -> str:
        """Map Semgrep severity to standard levels"""
        mapping = {
            "ERROR": "high",
            "WARNING": "medium",
            "INFO": "low"
        }
        return mapping.get(severity.upper(), "medium")
    
    def _map_bandit_severity(self, severity: str) -> str:
        """Map Bandit severity to standard levels"""
        mapping = {
            "HIGH": "high",
            "MEDIUM": "medium",
            "LOW": "low"
        }
        return mapping.get(severity.upper(), "medium")
    
    def _map_gosec_severity(self, severity: str) -> str:
        """Map Gosec severity to standard levels"""
        mapping = {
            "HIGH": "high",
            "MEDIUM": "medium",
            "LOW": "low"
        }
        return mapping.get(severity.upper(), "medium")
    
    def _calculate_finding_statistics(self, findings: List[Dict]) -> Dict:
        """Calculate statistics for findings"""
        stats = {
            "total_issues": len(findings),
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0
        }
        
        for finding in findings:
            severity = finding.get("severity", "medium").lower()
            if severity in stats:
                stats[f"{severity}_issues"] += 1
        
        return stats
    
    def _calculate_container_risk_score(self, issues: List[Dict]) -> float:
        """Calculate risk score for container/IaC issues"""
        if not issues:
            return 0.0
        
        total_score = 0
        for issue in issues:
            severity = issue.get("severity", "medium")
            weight = self.vulnerability_db["severity_weights"].get(severity, 4.0)
            total_score += weight
        
        return min(10.0, total_score / len(issues))
    
    def _calculate_overall_risk_score(self, scan_results: Dict) -> float:
        """Calculate overall risk score for the entire scan"""
        total_score = 0
        total_weight = 0
        
        # SAST findings
        sast_results = scan_results.get("sast_results", {})
        sast_score = (
            sast_results.get("critical_issues", 0) * 10 +
            sast_results.get("high_issues", 0) * 7 +
            sast_results.get("medium_issues", 0) * 4 +
            sast_results.get("low_issues", 0) * 1
        )
        total_score += sast_score
        total_weight += sast_results.get("total_issues", 0)
        
        # Dependency vulnerabilities
        dep_results = scan_results.get("dependency_results", {})
        dep_score = (
            dep_results.get("critical_vulnerabilities", 0) * 10 +
            dep_results.get("high_vulnerabilities", 0) * 7
        )
        total_score += dep_score
        total_weight += dep_results.get("total_vulnerabilities", 0)
        
        # Container and IaC issues
        container_results = scan_results.get("container_results", {})
        iac_results = scan_results.get("iac_results", {})
        
        container_score = container_results.get("total_issues", 0) * 5
        iac_score = iac_results.get("total_issues", 0) * 5
        
        total_score += container_score + iac_score
        total_weight += container_results.get("total_issues", 0) + iac_results.get("total_issues", 0)
        
        # Normalize to 0-10 scale
        if total_weight > 0:
            normalized_score = (total_score / total_weight) * 10 / 10  # Max possible score is 10
            return min(10.0, normalized_score)
        
        return 0.0
    
    def _generate_security_summary(self, scan_results: Dict) -> Dict:
        """Generate comprehensive security summary"""
        summary = {
            "overall_risk_level": "Low",
            "critical_findings": 0,
            "high_findings": 0,
            "total_vulnerabilities": 0,
            "security_coverage": "Comprehensive",
            "recommendations": []
        }
        
        # Calculate totals
        sast_results = scan_results.get("sast_results", {})
        dep_results = scan_results.get("dependency_results", {})
        
        summary["critical_findings"] = (
            sast_results.get("critical_issues", 0) +
            dep_results.get("critical_vulnerabilities", 0)
        )
        
        summary["high_findings"] = (
            sast_results.get("high_issues", 0) +
            dep_results.get("high_vulnerabilities", 0)
        )
        
        summary["total_vulnerabilities"] = (
            sast_results.get("total_issues", 0) +
            dep_results.get("total_vulnerabilities", 0)
        )
        
        # Determine overall risk level
        risk_score = scan_results.get("risk_score", 0)
        if risk_score >= 8:
            summary["overall_risk_level"] = "Critical"
        elif risk_score >= 6:
            summary["overall_risk_level"] = "High"
        elif risk_score >= 4:
            summary["overall_risk_level"] = "Medium"
        else:
            summary["overall_risk_level"] = "Low"
        
        # Generate recommendations
        if summary["critical_findings"] > 0:
            summary["recommendations"].append("Immediately address all critical security findings")
        
        if summary["high_findings"] > 0:
            summary["recommendations"].append("Address high-severity findings before next release")
        
        if dep_results.get("total_vulnerabilities", 0) > 0:
            summary["recommendations"].append("Update vulnerable dependencies to latest secure versions")
        
        return summary
    
    def _find_pattern_line(self, content: str, pattern: str) -> int:
        """Find the line number where a pattern occurs"""
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            if pattern in line:
                return i
        return 0
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

# Convenience functions
async def run_comprehensive_security_scan(repo_path: Path) -> Dict:
    """Run comprehensive security analysis on a repository"""
    scanner = AdvancedSecurityScanner(repo_path)
    return await scanner.run_comprehensive_security_scan()
