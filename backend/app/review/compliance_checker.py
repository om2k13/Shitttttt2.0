import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import re
from ..core.vcs import run_cmd

class ComplianceChecker:
    """Comprehensive compliance checking for various industry standards"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.compliance_frameworks = self._load_compliance_frameworks()
        self.custom_rules = self._load_custom_rules()
    
    def _load_compliance_frameworks(self) -> Dict:
        """Load compliance framework definitions"""
        return {
            "soc2": {
                "name": "SOC 2 Type II",
                "description": "Service Organization Control 2 compliance",
                "controls": {
                    "cc1": {
                        "name": "Control Environment",
                        "description": "Management's commitment to integrity and ethical values",
                        "rules": [
                            {
                                "id": "CC1.1",
                                "description": "Entity demonstrates commitment to integrity and ethical values",
                                "patterns": [
                                    r"code_of_conduct",
                                    r"ethics_policy",
                                    r"integrity_statement"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "cc2": {
                        "name": "Communication and Information",
                        "description": "Quality of information supporting the functioning of internal control",
                        "rules": [
                            {
                                "id": "CC2.1",
                                "description": "Information quality requirements are identified",
                                "patterns": [
                                    r"data_validation",
                                    r"input_sanitization",
                                    r"error_handling"
                                ],
                                "severity": "medium"
                            }
                        ]
                    },
                    "cc3": {
                        "name": "Risk Assessment",
                        "description": "Entity's process for identifying and responding to business risks",
                        "rules": [
                            {
                                "id": "CC3.1",
                                "description": "Risk assessment process is in place",
                                "patterns": [
                                    r"risk_assessment",
                                    r"threat_modeling",
                                    r"security_review"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "cc4": {
                        "name": "Monitoring Activities",
                        "description": "Ongoing evaluations to ascertain that controls are present and functioning",
                        "rules": [
                            {
                                "id": "CC4.1",
                                "description": "Ongoing monitoring is performed",
                                "patterns": [
                                    r"logging",
                                    r"monitoring",
                                    r"audit_trail"
                                ],
                                "severity": "medium"
                            }
                        ]
                    },
                    "cc5": {
                        "name": "Control Activities",
                        "description": "Policies and procedures that help ensure management directives are carried out",
                        "rules": [
                            {
                                "id": "CC5.1",
                                "description": "Control activities are implemented",
                                "patterns": [
                                    r"authentication",
                                    r"authorization",
                                    r"encryption"
                                ],
                                "severity": "high"
                            }
                        ]
                    }
                }
            },
            "pci_dss": {
                "name": "PCI DSS v4.0",
                "description": "Payment Card Industry Data Security Standard",
                "controls": {
                    "req1": {
                        "name": "Install and Maintain Network Security Controls",
                        "description": "Network security controls are implemented and maintained",
                        "rules": [
                            {
                                "id": "Req1.1",
                                "description": "Network security controls are implemented",
                                "patterns": [
                                    r"firewall",
                                    r"network_security",
                                    r"access_control"
                                ],
                                "severity": "critical"
                            }
                        ]
                    },
                    "req2": {
                        "name": "Apply Secure Configurations",
                        "description": "Secure configurations are applied to all system components",
                        "rules": [
                            {
                                "id": "Req2.1",
                                "description": "Secure configurations are applied",
                                "patterns": [
                                    r"secure_config",
                                    r"hardening",
                                    r"baseline"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "req3": {
                        "name": "Protect Stored Account Data",
                        "description": "Stored account data is protected",
                        "rules": [
                            {
                                "id": "Req3.1",
                                "description": "Account data is protected",
                                "patterns": [
                                    r"encryption",
                                    r"hashing",
                                    r"tokenization"
                                ],
                                "severity": "critical"
                            }
                        ]
                    },
                    "req4": {
                        "name": "Protect Cardholder Data",
                        "description": "Cardholder data is protected during transmission",
                        "rules": [
                            {
                                "id": "Req4.1",
                                "description": "Data is protected during transmission",
                                "patterns": [
                                    r"tls",
                                    r"ssl",
                                    r"https"
                                ],
                                "severity": "critical"
                            }
                        ]
                    },
                    "req5": {
                        "name": "Protect All Systems and Networks",
                        "description": "All systems and networks are protected from malicious software",
                        "rules": [
                            {
                                "id": "Req5.1",
                                "description": "Systems are protected from malware",
                                "patterns": [
                                    r"antivirus",
                                    r"malware_protection",
                                    r"security_scanning"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "req6": {
                        "name": "Develop and Maintain Secure Systems",
                        "description": "Secure systems and software are developed and maintained",
                        "rules": [
                            {
                                "id": "Req6.1",
                                "description": "Secure development practices are followed",
                                "patterns": [
                                    r"secure_coding",
                                    r"code_review",
                                    r"testing"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "req7": {
                        "name": "Restrict Access",
                        "description": "Access to system components and cardholder data is restricted",
                        "rules": [
                            {
                                "id": "Req7.1",
                                "description": "Access is restricted",
                                "patterns": [
                                    r"access_control",
                                    r"role_based_access",
                                    r"least_privilege"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "req8": {
                        "name": "Identify Users and Authenticate Access",
                        "description": "Users are identified and authenticated",
                        "rules": [
                            {
                                "id": "Req8.1",
                                "description": "Users are identified and authenticated",
                                "patterns": [
                                    r"authentication",
                                    r"multi_factor",
                                    r"user_management"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "req9": {
                        "name": "Restrict Physical Access",
                        "description": "Physical access to cardholder data is restricted",
                        "rules": [
                            {
                                "id": "Req9.1",
                                "description": "Physical access is restricted",
                                "patterns": [
                                    r"physical_security",
                                    r"access_control",
                                    r"surveillance"
                                ],
                                "severity": "medium"
                            }
                        ]
                    },
                    "req10": {
                        "name": "Log and Monitor Access",
                        "description": "Access to network resources and cardholder data is logged and monitored",
                        "rules": [
                            {
                                "id": "Req10.1",
                                "description": "Access is logged and monitored",
                                "patterns": [
                                    r"logging",
                                    r"monitoring",
                                    r"audit_trail"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "req11": {
                        "name": "Test Security",
                        "description": "Security of systems and networks is regularly tested",
                        "rules": [
                            {
                                "id": "Req11.1",
                                "description": "Security is regularly tested",
                                "patterns": [
                                    r"penetration_testing",
                                    r"vulnerability_assessment",
                                    r"security_testing"
                                ],
                                "severity": "medium"
                            }
                        ]
                    },
                    "req12": {
                        "name": "Support Information Security",
                        "description": "Information security is supported by policies and procedures",
                        "rules": [
                            {
                                "id": "Req12.1",
                                "description": "Security policies are in place",
                                "patterns": [
                                    r"security_policy",
                                    r"incident_response",
                                    r"security_awareness"
                                ],
                                "severity": "medium"
                            }
                        ]
                    }
                }
            },
            "hipaa": {
                "name": "HIPAA Security Rule",
                "description": "Health Insurance Portability and Accountability Act",
                "controls": {
                    "administrative": {
                        "name": "Administrative Safeguards",
                        "description": "Administrative actions and policies to manage security",
                        "rules": [
                            {
                                "id": "164.308(a)(1)",
                                "description": "Security management process",
                                "patterns": [
                                    r"risk_assessment",
                                    r"security_management",
                                    r"policies_procedures"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "physical": {
                        "name": "Physical Safeguards",
                        "description": "Physical measures to protect information systems",
                        "rules": [
                            {
                                "id": "164.310(a)(1)",
                                "description": "Facility access controls",
                                "patterns": [
                                    r"physical_access",
                                    r"facility_security",
                                    r"workstation_security"
                                ],
                                "severity": "medium"
                            }
                        ]
                    },
                    "technical": {
                        "name": "Technical Safeguards",
                        "description": "Technology and policies to protect electronic information",
                        "rules": [
                            {
                                "id": "164.312(a)(1)",
                                "description": "Access control",
                                "patterns": [
                                    r"authentication",
                                    r"encryption",
                                    r"audit_logs"
                                ],
                                "severity": "high"
                            }
                        ]
                    }
                }
            },
            "gdpr": {
                "name": "GDPR",
                "description": "General Data Protection Regulation",
                "controls": {
                    "data_protection": {
                        "name": "Data Protection Principles",
                        "description": "Principles for processing personal data",
                        "rules": [
                            {
                                "id": "Art5.1",
                                "description": "Lawfulness, fairness, and transparency",
                                "patterns": [
                                    r"privacy_policy",
                                    r"data_processing",
                                    r"consent_management"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "data_subject_rights": {
                        "name": "Data Subject Rights",
                        "description": "Rights of individuals regarding their data",
                        "rules": [
                            {
                                "id": "Art15-22",
                                "description": "Data subject rights implementation",
                                "patterns": [
                                    r"data_portability",
                                    r"right_to_erasure",
                                    r"access_requests"
                                ],
                                "severity": "high"
                            }
                        ]
                    },
                    "security": {
                        "name": "Security of Processing",
                        "description": "Appropriate technical and organizational measures",
                        "rules": [
                            {
                                "id": "Art32",
                                "description": "Security measures implementation",
                                "patterns": [
                                    r"encryption",
                                    r"access_control",
                                    r"data_minimization"
                                ],
                                "severity": "high"
                            }
                        ]
                    }
                }
            }
        }
    
    def _load_custom_rules(self) -> Dict:
        """Load custom compliance rules"""
        custom_rules_file = self.repo_path / ".compliance" / "custom_rules.yaml"
        if custom_rules_file.exists():
            try:
                with open(custom_rules_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load custom rules: {e}")
        
        return {}
    
    async def run_compliance_check(self, frameworks: List[str] = None) -> Dict:
        """Run compliance check against specified frameworks"""
        if frameworks is None:
            frameworks = list(self.compliance_frameworks.keys())
        
        compliance_results = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(self.repo_path),
            "frameworks_checked": frameworks,
            "results": {},
            "overall_compliance_score": 0.0,
            "summary": {}
        }
        
        try:
            total_score = 0
            total_controls = 0
            
            for framework in frameworks:
                if framework in self.compliance_frameworks:
                    framework_results = await self._check_framework_compliance(framework)
                    compliance_results["results"][framework] = framework_results
                    
                    # Calculate framework score
                    framework_score = framework_results.get("compliance_score", 0)
                    total_score += framework_score
                    total_controls += framework_results.get("total_controls", 1)
            
            # Calculate overall compliance score
            if total_controls > 0:
                compliance_results["overall_compliance_score"] = total_score / total_controls
            
            # Generate summary
            compliance_results["summary"] = self._generate_compliance_summary(compliance_results)
            
        except Exception as e:
            compliance_results["error"] = str(e)
        
        return compliance_results
    
    async def _check_framework_compliance(self, framework: str) -> Dict:
        """Check compliance for a specific framework"""
        framework_config = self.compliance_frameworks[framework]
        framework_results = {
            "framework_name": framework_config["name"],
            "description": framework_config["description"],
            "controls": {},
            "compliance_score": 0.0,
            "total_controls": 0,
            "compliant_controls": 0,
            "non_compliant_controls": 0,
            "partial_controls": 0
        }
        
        total_score = 0
        total_controls = 0
        
        for control_id, control_config in framework_config["controls"].items():
            control_results = await self._check_control_compliance(control_config)
            framework_results["controls"][control_id] = control_results
            
            # Calculate control score
            control_score = control_results.get("compliance_score", 0)
            total_score += control_score
            total_controls += 1
            
            # Count control status
            status = control_results.get("status", "non_compliant")
            if status == "compliant":
                framework_results["compliant_controls"] += 1
            elif status == "partial":
                framework_results["partial_controls"] += 1
            else:
                framework_results["non_compliant_controls"] += 1
        
        # Calculate framework compliance score
        if total_controls > 0:
            framework_results["compliance_score"] = total_score / total_controls
            framework_results["total_controls"] = total_controls
        
        return framework_results
    
    async def _check_control_compliance(self, control_config: Dict) -> Dict:
        """Check compliance for a specific control"""
        control_results = {
            "name": control_config["name"],
            "description": control_config["description"],
            "rules": {},
            "compliance_score": 0.0,
            "status": "non_compliant",
            "evidence": [],
            "recommendations": []
        }
        
        total_score = 0
        total_rules = 0
        
        for rule in control_config["rules"]:
            rule_results = await self._check_rule_compliance(rule)
            control_results["rules"][rule["id"]] = rule_results
            
            # Calculate rule score
            rule_score = rule_results.get("compliance_score", 0)
            total_score += rule_score
            total_rules += 1
            
            # Collect evidence and recommendations
            control_results["evidence"].extend(rule_results.get("evidence", []))
            control_results["recommendations"].extend(rule_results.get("recommendations", []))
        
        # Calculate control compliance score
        if total_rules > 0:
            control_results["compliance_score"] = total_score / total_rules
            
            # Determine control status
            if control_results["compliance_score"] >= 0.9:
                control_results["status"] = "compliant"
            elif control_results["compliance_score"] >= 0.6:
                control_results["status"] = "partial"
            else:
                control_results["status"] = "non_compliant"
        
        return control_results
    
    async def _check_rule_compliance(self, rule: Dict) -> Dict:
        """Check compliance for a specific rule"""
        rule_results = {
            "id": rule["id"],
            "description": rule["description"],
            "severity": rule["severity"],
            "compliance_score": 0.0,
            "status": "non_compliant",
            "evidence": [],
            "recommendations": [],
            "files_checked": 0,
            "patterns_found": 0
        }
        
        try:
            # Check for patterns in the codebase
            patterns_found = []
            files_checked = 0
            
            for pattern in rule["patterns"]:
                pattern_results = await self._search_pattern_in_codebase(pattern)
                if pattern_results["found"]:
                    patterns_found.extend(pattern_results["matches"])
                files_checked += pattern_results["files_checked"]
            
            rule_results["files_checked"] = files_checked
            rule_results["patterns_found"] = len(patterns_found)
            
            # Calculate compliance score based on pattern matches
            if patterns_found:
                rule_results["compliance_score"] = min(1.0, len(patterns_found) / 3)  # Cap at 100%
                rule_results["status"] = "compliant"
                rule_results["evidence"] = [
                    f"Found {len(patterns_found)} instances of {rule['description']}"
                ]
            else:
                rule_results["compliance_score"] = 0.0
                rule_results["status"] = "non_compliant"
                rule_results["recommendations"] = [
                    f"Implement {rule['description']} to meet compliance requirements"
                ]
            
        except Exception as e:
            rule_results["error"] = str(e)
            rule_results["compliance_score"] = 0.0
        
        return rule_results
    
    async def _search_pattern_in_codebase(self, pattern: str) -> Dict:
        """Search for a pattern in the codebase"""
        search_results = {
            "found": False,
            "matches": [],
            "files_checked": 0
        }
        
        try:
            # Search in various file types
            file_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rb", "*.php", "*.cs", "*.cpp", "*.c", "*.h", "*.hpp"]
            
            for file_pattern in file_patterns:
                files = list(self.repo_path.rglob(file_pattern))
                search_results["files_checked"] += len(files)
                
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # Simple pattern matching (could be enhanced with regex)
                            if pattern.lower() in content.lower():
                                search_results["found"] = True
                                search_results["matches"].append({
                                    "file": str(file_path.relative_to(self.repo_path)),
                                    "pattern": pattern,
                                    "context": self._extract_context(content, pattern)
                                })
                    except Exception:
                        continue
            
        except Exception as e:
            print(f"Pattern search failed for '{pattern}': {e}")
        
        return search_results
    
    def _extract_context(self, content: str, pattern: str, context_lines: int = 2) -> str:
        """Extract context around a pattern match"""
        try:
            lines = content.splitlines()
            pattern_lower = pattern.lower()
            
            for i, line in enumerate(lines):
                if pattern_lower in line.lower():
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = lines[start:end]
                    return "\n".join(context)
            
            return f"Pattern '{pattern}' found in file"
        except:
            return f"Pattern '{pattern}' found"
    
    def _generate_compliance_summary(self, compliance_results: Dict) -> Dict:
        """Generate compliance summary"""
        summary = {
            "overall_status": "Non-Compliant",
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "recommendations": [],
            "next_steps": []
        }
        
        overall_score = compliance_results.get("overall_compliance_score", 0)
        
        # Determine overall status
        if overall_score >= 0.9:
            summary["overall_status"] = "Fully Compliant"
        elif overall_score >= 0.7:
            summary["overall_status"] = "Mostly Compliant"
        elif overall_score >= 0.5:
            summary["overall_status"] = "Partially Compliant"
        else:
            summary["overall_status"] = "Non-Compliant"
        
        # Analyze results by framework
        for framework, results in compliance_results.get("results", {}).items():
            framework_score = results.get("compliance_score", 0)
            
            if framework_score < 0.5:
                summary["critical_issues"] += 1
                summary["recommendations"].append(f"Immediate attention required for {framework} compliance")
            elif framework_score < 0.7:
                summary["high_issues"] += 1
                summary["recommendations"].append(f"Address {framework} compliance gaps")
            elif framework_score < 0.9:
                summary["medium_issues"] += 1
                summary["recommendations"].append(f"Improve {framework} compliance")
            else:
                summary["low_issues"] += 1
        
        # Generate next steps
        if summary["critical_issues"] > 0:
            summary["next_steps"].append("Address critical compliance issues immediately")
        
        if summary["high_issues"] > 0:
            summary["next_steps"].append("Develop remediation plan for high-priority issues")
        
        if summary["medium_issues"] > 0:
            summary["next_steps"].append("Implement improvements for medium-priority issues")
        
        if summary["low_issues"] > 0:
            summary["next_steps"].append("Maintain compliance for low-priority areas")
        
        return summary
    
    async def generate_compliance_report(self, frameworks: List[str] = None, format: str = "json") -> str:
        """Generate compliance report in specified format"""
        compliance_results = await self.run_compliance_check(frameworks)
        
        if format == "json":
            return json.dumps(compliance_results, indent=2)
        elif format == "yaml":
            return yaml.dump(compliance_results, default_flow_style=False)
        elif format == "html":
            return self._generate_html_report(compliance_results)
        else:
            return str(compliance_results)
    
    def _generate_html_report(self, compliance_results: Dict) -> str:
        """Generate HTML compliance report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report - {compliance_results.get('repository', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .framework {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                .framework-header {{ background-color: #e0e0e0; padding: 10px; font-weight: bold; }}
                .control {{ margin: 10px; padding: 10px; border-left: 3px solid #ccc; }}
                .compliant {{ border-left-color: #4CAF50; }}
                .partial {{ border-left-color: #FF9800; }}
                .non-compliant {{ border-left-color: #f44336; }}
                .score {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Report</h1>
                <p><strong>Repository:</strong> {compliance_results.get('repository', 'Unknown')}</p>
                <p><strong>Generated:</strong> {compliance_results.get('timestamp', 'Unknown')}</p>
                <p><strong>Overall Compliance Score:</strong> <span class="score">{compliance_results.get('overall_compliance_score', 0):.1%}</span></p>
            </div>
        """
        
        # Add framework results
        for framework, results in compliance_results.get("results", {}).items():
            status_class = "compliant" if results.get("compliance_score", 0) >= 0.9 else "partial" if results.get("compliance_score", 0) >= 0.6 else "non-compliant"
            
            html_content += f"""
            <div class="framework">
                <div class="framework-header">
                    {results.get('framework_name', framework)} - {results.get('compliance_score', 0):.1%} Compliant
                </div>
            """
            
            # Add controls
            for control_id, control in results.get("controls", {}).items():
                control_status_class = control.get("status", "non_compliant")
                html_content += f"""
                <div class="control {control_status_class}">
                    <h3>{control.get('name', control_id)}</h3>
                    <p><strong>Status:</strong> {control.get('status', 'Unknown')}</p>
                    <p><strong>Score:</strong> {control.get('compliance_score', 0):.1%}</p>
                    <p><strong>Description:</strong> {control.get('description', 'No description')}</p>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content

# Convenience functions
async def run_compliance_check(repo_path: Path, frameworks: List[str] = None) -> Dict:
    """Run compliance check on a repository"""
    checker = ComplianceChecker(repo_path)
    return await checker.run_compliance_check(frameworks)

async def generate_compliance_report(repo_path: Path, frameworks: List[str] = None, format: str = "json") -> str:
    """Generate compliance report for a repository"""
    checker = ComplianceChecker(repo_path)
    return await checker.generate_compliance_report(frameworks, format)
