import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ..core.vcs import run_cmd

class EnhancedAnalyzer:
    """Enhanced analysis with code snippet extraction, risk scoring, and fix suggestions"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def extract_code_snippets(self, findings: List[Dict]) -> List[Dict]:
        """Extract relevant code snippets for each finding"""
        enhanced_findings = []
        
        for finding in findings:
            enhanced_finding = finding.copy()
            
            # Extract code snippet if file and line are available
            if finding.get('file') and finding.get('line'):
                snippet = await self._extract_code_snippet(
                    finding['file'], 
                    finding['line'],
                    finding.get('tool', '')
                )
                enhanced_finding['code_snippet'] = snippet
            
            # Add context information
            enhanced_finding['context'] = await self._get_finding_context(finding)
            
            enhanced_findings.append(enhanced_finding)
        
        return enhanced_findings
    
    async def _extract_code_snippet(self, file_path: str, line_number: int, tool: str) -> Dict:
        """Extract code snippet around the specified line"""
        try:
            # Convert relative path to absolute
            if file_path.startswith('/'):
                abs_file_path = Path(file_path)
            else:
                abs_file_path = self.repo_path / file_path
            
            if not abs_file_path.exists():
                return {"error": "File not found"}
            
            # Read file content
            content = abs_file_path.read_text()
            lines = content.split('\n')
            
            if line_number > len(lines):
                return {"error": "Line number out of range"}
            
            # Determine context window based on tool and issue type
            context_lines = self._get_context_window_size(tool)
            
            # Calculate start and end lines for snippet
            start_line = max(1, line_number - context_lines)
            end_line = min(len(lines), line_number + context_lines)
            
            # Extract snippet
            snippet_lines = lines[start_line - 1:end_line]
            
            # Add line numbers and highlight the target line
            formatted_snippet = []
            for i, line in enumerate(snippet_lines, start_line):
                if i == line_number:
                    # Highlight the problematic line
                    formatted_snippet.append(f"  {i:4d} >>> {line}")
                else:
                    formatted_snippet.append(f"  {i:4d}     {line}")
            
            return {
                "start_line": start_line,
                "end_line": end_line,
                "target_line": line_number,
                "content": "\n".join(formatted_snippet),
                "file_path": str(file_path),
                "total_lines": len(lines)
            }
            
        except Exception as e:
            return {"error": f"Failed to extract snippet: {str(e)}"}
    
    def _get_context_window_size(self, tool: str) -> int:
        """Determine how many lines of context to show based on the tool"""
        context_sizes = {
            'pip-audit': 3,      # Security issues - show dependency lines
            'ruff': 5,           # Code quality - show more context
            'mypy': 7,           # Type issues - show function/class context
            'bandit': 5,         # Security issues - show function context
            'eslint': 5,         # JavaScript issues - show function context
            'default': 5
        }
        
        return context_sizes.get(tool, context_sizes['default'])
    
    async def _get_finding_context(self, finding: Dict) -> Dict:
        """Get additional context for a finding"""
        try:
            file_path = finding.get('file', '')
            line_number = finding.get('line')
            tool = finding.get('tool', '')
            
            if not file_path or not line_number:
                return {}
            
            # Convert relative path to absolute
            if file_path.startswith('/'):
                abs_file_path = Path(file_path)
            else:
                abs_file_path = self.repo_path / file_path
            
            if not abs_file_path.exists():
                return {}
            
            context = {
                "file_type": abs_file_path.suffix,
                "file_size": abs_file_path.stat().st_size,
                "is_test_file": self._is_test_file(file_path),
                "is_config_file": self._is_config_file(file_path),
                "function_context": await self._get_function_context(abs_file_path, line_number),
                "import_context": await self._get_import_context(abs_file_path, tool)
            }
            
            return context
            
        except Exception as e:
            return {"error": f"Failed to get context: {str(e)}"}
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if the file is a test file"""
        test_patterns = [
            'test_', '_test', 'tests/', 'test/', 'spec.', '.spec.', '.test.'
        ]
        
        file_lower = file_path.lower()
        return any(pattern in file_lower for pattern in test_patterns)
    
    def _is_config_file(self, file_path: str) -> bool:
        """Check if the file is a configuration file"""
        config_patterns = [
            'requirements.txt', 'package.json', 'setup.py', 'pyproject.toml',
            'tox.ini', '.flake8', '.eslintrc', 'tsconfig.json', 'webpack.config'
        ]
        
        file_lower = file_path.lower()
        return any(pattern in file_lower for pattern in config_patterns)
    
    async def _get_function_context(self, file_path: Path, line_number: int) -> Dict:
        """Get the function/class context for a line"""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            if line_number > len(lines):
                return {}
            
            # Look for function/class definitions above the line
            function_context = {}
            current_line = line_number - 1  # Convert to 0-based index
            
            # Look backwards for function/class definition
            for i in range(current_line, -1, -1):
                line = lines[i].strip()
                
                # Check for function definition
                if re.match(r'^def\s+\w+', line):
                    function_context['function'] = line
                    function_context['function_line'] = i + 1
                    break
                
                # Check for class definition
                elif re.match(r'^class\s+\w+', line):
                    function_context['class'] = line
                    function_context['class_line'] = i + 1
                    break
            
            # Look forwards for function/class end
            for i in range(current_line, len(lines)):
                line = lines[i].strip()
                
                # Check for function/class end (simplified)
                if line.startswith('def ') or line.startswith('class '):
                    if 'function' in function_context or 'class' in function_context:
                        function_context['end_line'] = i
                        break
            
            return function_context
            
        except Exception:
            return {}
    
    async def _get_import_context(self, file_path: Path, tool: str) -> Dict:
        """Get import context for the file"""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            imports = []
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for Python imports
                if tool in ['pip-audit', 'ruff', 'mypy', 'bandit']:
                    if (line_stripped.startswith('import ') or 
                        line_stripped.startswith('from ') or
                        line_stripped.startswith('importlib.import_module')):
                        imports.append({
                            "line": i,
                            "content": line_stripped,
                            "type": "python"
                        })
                
                # Check for JavaScript imports
                elif tool in ['eslint', 'prettier']:
                    if (line_stripped.startswith('import ') or 
                        line_stripped.startswith('const ') or
                        line_stripped.startswith('let ') or
                        line_stripped.startswith('var ')):
                        imports.append({
                            "line": i,
                            "content": line_stripped,
                            "type": "javascript"
                        })
            
            return {
                "imports": imports,
                "total_imports": len(imports)
            }
            
        except Exception:
            return {"imports": [], "total_imports": 0}
    
    async def calculate_risk_scores(self, findings: List[Dict]) -> List[Dict]:
        """Calculate comprehensive risk scores for findings"""
        enhanced_findings = []
        
        for finding in findings:
            enhanced_finding = finding.copy()
            
            # Calculate base risk score
            base_score = self._calculate_base_risk_score(finding)
            
            # Calculate context risk multiplier
            context_multiplier = self._calculate_context_risk_multiplier(finding)
            
            # Calculate final risk score
            final_score = base_score * context_multiplier
            
            # Add risk assessment
            enhanced_finding['risk_assessment'] = {
                "base_score": base_score,
                "context_multiplier": context_multiplier,
                "final_score": final_score,
                "risk_level": self._get_risk_level(final_score),
                "priority": self._get_priority_level(final_score),
                "factors": self._get_risk_factors(finding)
            }
            
            enhanced_findings.append(enhanced_finding)
        
        return enhanced_findings
    
    def _calculate_base_risk_score(self, finding: Dict) -> float:
        """Calculate base risk score based on severity and tool"""
        # Base scores by severity
        severity_scores = {
            'critical': 10.0,
            'high': 8.0,
            'medium': 5.0,
            'low': 2.0,
            'info': 1.0
        }
        
        # Tool-specific multipliers
        tool_multipliers = {
            'pip-audit': 1.5,      # Security vulnerabilities
            'bandit': 1.4,         # Security issues
            'mypy': 1.2,           # Type safety issues
            'ruff': 1.0,           # Code quality
            'eslint': 1.1,         # JavaScript issues
            'default': 1.0
        }
        
        # Get severity score
        severity = finding.get('severity', 'medium')
        if isinstance(severity, str) and 'Severity.' in severity:
            severity = severity.split('.')[-1].lower()
        
        base_score = severity_scores.get(severity.lower(), 5.0)
        
        # Apply tool multiplier
        tool = finding.get('tool', 'default')
        tool_multiplier = tool_multipliers.get(tool, tool_multipliers['default'])
        
        return base_score * tool_multiplier
    
    def _calculate_context_risk_multiplier(self, finding: Dict) -> float:
        """Calculate context risk multiplier based on file and location"""
        multiplier = 1.0
        
        file_path = finding.get('file', '')
        line_number = finding.get('line')
        
        # File type multipliers
        if self._is_config_file(file_path):
            multiplier *= 1.3  # Config files are important
        
        if self._is_test_file(file_path):
            multiplier *= 0.7  # Test files are less critical
        
        # Location multipliers
        if line_number:
            if line_number <= 10:
                multiplier *= 1.2  # Top of file is more visible
            elif line_number >= 100:
                multiplier *= 1.1  # Deep in file might be complex
        
        # Autofixable multiplier
        if finding.get('autofixable', False):
            multiplier *= 0.9  # Auto-fixable issues are less risky
        
        return multiplier
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level"""
        if score >= 8.0:
            return "Critical"
        elif score >= 6.0:
            return "High"
        elif score >= 4.0:
            return "Medium"
        elif score >= 2.0:
            return "Low"
        else:
            return "Info"
    
    def _get_priority_level(self, score: float) -> str:
        """Convert risk score to priority level"""
        if score >= 8.0:
            return "Immediate"
        elif score >= 6.0:
            return "High"
        elif score >= 4.0:
            return "Medium"
        elif score >= 2.0:
            return "Low"
        else:
            return "Optional"
    
    def _get_risk_factors(self, finding: Dict) -> List[str]:
        """Get list of risk factors for a finding"""
        factors = []
        
        # Severity-based factors
        severity = finding.get('severity', 'medium')
        if isinstance(severity, str) and 'Severity.' in severity:
            severity = severity.split('.')[-1].lower()
        
        if severity in ['critical', 'high']:
            factors.append("High severity issue")
        
        # Tool-based factors
        tool = finding.get('tool', '')
        if tool == 'pip-audit':
            factors.append("Security vulnerability")
        elif tool == 'bandit':
            factors.append("Security concern")
        elif tool == 'mypy':
            factors.append("Type safety issue")
        
        # File-based factors
        file_path = finding.get('file', '')
        if self._is_config_file(file_path):
            factors.append("Configuration file")
        
        # Autofixable factors
        if finding.get('autofixable', False):
            factors.append("Automatically fixable")
        else:
            factors.append("Requires manual intervention")
        
        return factors
    
    async def generate_fix_suggestions(self, findings: List[Dict]) -> List[Dict]:
        """Generate detailed fix suggestions for findings"""
        enhanced_findings = []
        
        for finding in findings:
            enhanced_finding = finding.copy()
            
            # Generate fix suggestions
            suggestions = await self._generate_fix_suggestions_for_finding(finding)
            enhanced_finding['fix_suggestions'] = suggestions
            
            enhanced_findings.append(enhanced_finding)
        
        return enhanced_findings
    
    async def _generate_fix_suggestions_for_finding(self, finding: Dict) -> Dict:
        """Generate fix suggestions for a specific finding"""
        tool = finding.get('tool', '')
        rule_id = finding.get('rule_id', '')
        message = finding.get('message', '')
        remediation = finding.get('remediation', '')
        
        suggestions = {
            "automatic_fix": finding.get('autofixable', False),
            "manual_steps": [],
            "code_examples": [],
            "tools_to_use": [],
            "estimated_time": "Unknown",
            "difficulty": "Unknown"
        }
        
        # Generate tool-specific suggestions
        if tool == 'pip-audit':
            suggestions.update(await self._generate_security_fix_suggestions(finding))
        elif tool == 'ruff':
            suggestions.update(await self._generate_code_quality_suggestions(finding))
        elif tool == 'mypy':
            suggestions.update(await self._generate_type_fix_suggestions(finding))
        else:
            suggestions.update(await self._generate_generic_fix_suggestions(finding))
        
        return suggestions
    
    async def _generate_security_fix_suggestions(self, finding: Dict) -> Dict:
        """Generate security fix suggestions"""
        remediation = finding.get('remediation', '')
        
        suggestions = {
            "manual_steps": [
                "Update the vulnerable dependency to the recommended version",
                "Test the application thoroughly after the update",
                "Check for breaking changes in the new version",
                "Update any related dependencies if necessary"
            ],
            "code_examples": [],
            "tools_to_use": ["pip", "npm", "yarn", "poetry"],
            "estimated_time": "5-15 minutes",
            "difficulty": "Low"
        }
        
        # Add specific remediation steps
        if remediation:
            suggestions["manual_steps"].insert(0, f"Follow this specific instruction: {remediation}")
        
        return suggestions
    
    async def _generate_code_quality_suggestions(self, finding: Dict) -> Dict:
        """Generate code quality fix suggestions"""
        rule_id = finding.get('rule_id', '')
        
        suggestions = {
            "manual_steps": [
                "Review the code around the reported line",
                "Apply the suggested fix manually",
                "Run the linter again to verify the fix",
                "Ensure the fix doesn't introduce new issues"
            ],
            "code_examples": [],
            "tools_to_use": ["ruff", "black", "flake8"],
            "estimated_time": "2-10 minutes",
            "difficulty": "Low"
        }
        
        # Add rule-specific suggestions
        if rule_id == 'F401':  # Unused imports
            suggestions["manual_steps"].insert(1, "Remove the unused import statement")
            suggestions["code_examples"].append({
                "before": "import unused_module",
                "after": "# import unused_module  # Removed unused import"
            })
        
        elif rule_id == 'F841':  # Unused variables
            suggestions["manual_steps"].insert(1, "Either use the variable or prefix it with underscore")
            suggestions["code_examples"].append({
                "before": "unused_var = some_function()",
                "after": "_unused_var = some_function()  # Prefixed with underscore"
            })
        
        return suggestions
    
    async def _generate_type_fix_suggestions(self, finding: Dict) -> Dict:
        """Generate type checking fix suggestions"""
        message = finding.get('message', '')
        
        suggestions = {
            "manual_steps": [
                "Install missing type stubs if available",
                "Add proper type annotations",
                "Create missing __init__.py files for packages",
                "Fix import statements"
            ],
            "code_examples": [],
            "tools_to_use": ["mypy", "pip", "types-* packages"],
            "estimated_time": "5-20 minutes",
            "difficulty": "Medium"
        }
        
        # Add message-specific suggestions
        if "Cannot find implementation or library stub" in message:
            suggestions["manual_steps"].insert(0, "Install the types-* package for the missing module")
        
        elif "Cannot find module" in message:
            suggestions["manual_steps"].insert(0, "Check if the module path is correct")
            suggestions["manual_steps"].insert(1, "Verify the module is installed")
        
        return suggestions
    
    async def _generate_generic_fix_suggestions(self, finding: Dict) -> Dict:
        """Generate generic fix suggestions"""
        return {
            "manual_steps": [
                "Review the error message carefully",
                "Check the file and line number mentioned",
                "Apply appropriate fixes based on the tool's documentation",
                "Test the fix to ensure it resolves the issue"
            ],
            "code_examples": [],
            "tools_to_use": ["Manual review"],
            "estimated_time": "5-30 minutes",
            "difficulty": "Variable"
        }
    
    async def analyze_findings_comprehensive(self, findings: List[Dict]) -> Dict:
        """Perform comprehensive analysis of all findings"""
        print("🔍 Starting comprehensive analysis...")
        
        # Step 1: Extract code snippets
        print("📝 Extracting code snippets...")
        findings_with_snippets = await self.extract_code_snippets(findings)
        
        # Step 2: Calculate risk scores
        print("⚠️ Calculating risk scores...")
        findings_with_risk = await self.calculate_risk_scores(findings_with_snippets)
        
        # Step 3: Generate fix suggestions
        print("💡 Generating fix suggestions...")
        findings_with_suggestions = await self.generate_fix_suggestions(findings_with_risk)
        
        # Step 4: Generate summary
        print("📊 Generating analysis summary...")
        summary = self._generate_analysis_summary(findings_with_suggestions)
        
        return {
            "findings": findings_with_suggestions,
            "summary": summary,
            "analysis_metadata": {
                "total_findings": len(findings),
                "findings_with_snippets": len([f for f in findings_with_suggestions if f.get('code_snippet') and 'error' not in f.get('code_snippet', {})]),
                "findings_with_risk_assessment": len([f for f in findings_with_suggestions if f.get('risk_assessment')]),
                "findings_with_suggestions": len([f for f in findings_with_suggestions if f.get('fix_suggestions')])
            }
        }
    
    def _generate_analysis_summary(self, findings: List[Dict]) -> Dict:
        """Generate a comprehensive summary of the analysis"""
        total_findings = len(findings)
        
        # Risk level distribution
        risk_levels = {}
        priority_levels = {}
        tools_used = {}
        
        for finding in findings:
            risk_assessment = finding.get('risk_assessment', {})
            risk_level = risk_assessment.get('risk_level', 'Unknown')
            priority_level = risk_assessment.get('priority', 'Unknown')
            tool = finding.get('tool', 'Unknown')
            
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            priority_levels[priority_level] = priority_levels.get(priority_level, 0) + 1
            tools_used[tool] = tools_used.get(tool, 0) + 1
        
        # Calculate average risk score
        risk_scores = [f.get('risk_assessment', {}).get('final_score', 0) for f in findings]
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Find highest priority issues
        critical_issues = [f for f in findings if f.get('risk_assessment', {}).get('risk_level') == 'Critical']
        high_priority_issues = [f for f in findings if f.get('risk_assessment', {}).get('priority') in ['Immediate', 'High']]
        
        return {
            "total_findings": total_findings,
            "risk_distribution": risk_levels,
            "priority_distribution": priority_levels,
            "tools_used": tools_used,
            "average_risk_score": round(avg_risk_score, 2),
            "critical_issues_count": len(critical_issues),
            "high_priority_count": len(high_priority_issues),
            "recommendations": self._generate_recommendations(findings, avg_risk_score)
        }
    
    def _generate_recommendations(self, findings: List[Dict], avg_risk_score: float) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if avg_risk_score >= 7.0:
            recommendations.append("🚨 CRITICAL: Address high-risk security issues immediately")
        
        if avg_risk_score >= 5.0:
            recommendations.append("⚠️ HIGH: Focus on security and critical issues first")
        
        # Tool-specific recommendations
        pip_audit_findings = [f for f in findings if f.get('tool') == 'pip-audit']
        if pip_audit_findings:
            recommendations.append("🔒 Update vulnerable dependencies to fix security issues")
        
        ruff_findings = [f for f in findings if f.get('tool') == 'ruff']
        if ruff_findings:
            recommendations.append("✨ Fix code quality issues to improve maintainability")
        
        mypy_findings = [f for f in findings if f.get('tool') == 'mypy']
        if mypy_findings:
            recommendations.append("🔍 Install type stubs and fix type annotations")
        
        # Autofix recommendations
        autofixable_findings = [f for f in findings if f.get('autofixable', False)]
        if autofixable_findings:
            recommendations.append(f"🤖 {len(autofixable_findings)} issues can be fixed automatically")
        
        return recommendations
