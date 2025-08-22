import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from ..core.vcs import run_cmd
from ..core.llm import enrich_findings_with_llm
from ..core.settings import settings
from .enhanced_ml_analyzer import EnhancedMLAnalyzer
from .safe_neural_analyzer import SafeNeuralAnalyzer

@dataclass
class CodeReviewFinding:
    """Structured finding for code review analysis"""
    file: str
    line: int
    severity: str
    category: str
    message: str
    suggestion: str
    code_snippet: Optional[str] = None
    before_code: Optional[str] = None
    after_code: Optional[str] = None
    autofixable: bool = False
    confidence: float = 0.0
    impact: str = "medium"
    effort: str = "medium"

class CodeReviewAgent:
    """
    Advanced Code Review Agent that analyzes code for quality improvements,
    refactoring opportunities, and reusable method suggestions.
    
    This agent works both as part of the pipeline and standalone.
    """
    
    def __init__(self, repo_path: Path, standalone: bool = False):
        self.repo_path = repo_path
        self.standalone = standalone
        self.findings: List[CodeReviewFinding] = []
        self.code_metrics = {}
        
        # Initialize ML and Neural Network analyzers
        try:
            self.ml_analyzer = EnhancedMLAnalyzer()
            self.neural_analyzer = SafeNeuralAnalyzer()
            print("🧠 ML and Neural Network analyzers initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: ML analyzers initialization failed: {e}")
            self.ml_analyzer = None
            self.neural_analyzer = None
        
    async def run_code_review(self, input_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point for code review analysis
        
        Args:
            input_data: Input from previous agent (code scanning results) or None for standalone
            
        Returns:
            Dictionary containing review results and findings
        """
        print("🔍 Starting Code Review Agent analysis...")
        
        try:
            # Initialize analysis
            await self._analyze_codebase()
            
            # Analyze security findings from previous agent if available
            if input_data and "security_findings" in input_data:
                print("🔒 Analyzing security findings for code quality implications...")
                await self._analyze_security_findings(input_data["security_findings"])
            
            # Perform specific code review analyses
            await self._analyze_code_quality()
            await self._analyze_refactoring_opportunities()
            await self._analyze_reusable_methods()
            await self._analyze_code_efficiency()
            await self._analyze_hardcoded_values()
            await self._analyze_code_duplication()
            
            # Generate comprehensive report
            report = await self._generate_review_report()
            
            # Enrich findings with ML and Neural Network analysis
            if self.ml_analyzer or self.neural_analyzer:
                await self._enrich_with_ml_analysis()
            
            # Enrich findings with LLM if available
            if not self.standalone:
                await self._enrich_with_llm()
            
            print(f"✅ Code Review completed. Found {len(self.findings)} issues.")
            return report
            
        except Exception as e:
            print(f"❌ Error in code review: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "findings": []
            }
    
    async def _analyze_codebase(self):
        """Analyze the overall codebase structure and metrics"""
        print("📊 Analyzing codebase structure...")
        
        # Get file statistics
        python_files = list(self.repo_path.rglob("*.py"))
        js_files = list(self.repo_path.rglob("*.js")) + list(self.repo_path.rglob("*.ts"))
        java_files = list(self.repo_path.rglob("*.java"))
        
        self.code_metrics = {
            "total_files": len(python_files) + len(js_files) + len(java_files),
            "python_files": len(python_files),
            "javascript_files": len(js_files),
            "java_files": len(java_files),
            "total_lines": 0,
            "complexity_score": 0
        }
        
        # Analyze Python files for complexity
        if python_files:
            await self._analyze_python_complexity(python_files)
    
    async def _analyze_python_complexity(self, python_files: List[Path]):
        """Analyze Python code complexity using radon"""
        try:
            for file_path in python_files:
                relative_path = file_path.relative_to(self.repo_path)
                rc, out, err = run_cmd(["radon", "cc", "-j", str(file_path)], self.repo_path)
                
                if rc == 0 and out.strip():
                    try:
                        data = json.loads(out)
                        for func_name, metrics in data.get(str(relative_path), []).items():
                            complexity = metrics.get('complexity', 1)
                            if complexity > 10:  # High complexity threshold
                                self.findings.append(CodeReviewFinding(
                                    file=str(relative_path),
                                    line=metrics.get('lineno', 1),
                                    severity="medium",
                                    category="complexity",
                                    message=f"Function '{func_name}' has high cyclomatic complexity ({complexity})",
                                    suggestion="Consider breaking down this function into smaller, more focused functions",
                                    code_snippet=await self._extract_function_snippet(file_path, metrics.get('lineno', 1)),
                                    confidence=0.8,
                                    impact="medium",
                                    effort="medium"
                                ))
                    except Exception as e:
                        print(f"Warning: Could not parse radon output for {file_path}: {e}")
        except Exception as e:
            print(f"Warning: Could not analyze Python complexity: {e}")
    
    async def _analyze_security_findings(self, security_findings: List[Dict]):
        """Analyze security findings and generate code quality recommendations"""
        try:
            print(f"🔒 Analyzing {len(security_findings)} security findings...")
            
            for finding in security_findings:
                # Generate code quality recommendations based on security issues
                if finding.get("category") == "security":
                    await self._generate_security_quality_recommendations(finding)
                elif finding.get("category") == "dependency":
                    await self._generate_dependency_quality_recommendations(finding)
                    
        except Exception as e:
            print(f"Warning: Could not analyze security findings: {e}")
    
    async def _generate_security_quality_recommendations(self, security_finding: Dict):
        """Generate code quality recommendations for security findings"""
        try:
            file_path = security_finding.get("file")
            line = security_finding.get("line")
            message = security_finding.get("message", "")
            tool = security_finding.get("tool", "unknown")
            
            # Generate specific recommendations based on security tool
            if tool == "bandit":
                if "sql injection" in message.lower():
                    self.findings.append(CodeReviewFinding(
                        file=file_path,
                        line=line,
                        severity="high",
                        category="security_quality",
                        message=f"Security issue detected: {message}",
                        suggestion="Refactor to use parameterized queries and input validation. Consider using an ORM with built-in SQL injection protection.",
                        code_snippet=await self._extract_code_snippet(file_path, line),
                        autofixable=False,
                        confidence=0.9,
                        impact="high",
                        effort="medium"
                    ))
                elif "hardcoded" in message.lower():
                    self.findings.append(CodeReviewFinding(
                        file=file_path,
                        line=line,
                        severity="medium",
                        category="configuration",
                        message=f"Hardcoded value detected: {message}",
                        suggestion="Move hardcoded values to environment variables or configuration files. Use a configuration management system.",
                        code_snippet=await self._extract_code_snippet(file_path, line),
                        autofixable=True,
                        confidence=0.8,
                        impact="medium",
                        effort="low"
                    ))
            
            elif tool == "semgrep":
                if "xss" in message.lower():
                    self.findings.append(CodeReviewFinding(
                        file=file_path,
                        line=line,
                        severity="high",
                        category="security_quality",
                        message=f"XSS vulnerability: {message}",
                        suggestion="Implement proper input sanitization and output encoding. Use security libraries for HTML escaping.",
                        code_snippet=await self._extract_code_snippet(file_path, line),
                        autofixable=False,
                        confidence=0.9,
                        impact="high",
                        effort="medium"
                    ))
                    
        except Exception as e:
            print(f"Warning: Could not generate security quality recommendations: {e}")
    
    async def _generate_dependency_quality_recommendations(self, dependency_finding: Dict):
        """Generate code quality recommendations for dependency findings"""
        try:
            file_path = dependency_finding.get("file")
            message = dependency_finding.get("message", "")
            
            self.findings.append(CodeReviewFinding(
                file=file_path,
                line=None,
                severity="medium",
                category="dependency_management",
                message=f"Dependency vulnerability: {message}",
                suggestion="Update vulnerable dependencies to latest secure versions. Implement dependency scanning in CI/CD pipeline.",
                code_snippet=None,
                autofixable=True,
                confidence=0.9,
                impact="medium",
                effort="low"
            ))
            
        except Exception as e:
            print(f"Warning: Could not generate dependency quality recommendations: {e}")
    
    async def _analyze_code_quality(self):
        """Analyze general code quality issues"""
        print("🎯 Analyzing code quality...")
        
        # Use ruff for Python code quality
        if self.repo_path.glob("*.py"):
            await self._run_ruff_analysis()
        
        # Use ESLint for JavaScript/TypeScript
        if self.repo_path.glob("*.js") or self.repo_path.glob("*.ts"):
            await self._run_eslint_analysis()
    
    async def _run_ruff_analysis(self):
        """Run ruff for Python code quality analysis"""
        try:
            rc, out, err = run_cmd(["ruff", "check", "--output-format", "json", "."], self.repo_path)
            if rc == 0 and out.strip():
                try:
                    data = json.loads(out)
                    for item in data:
                        # Map ruff rules to our categories
                        category = self._map_ruff_rule_to_category(item.get('code', ''))
                        
                        self.findings.append(CodeReviewFinding(
                            file=item.get('filename', ''),
                            line=item.get('location', {}).get('row', 1),
                            severity="low",
                            category=category,
                            message=item.get('message', ''),
                            suggestion=self._get_ruff_suggestion(item.get('code', '')),
                            code_snippet=await self._extract_code_snippet(
                                item.get('filename', ''), 
                                item.get('location', {}).get('row', 1)
                            ),
                            autofixable=True,
                            confidence=0.9,
                            impact="low",
                            effort="low"
                        ))
                except Exception as e:
                    print(f"Warning: Could not parse ruff output: {e}")
        except Exception as e:
            print(f"Warning: Could not run ruff analysis: {e}")
    
    async def _run_eslint_analysis(self):
        """Run ESLint for JavaScript/TypeScript code quality analysis"""
        try:
            # Check if ESLint is available
            rc, out, err = run_cmd(["eslint", "--version"], self.repo_path)
            if rc != 0:
                print("ESLint not available, skipping JavaScript analysis")
                return
            
            rc, out, err = run_cmd(["eslint", "--format", "json", "."], self.repo_path)
            if out.strip():
                try:
                    data = json.loads(out)
                    for file_data in data:
                        for message in file_data.get('messages', []):
                            category = self._map_eslint_rule_to_category(message.get('ruleId', ''))
                            
                            self.findings.append(CodeReviewFinding(
                                file=file_data.get('filePath', ''),
                                line=message.get('line', 1),
                                severity=self._map_eslint_severity(message.get('severity', 1)),
                                category=category,
                                message=message.get('message', ''),
                                suggestion=message.get('fix', {}).get('text', ''),
                                code_snippet=await self._extract_code_snippet(
                                    file_data.get('filePath', ''), 
                                    message.get('line', 1)
                                ),
                                autofixable=bool(message.get('fix')),
                                confidence=0.8,
                                impact="low",
                                effort="low"
                            ))
                except Exception as e:
                    print(f"Warning: Could not parse ESLint output: {e}")
        except Exception as e:
            print(f"Warning: Could not run ESLint analysis: {e}")
    
    async def _analyze_refactoring_opportunities(self):
        """Analyze code for refactoring opportunities"""
        print("🔄 Analyzing refactoring opportunities...")
        
        # Look for long functions
        await self._find_long_functions()
        
        # Look for large classes
        await self._find_large_classes()
        
        # Look for nested conditionals
        await self._find_nested_conditionals()
    
    async def _find_long_functions(self):
        """Find functions that are too long and could be refactored"""
        try:
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                # Simple regex to find function definitions
                function_pattern = r'^(\s*)(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    match = re.match(function_pattern, line)
                    if match:
                        # Find the end of the function
                        start_line = i + 1
                        end_line = await self._find_function_end(lines, start_line)
                        function_length = end_line - start_line + 1
                        
                        if function_length > 20:  # Threshold for long functions
                            self.findings.append(CodeReviewFinding(
                                file=str(relative_path),
                                line=start_line,
                                severity="medium",
                                category="refactoring",
                                message=f"Function '{match.group(2)}' is {function_length} lines long",
                                suggestion="Consider breaking this function into smaller, more focused functions",
                                code_snippet=await self._extract_code_snippet(str(relative_path), start_line, end_line),
                                confidence=0.7,
                                impact="medium",
                                effort="medium"
                            ))
        except Exception as e:
            print(f"Warning: Could not analyze function lengths: {e}")
    
    async def _find_large_classes(self):
        """Find classes that are too large and could be refactored"""
        try:
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                # Find class definitions
                class_pattern = r'^(\s*)class\s+(\w+)\s*[:\(]'
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    match = re.match(class_pattern, line)
                    if match:
                        # Find the end of the class
                        start_line = i + 1
                        end_line = await self._find_class_end(lines, start_line)
                        class_length = end_line - start_line + 1
                        
                        if class_length > 50:  # Threshold for large classes
                            self.findings.append(CodeReviewFinding(
                                file=str(relative_path),
                                line=start_line,
                                severity="medium",
                                category="refactoring",
                                message=f"Class '{match.group(2)}' is {class_length} lines long",
                                suggestion="Consider breaking this class into smaller, more focused classes",
                                code_snippet=await self._extract_code_snippet(str(relative_path), start_line, end_line),
                                confidence=0.7,
                                impact="medium",
                                effort="medium"
                            ))
        except Exception as e:
            print(f"Warning: Could not analyze class sizes: {e}")
    
    async def _analyze_reusable_methods(self):
        """Analyze code for opportunities to create reusable methods"""
        print("♻️ Analyzing reusable method opportunities...")
        
        try:
            # Look for code duplication patterns
            await self._find_code_duplication()
            
            # Look for similar logic that could be extracted
            await self._find_similar_logic()
            
        except Exception as e:
            print(f"Warning: Could not analyze reusable methods: {e}")
    
    async def _find_code_duplication(self):
        """Find duplicated code that could be extracted into reusable methods"""
        try:
            # Simple approach: look for repeated patterns in Python files
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                # Look for common patterns that could be extracted
                patterns = [
                    (r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', 'main guard'),
                    (r'try:\s*\n\s*.*?\nexcept\s+Exception\s+as\s+e:', 'exception handling'),
                    (r'logging\.(debug|info|warning|error)\(', 'logging statements'),
                    (r'print\s*\(', 'print statements'),
                ]
                
                for pattern, description in patterns:
                    matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
                    if len(matches) > 1:
                        # Found potential duplication
                        for i, match in enumerate(matches):
                            line_num = content[:match.start()].count('\n') + 1
                            
                            self.findings.append(CodeReviewFinding(
                                file=str(relative_path),
                                line=line_num,
                                severity="low",
                                category="reusability",
                                message=f"Potential {description} pattern that could be extracted",
                                suggestion=f"Consider creating a reusable function for {description}",
                                code_snippet=await self._extract_code_snippet(str(relative_path), line_num),
                                confidence=0.6,
                                impact="low",
                                effort="low"
                            ))
                            break  # Only add one finding per pattern per file
                            
        except Exception as e:
            print(f"Warning: Could not analyze code duplication: {e}")
    
    async def _analyze_code_efficiency(self):
        """Analyze code for efficiency improvements"""
        print("⚡ Analyzing code efficiency...")
        
        try:
            # Look for inefficient patterns
            await self._find_inefficient_patterns()
            
        except Exception as e:
            print(f"Warning: Could not analyze code efficiency: {e}")
    
    async def _find_inefficient_patterns(self):
        """Find inefficient code patterns"""
        try:
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                # Look for inefficient patterns
                inefficient_patterns = [
                    (r'for\s+\w+\s+in\s+range\(len\([^)]+\)\):', 'range(len()) pattern'),
                    (r'\.append\([^)]*\)\s*\n\s*\.append\([^)]*\)', 'multiple append calls'),
                    (r'if\s+\w+\s+in\s+\[[^\]]+\]:', 'list membership test'),
                    (r'for\s+\w+\s+in\s+\[[^\]]+\]:', 'list iteration'),
                ]
                
                for pattern, description in inefficient_patterns:
                    matches = list(re.finditer(pattern, content, re.MULTILINE))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        suggestion = self._get_efficiency_suggestion(pattern, description)
                        
                        self.findings.append(CodeReviewFinding(
                            file=str(relative_path),
                            line=line_num,
                            severity="low",
                            category="efficiency",
                            message=f"Inefficient {description} detected",
                            suggestion=suggestion,
                            code_snippet=await self._extract_code_snippet(str(relative_path), line_num),
                            confidence=0.7,
                            impact="low",
                            effort="low"
                        ))
                        
        except Exception as e:
            print(f"Warning: Could not analyze inefficient patterns: {e}")
    
    async def _analyze_hardcoded_values(self):
        """Analyze code for hardcoded values that should be configurable"""
        print("🔧 Analyzing hardcoded values...")
        
        try:
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                # Look for hardcoded values
                hardcoded_patterns = [
                    (r'["\']\d{4,}["\']', 'large numbers'),
                    (r'["\'][A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.[A-Za-z]{2,}["\']', 'email addresses'),
                    (r'["\']https?://[^\s"\']+["\']', 'URLs'),
                    (r'["\']\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}["\']', 'IP addresses'),
                    (r'["\']\w{32,}["\']', 'long strings (potential hashes/keys)'),
                ]
                
                for pattern, description in hardcoded_patterns:
                    matches = list(re.finditer(pattern, content))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(CodeReviewFinding(
                            file=str(relative_path),
                            line=line_num,
                            severity="medium",
                            category="configuration",
                            message=f"Hardcoded {description} detected",
                            suggestion="Consider moving this value to a configuration file or environment variable",
                            code_snippet=await self._extract_code_snippet(str(relative_path), line_num),
                            confidence=0.8,
                            impact="medium",
                            effort="low"
                        ))
                        
        except Exception as e:
            print(f"Warning: Could not analyze hardcoded values: {e}")
    
    async def _analyze_code_duplication(self):
        """Analyze for duplicate code blocks"""
        print("📋 Analyzing code duplication...")
        
        try:
            # This is a simplified approach - in production you might use more sophisticated tools
            await self._find_similar_functions()
            
        except Exception as e:
            print(f"Warning: Could not analyze code duplication: {e}")
    
    async def _find_similar_functions(self):
        """Find functions with similar structure"""
        try:
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                # Extract function signatures and basic structure
                functions = await self._extract_function_info(content)
                
                # Look for similar functions
                for i, func1 in enumerate(functions):
                    for j, func2 in enumerate(functions[i+1:], i+1):
                        similarity = await self._calculate_function_similarity(func1, func2)
                        
                        if similarity > 0.7:  # 70% similarity threshold
                            self.findings.append(CodeReviewFinding(
                                file=str(relative_path),
                                line=func1['start_line'],
                                severity="low",
                                category="duplication",
                                message=f"Function '{func1['name']}' is similar to '{func2['name']}'",
                                suggestion="Consider creating a common base function or using inheritance",
                                code_snippet=await self._extract_code_snippet(str(relative_path), func1['start_line'], func1['end_line']),
                                confidence=similarity,
                                impact="low",
                                effort="medium"
                            ))
                            break  # Only add one finding per similar pair
                            
        except Exception as e:
            print(f"Warning: Could not analyze function similarity: {e}")
    
    async def _enrich_with_ml_analysis(self):
        """Enrich findings with ML and Neural Network analysis"""
        print("🧠 Enriching findings with ML and Neural Network analysis...")
        
        try:
            for i, finding in enumerate(self.findings):
                # Convert finding to dict format for ML analysis
                finding_dict = {
                    "file": finding.file,
                    "line": finding.line,
                    "category": finding.category,
                    "message": finding.message,
                    "suggestion": finding.suggestion,
                    "code_snippet": finding.code_snippet,
                    "severity": finding.severity,
                    "confidence": finding.confidence,
                    "impact": finding.impact,
                    "effort": finding.effort,
                    "tool": "code_review_agent"
                }
                
                # Enhanced ML analysis
                if self.ml_analyzer:
                    try:
                        ml_results = self.ml_analyzer.analyze_finding_with_ml(finding_dict)
                        
                        # Update finding with ML insights
                        if 'predicted_severity' in ml_results:
                            finding.severity = ml_results['predicted_severity']
                        
                        if 'ensemble_confidence' in ml_results:
                            finding.confidence = ml_results['ensemble_confidence']
                        
                        # Add ML analysis to suggestion
                        if 'enhanced_analysis' in ml_results:
                            ml_analysis = ml_results['enhanced_analysis']
                            finding.suggestion = f"{finding.suggestion}\n\n🤖 ML Analysis: {ml_analysis}"
                        
                        # Add ML recommendations
                        if 'ml_recommendations' in ml_results:
                            ml_recs = ml_results['ml_recommendations']
                            if ml_recs:
                                recs_text = "\n".join([f"• {rec}" for rec in ml_recs])
                                finding.suggestion = f"{finding.suggestion}\n\n📊 ML Recommendations:\n{recs_text}"
                        
                    except Exception as e:
                        print(f"⚠️ ML analysis failed for finding {i}: {e}")
                
                # Neural Network analysis
                if self.neural_analyzer and finding.code_snippet:
                    try:
                        # Security analysis
                        security_result = self.neural_analyzer.analyze_code_security(
                            finding.code_snippet, 
                            {'file': finding.file, 'line': finding.line}
                        )
                        
                        # Quality prediction
                        quality_metrics = {
                            'lines': len(finding.code_snippet.split('\n')),
                            'complexity': 5,  # Default value
                            'nesting': 3,     # Default value
                            'imports': 5,     # Default value
                            'functions': 3,   # Default value
                            'classes': 2,     # Default value
                        }
                        
                        quality_result = self.neural_analyzer.predict_code_quality(quality_metrics)
                        
                        # Enhance finding with neural insights
                        neural_insights = []
                        
                        if security_result.get('severity') == 'high':
                            neural_insights.append(f"🚨 High security risk detected: {security_result.get('analysis', '')}")
                        
                        if quality_result.get('quality_score', 0.5) < 0.6:
                            neural_insights.append(f"⚠️ Quality concerns: {', '.join(quality_result.get('recommendations', []))}")
                        
                        if neural_insights:
                            finding.suggestion = f"{finding.suggestion}\n\n🧠 Neural Network Insights:\n" + "\n".join(neural_insights)
                        
                    except Exception as e:
                        print(f"⚠️ Neural analysis failed for finding {i}: {e}")
            
            print(f"✅ ML and Neural Network analysis completed for {len(self.findings)} findings")
            
        except Exception as e:
            print(f"❌ Error in ML analysis enrichment: {e}")
    
    async def _enrich_with_llm(self):
        """Enrich findings with LLM analysis"""
        if settings.LLM_PROVIDER == "none":
            return
            
        try:
            # Convert findings to format expected by LLM
            findings_data = []
            for finding in self.findings:
                findings_data.append({
                    "file": finding.file,
                    "line": finding.line,
                    "category": finding.category,
                    "message": finding.message,
                    "suggestion": finding.suggestion
                })
            
            # Use existing LLM enrichment
            enriched_findings = enrich_findings_with_llm(findings_data, {"repo": str(self.repo_path)})
            
            # Update our findings with enriched data
            for i, enriched in enumerate(enriched_findings):
                if i < len(self.findings):
                    if enriched.get('remediation'):
                        self.findings[i].suggestion = enriched['remediation']
                        
        except Exception as e:
            print(f"Warning: Could not enrich findings with LLM: {e}")
    
    async def _generate_review_report(self) -> Dict[str, Any]:
        """Generate comprehensive review report"""
        print("📊 Generating review report...")
        
        # Categorize findings
        categories = {}
        severities = {}
        
        for finding in self.findings:
            # Count by category
            if finding.category not in categories:
                categories[finding.category] = 0
            categories[finding.category] += 1
            
            # Count by severity
            if finding.severity not in severities:
                severities[finding.severity] = 0
            severities[finding.severity] += 1
        
        # Calculate metrics
        total_findings = len(self.findings)
        autofixable_count = sum(1 for f in self.findings if f.autofixable)
        
        report = {
            "status": "completed",
            "total_findings": total_findings,
            "findings_by_category": categories,
            "findings_by_severity": severities,
            "autofixable_count": autofixable_count,
            "code_metrics": self.code_metrics,
            "findings": [
                {
                    "file": f.file,
                    "line": f.line,
                    "severity": f.severity,
                    "category": f.category,
                    "message": f.message,
                    "suggestion": f.suggestion,
                    "code_snippet": f.code_snippet,
                    "autofixable": f.autofixable,
                    "confidence": f.confidence,
                    "impact": f.impact,
                    "effort": f.effort
                }
                for f in self.findings
            ],
            "summary": {
                "critical_issues": severities.get("critical", 0),
                "high_priority": severities.get("high", 0),
                "medium_priority": severities.get("medium", 0),
                "low_priority": severities.get("low", 0),
                "refactoring_opportunities": categories.get("refactoring", 0),
                "reusability_improvements": categories.get("reusability", 0),
                "efficiency_gains": categories.get("efficiency", 0)
            }
        }
        
        return report
    
    # Helper methods
    async def _extract_code_snippet(self, file_path: str, line: int, context_lines: int = 3) -> Optional[str]:
        """Extract code snippet around a specific line"""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return None
                
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            start_line = max(0, line - context_lines - 1)
            end_line = min(len(lines), line + context_lines)
            
            snippet_lines = []
            for i in range(start_line, end_line):
                marker = ">>> " if i == line - 1 else "    "
                snippet_lines.append(f"{i+1:4d}{marker}{lines[i].rstrip()}")
                
            return "".join(snippet_lines)
            
        except Exception as e:
            print(f"Warning: Could not extract code snippet: {e}")
            return None
    
    async def _extract_function_snippet(self, file_path: Path, line_number: int) -> Optional[str]:
        """Extract function snippet around the specified line"""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            # Find function boundaries
            start_line = max(1, line_number - 5)
            end_line = min(len(lines), line_number + 5)
            
            return await self._extract_code_snippet(str(file_path), start_line, end_line)
            
        except Exception as e:
            print(f"Warning: Could not extract function snippet: {e}")
            return None
    
    async def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a function"""
        indent_level = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            if line.strip().startswith("#"):
                continue
            if len(line) - len(line.lstrip()) <= indent_level and line.strip():
                return i - 1
        
        return len(lines) - 1
    
    async def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a class"""
        indent_level = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            if line.strip().startswith("#"):
                continue
            if len(line) - len(line.lstrip()) <= indent_level and line.strip():
                return i - 1
        
        return len(lines) - 1
    
    async def _find_nested_conditionals(self):
        """Find deeply nested conditional statements"""
        try:
            for py_file in self.repo_path.rglob("*.py"):
                relative_path = py_file.relative_to(self.repo_path)
                content = py_file.read_text()
                
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    # Count indentation levels
                    indent_level = len(line) - len(line.lstrip())
                    if indent_level > 12:  # More than 3 levels of nesting
                        # Check if it's a conditional
                        if any(keyword in line.strip() for keyword in ['if ', 'elif ', 'else:', 'for ', 'while ']):
                            self.findings.append(CodeReviewFinding(
                                file=str(relative_path),
                                line=i + 1,
                                severity="medium",
                                category="refactoring",
                                message=f"Deeply nested conditional at line {i + 1}",
                                suggestion="Consider extracting nested logic into separate functions",
                                code_snippet=await self._extract_code_snippet(str(relative_path), i + 1),
                                confidence=0.8,
                                impact="medium",
                                effort="medium"
                            ))
                            
        except Exception as e:
            print(f"Warning: Could not analyze nested conditionals: {e}")
    
    async def _find_similar_logic(self):
        """Find similar logic patterns that could be extracted"""
        # This is a placeholder for more sophisticated analysis
        pass
    
    async def _extract_function_info(self, content: str) -> List[Dict]:
        """Extract basic information about functions"""
        functions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Look for function definitions
            if re.match(r'^\s*(?:async\s+)?def\s+\w+\s*\(', line):
                func_name = re.search(r'def\s+(\w+)', line)
                if func_name:
                    end_line = await self._find_function_end(lines, i + 1)
                    functions.append({
                        'name': func_name.group(1),
                        'start_line': i + 1,
                        'end_line': end_line,
                        'content': '\n'.join(lines[i:end_line])
                    })
        
        return functions
    
    async def _calculate_function_similarity(self, func1: Dict, func2: Dict) -> float:
        """Calculate similarity between two functions"""
        # Simple similarity calculation based on content length and structure
        # In production, you might use more sophisticated algorithms
        
        content1 = func1['content']
        content2 = func2['content']
        
        # Basic similarity based on length and common patterns
        len1, len2 = len(content1), len(content2)
        min_len = min(len1, len2)
        max_len = max(len1, len2)
        
        if min_len == 0:
            return 0.0
        
        # Count common lines (simplified)
        lines1 = set(content1.split('\n'))
        lines2 = set(content2.split('\n'))
        common_lines = len(lines1.intersection(lines2))
        
        # Calculate similarity score
        similarity = (common_lines / max_len) * 0.7 + (min_len / max_len) * 0.3
        
        return min(similarity, 1.0)
    
    def _map_ruff_rule_to_category(self, rule_code: str) -> str:
        """Map ruff rule codes to our categories"""
        category_mapping = {
            'E': 'style',      # pycodestyle errors
            'W': 'style',      # pycodestyle warnings
            'F': 'quality',    # pyflakes
            'I': 'imports',    # isort
            'B': 'bugs',       # flake8-bugbear
            'C4': 'complexity', # flake8-comprehensions
            'UP': 'modernization', # pyupgrade
        }
        
        if rule_code:
            prefix = rule_code[0] if rule_code[0].isalpha() else rule_code[:2]
            return category_mapping.get(prefix, 'quality')
        
        return 'quality'
    
    def _map_eslint_rule_to_category(self, rule_id: str) -> str:
        """Map ESLint rule IDs to our categories"""
        if not rule_id:
            return 'quality'
        
        if 'complexity' in rule_id:
            return 'complexity'
        elif 'prefer' in rule_id:
            return 'style'
        elif 'no-' in rule_id:
            return 'quality'
        else:
            return 'quality'
    
    def _map_eslint_severity(self, severity: int) -> str:
        """Map ESLint severity levels to our severity levels"""
        if severity == 2:
            return "high"
        elif severity == 1:
            return "medium"
        else:
            return "low"
    
    def _get_ruff_suggestion(self, rule_code: str) -> str:
        """Get suggestion for ruff rule violations"""
        suggestions = {
            'E501': 'Line too long. Consider breaking into multiple lines.',
            'E302': 'Expected 2 blank lines before class definition.',
            'E303': 'Too many blank lines.',
            'F401': 'Unused import. Remove if not needed.',
            'F841': 'Variable assigned but never used.',
            'B006': 'Do not use mutable data structures for default arguments.',
        }
        
        return suggestions.get(rule_code, 'Review and fix according to the rule.')
    
    def _get_efficiency_suggestion(self, pattern: str, description: str) -> str:
        """Get efficiency improvement suggestions"""
        suggestions = {
            'range(len()) pattern': 'Use enumerate() instead of range(len()) for index and value.',
            'multiple append calls': 'Consider using list comprehension or extend() for better performance.',
            'list membership test': 'Use set() for faster membership testing.',
            'list iteration': 'Consider using generator expressions for large lists.',
        }
        
        return suggestions.get(description, 'Review for potential performance improvements.')
    
    async def get_standalone_report(self) -> Dict[str, Any]:
        """Get report for standalone usage"""
        if not self.standalone:
            raise ValueError("This method is only available for standalone usage")
        
        return await self._generate_review_report()
    
    async def export_findings_to_json(self, output_path: str) -> None:
        """Export findings to JSON file"""
        report = await self._generate_review_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Findings exported to {output_path}")
    
    async def export_findings_to_markdown(self, output_path: str) -> None:
        """Export findings to Markdown file"""
        report = await self._generate_review_report()
        
        with open(output_path, 'w') as f:
            f.write("# Code Review Report\n\n")
            f.write(f"**Total Findings:** {report['total_findings']}\n\n")
            
            f.write("## Summary\n\n")
            summary = report['summary']
            f.write(f"- Critical Issues: {summary['critical_issues']}\n")
            f.write(f"- High Priority: {summary['high_priority']}\n")
            f.write(f"- Medium Priority: {summary['medium_priority']}\n")
            f.write(f"- Low Priority: {summary['low_priority']}\n")
            f.write(f"- Refactoring Opportunities: {summary['refactoring_opportunities']}\n")
            f.write(f"- Reusability Improvements: {summary['reusability_improvements']}\n")
            f.write(f"- Efficiency Gains: {summary['efficiency_gains']}\n\n")
            
            f.write("## Detailed Findings\n\n")
            for finding in report['findings']:
                f.write(f"### {finding['file']}:{finding['line']}\n")
                f.write(f"**Category:** {finding['category']}\n")
                f.write(f"**Severity:** {finding['severity']}\n")
                f.write(f"**Message:** {finding['message']}\n")
                f.write(f"**Suggestion:** {finding['suggestion']}\n")
                if finding['code_snippet']:
                    f.write(f"**Code:**\n```\n{finding['code_snippet']}\n```\n")
                f.write("\n")
        
        print(f"📄 Findings exported to {output_path}")
