# 🔗 Agent Data Flow Specification

## Overview

This document specifies the exact data structures and flow between agents in our Agentic SDLC system. It details what each agent expects as input and what it produces as output, ensuring seamless integration between Code Scanning, Code Review, and Testing agents.

---

## 📥 INPUT: What Code Review Agent Expects from Code Scanning Agent

### Data Structure

```python
input_data = {
    "security_findings": [
        {
            "tool": "bandit",                    # Security tool used
            "severity": "high",                  # low/medium/high/critical
            "file": "app/auth.py",              # File path
            "line": 45,                         # Line number
            "rule_id": "B608",                  # Rule identifier
            "message": "SQL injection vulnerability", # Issue description
            "category": "security",             # Finding category
            "stage": "security_scanning",       # Analysis stage
            "autofixable": False                # Whether it can be auto-fixed
        },
        {
            "tool": "pip-audit",
            "severity": "medium",
            "file": "requirements.txt",
            "line": None,
            "rule_id": "CVE-2023-1234",
            "message": "Django 3.2.0: SQL injection vulnerability",
            "category": "dependency",
            "stage": "security_scanning",
            "autofixable": True
        }
    ]
}
```

### Expected Security Finding Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **`security`** | Direct code vulnerabilities | SQL injection, XSS, command injection |
| **`dependency`** | Vulnerable third-party packages | Outdated packages with CVEs |
| **`quality`** | Code quality issues | Style violations, unused imports |
| **`complexity`** | High cyclomatic complexity | Functions with too many branches |

### Required Fields

- ✅ **`tool`**: Must be present (bandit, semgrep, pip-audit, npm-audit, ruff, radon)
- ✅ **`severity`**: Must be one of: low, medium, high, critical
- ✅ **`file`**: File path (relative to repository root)
- ✅ **`message`**: Description of the issue
- ✅ **`category`**: Type of finding

### Optional Fields

- 🔶 **`line`**: Line number (can be None for dependency issues)
- 🔶 **`rule_id`**: Rule identifier
- 🔶 **`autofixable`**: Whether issue can be auto-fixed
- 🔶 **`stage`**: Analysis stage identifier

---

## 📤 OUTPUT: What Code Review Agent Sends to Testing Agents

### Data Structure

```python
code_review_output = {
    "status": "completed",
    "total_findings": 25,
    "findings_by_category": {
        "security_quality": 8,      # Security-related quality issues
        "refactoring": 6,           # Refactoring opportunities
        "reusability": 4,           # Reusable method suggestions
        "efficiency": 3,            # Performance improvements
        "configuration": 2,          # Hardcoded values
        "complexity": 2             # High complexity functions
    },
    "findings_by_severity": {
        "critical": 3,
        "high": 8,
        "medium": 10,
        "low": 4
    },
    "autofixable_count": 12,
    "code_metrics": {
        "total_files": 45,
        "python_files": 30,
        "javascript_files": 15,
        "total_lines": 2500,
        "complexity_score": 7.2
    },
    "findings": [
        {
            "file": "app/auth.py",
            "line": 45,
            "severity": "high",
            "category": "security_quality",
            "message": "Security issue detected: SQL injection vulnerability",
            "suggestion": "Refactor to use parameterized queries and input validation. Consider using an ORM with built-in SQL injection protection.",
            "code_snippet": "45>>> def authenticate_user(username, password):\n46>>>     query = f\"SELECT * FROM users WHERE username='{username}'\"\n47>>>     # ... rest of function",
            "autofixable": False,
            "confidence": 0.9,
            "impact": "high",
            "effort": "medium"
        },
        {
            "file": "app/utils.py",
            "line": 120,
            "severity": "medium",
            "category": "refactoring",
            "message": "Function 'process_data' has high cyclomatic complexity (15)",
            "suggestion": "Consider breaking down this function into smaller, more focused functions",
            "code_snippet": "118>>> def process_data(data, config):\n119>>>     # ... complex logic\n120>>>     if condition1 and condition2 or condition3:\n121>>>     # ... more logic",
            "autofixable": True,
            "confidence": 0.8,
            "impact": "medium",
            "effort": "medium"
        }
    ],
    "summary": {
        "critical_issues": 3,
        "high_priority": 8,
        "medium_priority": 10,
        "low_priority": 4,
        "refactoring_opportunities": 6,
        "reusability_improvements": 4,
        "efficiency_gains": 3
    }
}
```

### Finding Categories Produced

| Category | Description | Severity Levels |
|----------|-------------|-----------------|
| **`security_quality`** | Security-related code quality issues | high, critical |
| **`refactoring`** | Code refactoring opportunities | medium, high |
| **`reusability`** | Reusable method suggestions | medium |
| **`efficiency`** | Performance improvements | medium |
| **`configuration`** | Hardcoded values, env vars | medium |
| **`complexity`** | High complexity functions | medium, high |

---

## 🔄 Data Transformation: How Code Review Agent Processes Input

### 1. Security Findings Analysis

```python
# Code Review Agent receives security findings and generates quality recommendations
if finding.get("category") == "security":
    if finding.get("tool") == "bandit":
        if "sql injection" in finding.get("message", "").lower():
            # Generate security quality finding
            self.findings.append(CodeReviewFinding(
                file=finding.get("file"),
                line=finding.get("line"),
                severity="high",
                category="security_quality",
                message=f"Security issue detected: {finding.get('message')}",
                suggestion="Refactor to use parameterized queries and input validation...",
                autofixable=False,
                confidence=0.9,
                impact="high",
                effort="medium"
            ))
```

### 2. Dependency Vulnerability Analysis

```python
# Code Review Agent analyzes dependency issues for configuration management
if finding.get("category") == "dependency":
    self.findings.append(CodeReviewFinding(
        file=finding.get("file"),
        line=None,
        severity="medium",
        category="dependency_management",
        message=f"Dependency vulnerability: {finding.get('message')}",
        suggestion="Update vulnerable dependencies to latest secure versions...",
        autofixable=True,
        confidence=0.9,
        impact="medium",
        effort="low"
    ))
```

### 3. Code Quality Enhancement

```python
# Code Review Agent adds its own quality findings
await self._analyze_code_quality()           # Style, imports, syntax
await self._analyze_refactoring_opportunities() # Long functions, large classes
await self._analyze_reusable_methods()       # Code duplication, extract methods
await self._analyze_code_efficiency()        # Performance patterns
await self._analyze_hardcoded_values()       # Configuration management
await self._analyze_code_duplication()       # Similar code patterns
```

---

## 🧪 What Testing Agents Receive and Use

### Test Generation Input

```python
# Test Generator receives findings from Code Review Agent
test_plan = await test_generator.generate_test_plan(
    changed_files=[{
        "path": "app/auth.py",
        "type": "python",
        "findings": [
            {
                "file": "app/auth.py",
                "line": 45,
                "severity": "high",
                "category": "security_quality",
                "message": "Security issue detected: SQL injection vulnerability",
                "suggestion": "Refactor to use parameterized queries...",
                "autofixable": False,
                "confidence": 0.9,
                "impact": "high",
                "effort": "medium"
            }
        ]
    }],
    findings=all_findings  # All findings from Code Review Agent
)
```

### Test Generation Output

```python
test_plan = {
    "total_files": 1,
    "test_files": [
        {
            "path": "tests/test_auth.py",
            "content": "# Generated test file content",
            "framework": "pytest",
            "test_cases": 5
        }
    ],
    "coverage_analysis": {
        "total_critical_paths": 3,
        "test_coverage_percentage": 85,
        "uncovered_critical_paths": []
    },
    "priority_tests": [
        {
            "file": "app/auth.py",
            "priority": "high",
            "reason": "Critical findings: 1",
            "findings": [/* high severity findings */]
        }
    ]
}
```

---

## 📋 Data Validation Requirements

### For Code Scanning Agent Output

```python
# Required validation
assert "security_findings" in input_data
assert isinstance(input_data["security_findings"], list)

for finding in input_data["security_findings"]:
    assert "tool" in finding
    assert "severity" in finding
    assert "file" in finding
    assert "message" in finding
    assert "category" in finding
    
    # Validate severity values
    assert finding["severity"] in ["low", "medium", "high", "critical"]
    
    # Validate category values
    assert finding["category"] in ["security", "dependency", "quality", "complexity"]
```

### For Code Review Agent Output

```python
# Required validation
assert "status" in output
assert "findings" in output
assert "summary" in output

# Validate findings structure
for finding in output["findings"]:
    assert "file" in finding
    assert "severity" in finding
    assert "category" in finding
    assert "message" in finding
    assert "suggestion" in finding
    
    # Validate severity values
    assert finding["severity"] in ["low", "medium", "high", "critical"]
    
    # Validate category values
    assert finding["category"] in [
        "security_quality", "refactoring", "reusability", 
        "efficiency", "configuration", "complexity"
    ]
```

---

## 🚀 Integration Example

### Complete Data Flow

```python
# 1. Code Scanning Agent sends findings
security_findings = [
    {
        "tool": "bandit",
        "severity": "high",
        "file": "app/auth.py",
        "line": 45,
        "message": "SQL injection vulnerability",
        "category": "security"
    }
]

# 2. Code Review Agent receives and processes
input_data = {"security_findings": security_findings}
code_review_results = await code_review_agent.run_code_review(input_data)

# 3. Code Review Agent sends enhanced findings to Testing Agent
test_plan = await test_generator.generate_test_plan(
    changed_files=[{"path": "app/auth.py", "type": "python"}],
    findings=code_review_results["findings"]  # Enhanced findings with quality recommendations
)

# 4. Testing Agent generates priority tests based on findings
priority_tests = test_plan["priority_tests"]
# Result: High-priority tests for critical security issues
```

### Pipeline Execution Flow

```
Code Scanning Agent
       ↓
   [Security Findings]
       ↓
Code Review Agent
       ↓
[Enhanced Findings + Quality Issues]
       ↓
Test Generation Agent
       ↓
[Priority Test Plan + Coverage Analysis]
```

---

## 🔧 Implementation Notes

### Error Handling

- **Missing Fields**: If required fields are missing, the agent will log warnings and continue
- **Invalid Data**: Invalid severity or category values will be mapped to defaults
- **File Not Found**: If referenced files don't exist, findings will be marked as invalid

### Performance Considerations

- **Large Repositories**: Analysis is performed incrementally to handle large codebases
- **Caching**: Results are cached to avoid re-analyzing unchanged files
- **Parallel Processing**: Multiple analysis tools run in parallel where possible

### Extensibility

- **New Tools**: Additional security tools can be added by extending the pipeline
- **Custom Rules**: Project-specific rules can be configured via configuration files
- **Plugin System**: Analysis modules can be added as plugins

---

## 📚 Related Documentation

- [Code Review Agent README](CODE_REVIEW_AGENT_README.md)
- [Enhanced Features Documentation](ENHANCED_FEATURES.md)
- [API Reference](backend/app/api/)
- [CLI Usage](backend/app/cli/)

---

## 🎯 Summary

This data flow specification ensures that:

- **Security findings** from Code Scanning Agent are **enriched** with quality recommendations
- **Testing priorities** are based on **actual findings** and severity levels
- **Code quality** and **security** are **integrated** rather than analyzed separately
- **Test coverage** focuses on **critical paths** identified by the analysis
- **Agent communication** follows **standardized data structures** for seamless integration

Your Code Review Agent is now a **real bridge** between security scanning and quality testing, with well-defined interfaces that other agents can rely on! 🎉
