import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from ..core.vcs import run_cmd

class TestGenerator:
    """Generates test plans and test cases based on code changes"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.test_templates = self._load_test_templates()
    
    def _load_test_templates(self) -> Dict:
        """Load test templates for different languages and frameworks"""
        return {
            "python": {
                "pytest": {
                    "imports": [
                        "import pytest",
                        "from unittest.mock import Mock, patch, MagicMock",
                        "import sys",
                        "import os"
                    ],
                    "test_class_template": """class Test{ClassName}:
    def setup_method(self):
        \"\"\"Set up test fixtures before each test method.\"\"\"
        pass
    
    def teardown_method(self):
        \"\"\"Clean up after each test method.\"\"\"
        pass
    
    def test_{method_name}_success(self):
        \"\"\"Test successful execution of {method_name}.\"\"\"
        # TODO: Implement test
        assert True
    
    def test_{method_name}_failure(self):
        \"\"\"Test failure handling of {method_name}.\"\"\"
        # TODO: Implement test
        assert True
    
    def test_{method_name}_edge_cases(self):
        \"\"\"Test edge cases for {method_name}.\"\"\"
        # TODO: Implement test
        assert True""",
                    "test_function_template": """def test_{function_name}_success():
    \"\"\"Test successful execution of {function_name}.\"\"\"
    # TODO: Implement test
    assert True

def test_{function_name}_failure():
    \"\"\"Test failure handling of {function_name}.\"\"\"
    # TODO: Implement test
    assert True

def test_{function_name}_edge_cases():
    \"\"\"Test edge cases for {function_name}.\"\"\"
    # TODO: Implement test
    assert True"""
                }
            },
            "javascript": {
                "jest": {
                    "imports": [
                        "// Jest test file",
                        "// TODO: Add proper imports"
                    ],
                    "test_template": """describe('{ClassName}', () => {{
    let {instanceName};
    
    beforeEach(() => {{
        // TODO: Set up test fixtures
        {instanceName} = new {ClassName}();
    }});
    
    afterEach(() => {{
        // TODO: Clean up after tests
    }});
    
    describe('{methodName}', () => {{
        it('should handle success case', () => {{
            // TODO: Implement test
            expect(true).toBe(true);
        }});
        
        it('should handle failure case', () => {{
            // TODO: Implement test
            expect(true).toBe(true);
        }});
        
        it('should handle edge cases', () => {{
            // TODO: Implement test
            expect(true).toBe(true);
        }});
    }});
}});"""
                }
            }
        }
    
    async def generate_test_plan(self, changed_files: List[Dict], findings: List[Dict]) -> Dict:
        """Generate a comprehensive test plan based on code changes"""
        test_plan = {
            "summary": {
                "total_files_changed": len(changed_files),
                "total_test_cases_needed": 0,
                "critical_paths": [],
                "test_priorities": []
            },
            "test_cases": [],
            "test_files": [],
            "coverage_analysis": {},
            "recommendations": []
        }
        
        # Analyze each changed file
        for file_info in changed_files:
            if file_info["status"] == "D":  # Skip deleted files
                continue
            
            file_path = self.repo_path / file_info["path"]
            if not file_path.exists():
                continue
            
            # Generate test cases for the file
            file_tests = await self._generate_file_tests(file_path, file_info, findings)
            test_plan["test_cases"].extend(file_tests)
            
            # Generate test file
            test_file = await self._generate_test_file(file_path, file_info, file_tests)
            if test_file:
                test_plan["test_files"].append(test_file)
        
        # Analyze test coverage
        test_plan["coverage_analysis"] = await self._analyze_test_coverage(changed_files, findings)
        
        # Generate recommendations
        test_plan["recommendations"] = self._generate_test_recommendations(test_plan)
        
        # Update summary
        test_plan["summary"]["total_test_cases_needed"] = len(test_plan["test_cases"])
        
        return test_plan
    
    async def _generate_file_tests(self, file_path: Path, file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate test cases for a specific file"""
        test_cases = []
        
        try:
            if file_info["type"] == "python":
                test_cases.extend(await self._generate_python_tests(file_path, file_info, findings))
            elif file_info["type"] == "javascript":
                test_cases.extend(await self._generate_javascript_tests(file_path, file_info, findings))
            elif file_info["type"] == "yaml":
                test_cases.extend(await self._generate_yaml_tests(file_path, file_info, findings))
            elif file_info["type"] == "json":
                test_cases.extend(await self._generate_json_tests(file_path, file_info, findings))
        except Exception as e:
            # Skip files that can't be analyzed
            pass
        
        return test_cases
    
    async def _generate_python_tests(self, file_path: Path, file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate Python test cases"""
        test_cases = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse Python AST
            tree = ast.parse(content)
            
            # Find classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    test_cases.extend(self._generate_class_tests(node, file_path, file_info, findings))
                elif isinstance(node, ast.FunctionDef):
                    test_cases.extend(self._generate_function_tests(node, file_path, file_info, findings))
            
            # Generate integration tests based on findings
            test_cases.extend(self._generate_integration_tests(file_path, file_info, findings))
            
        except Exception as e:
            # If AST parsing fails, generate basic tests
            test_cases.append({
                "type": "basic_test",
                "name": f"test_{file_path.stem}_basic",
                "description": f"Basic test for {file_path.name}",
                "priority": "medium",
                "category": "unit_test",
                "file": str(file_path.relative_to(self.repo_path)),
                "test_code": f"def test_{file_path.stem}_basic():\n    \"\"\"Basic test for {file_path.name}\"\"\"\n    assert True",
                "framework": "pytest"
            })
        
        return test_cases
    
    def _generate_class_tests(self, class_node: ast.ClassDef, file_path: Path, 
                             file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate test cases for a Python class"""
        test_cases = []
        
        # Find methods in the class
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        for method in methods:
            if method.name.startswith('_'):  # Skip private methods
                continue
            
            # Check if this method has any findings
            method_findings = [f for f in findings if f.get("file") == str(file_path.relative_to(self.repo_path)) 
                             and f.get("line") == method.lineno]
            
            priority = "high" if any(f.get("severity") in ["critical", "high"] for f in method_findings) else "medium"
            
            test_cases.append({
                "type": "class_method_test",
                "name": f"test_{class_node.name}_{method.name}",
                "description": f"Test {method.name} method in {class_node.name} class",
                "priority": priority,
                "category": "unit_test",
                "file": str(file_path.relative_to(self.repo_path)),
                "class_name": class_node.name,
                "method_name": method.name,
                "line_number": method.lineno,
                "findings": method_findings,
                "test_code": self.test_templates["python"]["pytest"]["test_class_template"].format(
                    ClassName=class_node.name,
                    method_name=method.name
                ),
                "framework": "pytest"
            })
        
        return test_cases
    
    def _generate_function_tests(self, func_node: ast.FunctionDef, file_path: Path, 
                                file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate test cases for a Python function"""
        test_cases = []
        
        if func_node.name.startswith('_'):  # Skip private functions
            return test_cases
        
        # Check if this function has any findings
        func_findings = [f for f in findings if f.get("file") == str(file_path.relative_to(self.repo_path)) 
                        and f.get("line") == func_node.lineno]
        
        priority = "high" if any(f.get("severity") in ["critical", "high"] for f in func_findings) else "medium"
        
        test_cases.append({
            "type": "function_test",
            "name": f"test_{func_node.name}",
            "description": f"Test {func_node.name} function",
            "priority": priority,
            "category": "unit_test",
            "file": str(file_path.relative_to(self.repo_path)),
            "function_name": func_node.name,
            "line_number": func_node.lineno,
            "findings": func_findings,
            "test_code": self.test_templates["python"]["pytest"]["test_function_template"].format(
                function_name=func_node.name
            ),
            "framework": "pytest"
        })
        
        return test_cases
    
    async def _generate_javascript_tests(self, file_path: Path, file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate JavaScript test cases"""
        test_cases = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex-based parsing for JavaScript
            # Look for class definitions and function definitions
            class_pattern = r"class\s+(\w+)\s*\{"
            function_pattern = r"(?:function\s+)?(\w+)\s*\([^)]*\)\s*\{"
            
            classes = re.findall(class_pattern, content)
            functions = re.findall(function_pattern, content)
            
            # Generate tests for classes
            for class_name in classes:
                test_cases.append({
                    "type": "class_test",
                    "name": f"test_{class_name}",
                    "description": f"Test {class_name} class",
                    "priority": "medium",
                    "category": "unit_test",
                    "file": str(file_path.relative_to(self.repo_path)),
                    "class_name": class_name,
                    "test_code": self.test_templates["javascript"]["jest"]["test_template"].format(
                        ClassName=class_name,
                        instanceName=class_name.lower(),
                        methodName="method"
                    ),
                    "framework": "jest"
                })
            
            # Generate tests for functions
            for func_name in functions:
                if func_name not in classes:  # Avoid duplicates
                    test_cases.append({
                        "type": "function_test",
                        "name": f"test_{func_name}",
                        "description": f"Test {func_name} function",
                        "priority": "medium",
                        "category": "unit_test",
                        "file": str(file_path.relative_to(self.repo_path)),
                        "function_name": func_name,
                        "test_code": f"""describe('{func_name}', () => {{
    it('should handle success case', () => {{
        // TODO: Implement test
        expect(true).toBe(true);
    }});
    
    it('should handle failure case', () => {{
        // TODO: Implement test
        expect(true).toBe(true);
    }});
}});""",
                        "framework": "jest"
                    })
            
        except Exception as e:
            # Generate basic test if parsing fails
            test_cases.append({
                "type": "basic_test",
                "name": f"test_{file_path.stem}_basic",
                "description": f"Basic test for {file_path.name}",
                "priority": "medium",
                "category": "unit_test",
                "file": str(file_path.relative_to(self.repo_path)),
                "test_code": f"""describe('{file_path.stem}', () => {{
    it('should work correctly', () => {{
        expect(true).toBe(true);
    }});
}});""",
                "framework": "jest"
            })
        
        return test_cases
    
    async def _generate_yaml_tests(self, file_path: Path, file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate tests for YAML configuration files"""
        test_cases = []
        
        test_cases.append({
            "type": "config_test",
            "name": f"test_{file_path.stem}_valid_yaml",
            "description": f"Test that {file_path.name} is valid YAML",
            "priority": "high",
            "category": "config_test",
            "file": str(file_path.relative_to(self.repo_path)),
            "test_code": f"""import yaml
import pytest

def test_{file_path.stem}_valid_yaml():
    \"\"\"Test that {file_path.name} is valid YAML\"\"\"
    with open('{file_path.name}', 'r') as f:
        try:
            yaml.safe_load(f)
            assert True
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML: {{e}}")

def test_{file_path.stem}_required_fields():
    \"\"\"Test that {file_path.name} has required fields\"\"\"
    with open('{file_path.name}', 'r') as f:
        data = yaml.safe_load(f)
        # TODO: Add specific field checks based on your schema
        assert isinstance(data, dict)""",
            "framework": "pytest"
        })
        
        return test_cases
    
    async def _generate_json_tests(self, file_path: Path, file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate tests for JSON configuration files"""
        test_cases = []
        
        test_cases.append({
            "type": "config_test",
            "name": f"test_{file_path.stem}_valid_json",
            "description": f"Test that {file_path.name} is valid JSON",
            "priority": "high",
            "category": "config_test",
            "file": str(file_path.relative_to(self.repo_path)),
            "test_code": f"""import json
import pytest

def test_{file_path.stem}_valid_json():
    \"\"\"Test that {file_path.name} is valid JSON\"\"\"
    with open('{file_path.name}', 'r') as f:
        try:
            json.load(f)
            assert True
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON: {{e}}")

def test_{file_path.stem}_required_fields():
    \"\"\"Test that {file_path.name} has required fields\"\"\"
    with open('{file_path.name}', 'r') as f:
        data = json.load(f)
        # TODO: Add specific field checks based on your schema
        assert isinstance(data, dict)""",
            "framework": "pytest"
        })
        
        return test_cases
    
    def _generate_integration_tests(self, file_path: Path, file_info: Dict, findings: List[Dict]) -> List[Dict]:
        """Generate integration tests based on findings"""
        integration_tests = []
        
        # Look for critical findings that suggest integration issues
        critical_findings = [f for f in findings if f.get("severity") == "critical" 
                           and f.get("file") == str(file_path.relative_to(self.repo_path))]
        
        for finding in critical_findings:
            if finding.get("vulnerability_type") in ["sql_injection", "command_injection", "path_traversal"]:
                integration_tests.append({
                    "type": "integration_test",
                    "name": f"test_{file_path.stem}_{finding.get('vulnerability_type', 'security')}_integration",
                    "description": f"Integration test for {finding.get('vulnerability_type', 'security')} vulnerability",
                    "priority": "critical",
                    "category": "integration_test",
                    "file": str(file_path.relative_to(self.repo_path)),
                    "finding": finding,
                    "test_code": f"""import pytest
from unittest.mock import patch, MagicMock

def test_{file_path.stem}_{finding.get('vulnerability_type', 'security')}_integration():
    \"\"\"Integration test for {finding.get('vulnerability_type', 'security')} vulnerability\"\"\"
    # TODO: Implement integration test that validates the fix
    # This should test the actual integration point where the vulnerability was found
    assert True""",
                    "framework": "pytest"
                })
        
        return integration_tests
    
    async def _generate_test_file(self, file_path: Path, file_info: Dict, test_cases: List[Dict]) -> Optional[Dict]:
        """Generate a complete test file"""
        if not test_cases:
            return None
        
        # Determine test file path
        if file_info["type"] == "python":
            test_file_path = file_path.parent / "tests" / f"test_{file_path.name}"
        elif file_info["type"] == "javascript":
            test_file_path = file_path.parent / "__tests__" / f"{file_path.stem}.test.{file_path.suffix}"
        else:
            test_file_path = file_path.parent / "tests" / f"test_{file_path.name}"
        
        # Generate test file content
        test_content = self._generate_test_file_content(file_info, test_cases)
        
        return {
            "path": str(test_file_path.relative_to(self.repo_path)),
            "content": test_content,
            "framework": test_cases[0].get("framework", "unknown"),
            "test_cases": len(test_cases)
        }
    
    def _generate_test_file_content(self, file_info: Dict, test_cases: List[Dict]) -> str:
        """Generate the content of a test file"""
        if file_info["type"] == "python":
            return self._generate_python_test_file(file_info, test_cases)
        elif file_info["type"] == "javascript":
            return self._generate_javascript_test_file(file_info, test_cases)
        else:
            return self._generate_generic_test_file(file_info, test_cases)
    
    def _generate_python_test_file(self, file_info: Dict, test_cases: List[Dict]) -> str:
        """Generate Python test file content"""
        content = "# Generated test file\n"
        content += "# TODO: Review and customize these tests\n\n"
        
        # Add imports
        for import_line in self.test_templates["python"]["pytest"]["imports"]:
            content += f"{import_line}\n"
        content += "\n"
        
        # Add test cases
        for test_case in test_cases:
            content += f"# {test_case['description']}\n"
            content += test_case["test_code"]
            content += "\n\n"
        
        return content
    
    def _generate_javascript_test_file(self, file_info: Dict, test_cases: List[Dict]) -> str:
        """Generate JavaScript test file content"""
        content = "// Generated test file\n"
        content += "// TODO: Review and customize these tests\n\n"
        
        # Add test cases
        for test_case in test_cases:
            content += f"// {test_case['description']}\n"
            content += test_case["test_code"]
            content += "\n\n"
        
        return content
    
    def _generate_generic_test_file(self, file_info: Dict, test_cases: List[Dict]) -> str:
        """Generate generic test file content"""
        content = f"# Generated test file for {file_info['path']}\n"
        content += "# TODO: Review and customize these tests\n\n"
        
        for test_case in test_cases:
            content += f"# {test_case['description']}\n"
            content += test_case["test_code"]
            content += "\n\n"
        
        return content
    
    async def _analyze_test_coverage(self, changed_files: List[Dict], findings: List[Dict]) -> Dict:
        """Analyze test coverage for changed files"""
        coverage = {
            "files_with_tests": 0,
            "files_without_tests": 0,
            "critical_paths_tested": 0,
            "total_critical_paths": 0,
            "recommendations": []
        }
        
        for file_info in changed_files:
            if file_info["status"] == "D":
                continue
            
            file_path = self.repo_path / file_info["path"]
            test_file_path = self._find_test_file(file_path, file_info)
            
            if test_file_path and test_file_path.exists():
                coverage["files_with_tests"] += 1
            else:
                coverage["files_without_tests"] += 1
                coverage["recommendations"].append(f"Add tests for {file_info['path']}")
        
        # Count critical paths
        critical_findings = [f for f in findings if f.get("severity") == "critical"]
        coverage["total_critical_paths"] = len(critical_findings)
        
        # Count tested critical paths (simplified)
        coverage["critical_paths_tested"] = min(coverage["files_with_tests"], coverage["total_critical_paths"])
        
        return coverage
    
    def _find_test_file(self, file_path: Path, file_info: Dict) -> Optional[Path]:
        """Find the corresponding test file"""
        if file_info["type"] == "python":
            test_paths = [
                file_path.parent / "tests" / f"test_{file_path.name}",
                file_path.parent / "tests" / f"test_{file_path.stem}.py",
                file_path.parent / f"test_{file_path.name}"
            ]
        elif file_info["type"] == "javascript":
            test_paths = [
                file_path.parent / "__tests__" / f"{file_path.stem}.test.{file_path.suffix}",
                file_path.parent / f"{file_path.stem}.test.{file_path.suffix}",
                file_path.parent / f"{file_path.stem}.spec.{file_path.suffix}"
            ]
        else:
            test_paths = [
                file_path.parent / "tests" / f"test_{file_path.name}",
                file_path.parent / f"test_{file_path.name}"
            ]
        
        for test_path in test_paths:
            if test_path.exists():
                return test_path
        
        return None
    
    def _generate_test_recommendations(self, test_plan: Dict) -> List[str]:
        """Generate test recommendations based on the test plan"""
        recommendations = []
        
        # Priority-based recommendations
        high_priority_tests = [t for t in test_plan["test_cases"] if t.get("priority") == "high"]
        if high_priority_tests:
            recommendations.append(f"Focus on implementing {len(high_priority_tests)} high-priority test cases first")
        
        # Coverage recommendations
        if test_plan["coverage_analysis"]["files_without_tests"] > 0:
            recommendations.append(f"Add tests for {test_plan['coverage_analysis']['files_without_tests']} files without test coverage")
        
        # Critical path recommendations
        if test_plan["coverage_analysis"]["total_critical_paths"] > 0:
            recommendations.append(f"Ensure {test_plan['coverage_analysis']['total_critical_paths']} critical paths are properly tested")
        
        # Framework recommendations
        frameworks_used = set(t.get("framework") for t in test_plan["test_cases"])
        if len(frameworks_used) > 1:
            recommendations.append("Consider standardizing on a single testing framework for consistency")
        
        # Integration test recommendations
        integration_tests = [t for t in test_plan["test_cases"] if t.get("category") == "integration_test"]
        if not integration_tests:
            recommendations.append("Add integration tests for critical functionality and API endpoints")
        
        return recommendations

async def generate_test_plan_for_changes(changed_files: List[Dict], findings: List[Dict], repo_path: Path) -> Dict:
    """Generate a comprehensive test plan for code changes"""
    generator = TestGenerator(repo_path)
    return await generator.generate_test_plan(changed_files, findings)
