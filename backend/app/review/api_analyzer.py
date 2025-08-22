import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from ..core.vcs import run_cmd

class APIAnalyzer:
    """Analyzes APIs for breaking changes and compatibility issues"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.api_patterns = self._load_api_patterns()
    
    def _load_api_patterns(self) -> Dict:
        """Load patterns for API analysis"""
        return {
            "python_frameworks": {
                "fastapi": {
                    "route_patterns": [
                        r"@app\.(get|post|put|delete|patch)\s*\(\s*[\"'][^\"']*[\"']",
                        r"@router\.(get|post|put|delete|patch)\s*\(\s*[\"'][^\"']*[\"']",
                        r"@api_router\.(get|post|put|delete|patch)\s*\(\s*[\"'][^\"']*[\"']",
                    ],
                    "model_patterns": [
                        r"class\s+\w+\(BaseModel\):",
                        r"class\s+\w+\(SQLModel\):",
                        r"class\s+\w+\(PydanticBaseModel\):",
                    ],
                    "response_patterns": [
                        r"Response\s*\[[^\]]*\]",
                        r"List\s*\[[^\]]*\]",
                        r"Dict\s*\[[^,]*,\s*[^\]]*\]",
                    ]
                },
                "flask": {
                    "route_patterns": [
                        r"@app\.route\s*\(\s*[\"'][^\"']*[\"']",
                        r"@blueprint\.route\s*\(\s*[\"'][^\"']*[\"']",
                    ],
                    "model_patterns": [
                        r"class\s+\w+\(db\.Model\):",
                        r"class\s+\w+\(marshmallow\.Schema\):",
                    ]
                },
                "django": {
                    "route_patterns": [
                        r"path\s*\(\s*[\"'][^\"']*[\"']",
                        r"url\s*\(\s*[\"'][^\"']*[\"']",
                        r"re_path\s*\(\s*[\"'][^\"']*[\"']",
                    ],
                    "model_patterns": [
                        r"class\s+\w+\(models\.Model\):",
                        r"class\s+\w+\(serializers\.ModelSerializer\):",
                    ]
                }
            },
            "javascript_frameworks": {
                "express": {
                    "route_patterns": [
                        r"app\.(get|post|put|delete|patch)\s*\(\s*[\"'][^\"']*[\"']",
                        r"router\.(get|post|put|delete|patch)\s*\(\s*[\"'][^\"']*[\"']",
                    ],
                    "model_patterns": [
                        r"const\s+\w+\s*=\s*mongoose\.model\s*\(",
                        r"class\s+\w+\s*extends\s+Model",
                    ]
                },
                "koa": {
                    "route_patterns": [
                        r"router\.(get|post|put|delete|patch)\s*\(\s*[\"'][^\"']*[\"']",
                    ]
                }
            }
        }
    
    async def analyze_api_changes(self, base_branch: str, head_branch: str) -> Dict:
        """Analyze API changes between two branches"""
        try:
            # Get the diff between branches
            diff_output = await self._get_api_diff(base_branch, head_branch)
            
            # Parse API changes
            api_changes = self._parse_api_changes(diff_output)
            
            # Analyze breaking changes
            breaking_changes = await self._analyze_breaking_changes(api_changes)
            
            # Generate compatibility report
            compatibility_report = self._generate_compatibility_report(api_changes, breaking_changes)
            
            return {
                "api_changes": api_changes,
                "breaking_changes": breaking_changes,
                "compatibility_report": compatibility_report,
                "summary": {
                    "total_endpoints_changed": len(api_changes.get("endpoints", [])),
                    "total_models_changed": len(api_changes.get("models", [])),
                    "breaking_changes_count": len(breaking_changes),
                    "compatibility_score": compatibility_report.get("score", 100)
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to analyze API changes: {str(e)}",
                "api_changes": {},
                "breaking_changes": [],
                "compatibility_report": {}
            }
    
    async def _get_api_diff(self, base_branch: str, head_branch: str) -> str:
        """Get git diff for API-related files"""
        try:
            # Fetch latest changes
            await run_cmd(["git", "fetch", "origin"], self.repo_path)
            
            # Get diff for API-related files
            rc, out, err = await run_cmd([
                "git", "diff", "--name-only", 
                f"origin/{base_branch}..origin/{head_branch}"
            ], self.repo_path)
            
            if rc != 0:
                raise Exception(f"Git diff failed: {err}")
            
            # Get detailed diff for API files
            api_files = []
            for line in out.splitlines():
                if self._is_api_file(line):
                    api_files.append(line)
            
            if not api_files:
                return ""
            
            # Get detailed diff for API files
            rc, detailed_diff, err = await run_cmd([
                "git", "diff", f"origin/{base_branch}..origin/{head_branch}", "--"
            ] + api_files, self.repo_path)
            
            if rc != 0:
                return ""
            
            return detailed_diff
            
        except Exception as e:
            return ""
    
    def _is_api_file(self, file_path: str) -> bool:
        """Check if a file is likely to contain API definitions"""
        api_indicators = [
            "api/", "routes/", "endpoints/", "views/", "controllers/",
            "models.py", "schemas.py", "serializers.py", "views.py",
            "routes.js", "controllers.js", "models.js", "schemas.js"
        ]
        
        return any(indicator in file_path.lower() for indicator in api_indicators)
    
    def _parse_api_changes(self, diff_output: str) -> Dict:
        """Parse API changes from git diff output"""
        api_changes = {
            "endpoints": [],
            "models": [],
            "schemas": [],
            "files_modified": []
        }
        
        if not diff_output:
            return api_changes
        
        lines = diff_output.splitlines()
        current_file = None
        
        for line in lines:
            # Track current file
            if line.startswith("+++ b/"):
                current_file = line[6:]  # Remove "+++ b/" prefix
                if self._is_api_file(current_file):
                    api_changes["files_modified"].append(current_file)
            
            # Look for endpoint changes
            elif line.startswith("+") and "route" in line.lower():
                if current_file and self._is_api_file(current_file):
                    api_changes["endpoints"].append({
                        "file": current_file,
                        "line": line,
                        "type": "added",
                        "content": line[1:].strip()
                    })
            
            elif line.startswith("-") and "route" in line.lower():
                if current_file and self._is_api_file(current_file):
                    api_changes["endpoints"].append({
                        "file": current_file,
                        "line": line,
                        "type": "removed",
                        "content": line[1:].strip()
                    })
            
            # Look for model changes
            elif line.startswith("+") and ("class" in line.lower() and "model" in line.lower()):
                if current_file and self._is_api_file(current_file):
                    api_changes["models"].append({
                        "file": current_file,
                        "line": line,
                        "type": "added",
                        "content": line[1:].strip()
                    })
            
            elif line.startswith("-") and ("class" in line.lower() and "model" in line.lower()):
                if current_file and self._is_api_file(current_file):
                    api_changes["models"].append({
                        "file": current_file,
                        "line": line,
                        "type": "removed",
                        "content": line[1:].strip()
                    })
        
        return api_changes
    
    async def _analyze_breaking_changes(self, api_changes: Dict) -> List[Dict]:
        """Analyze API changes for breaking changes"""
        breaking_changes = []
        
        # Check for removed endpoints
        for endpoint in api_changes.get("endpoints", []):
            if endpoint["type"] == "removed":
                breaking_changes.append({
                    "type": "removed_endpoint",
                    "severity": "critical",
                    "file": endpoint["file"],
                    "description": f"API endpoint removed: {endpoint['content']}",
                    "impact": "Clients using this endpoint will receive 404 errors",
                    "remediation": "Consider deprecating endpoints gradually or maintaining backward compatibility"
                })
        
        # Check for removed models
        for model in api_changes.get("models", []):
            if model["type"] == "removed":
                breaking_changes.append({
                    "type": "removed_model",
                    "severity": "critical",
                    "file": model["file"],
                    "description": f"Data model removed: {model['content']}",
                    "impact": "API responses may fail or return unexpected data",
                    "remediation": "Maintain model compatibility or provide migration paths"
                })
        
        # Check for schema changes
        breaking_changes.extend(await self._check_schema_changes(api_changes))
        
        # Check for URL pattern changes
        breaking_changes.extend(await self._check_url_changes(api_changes))
        
        return breaking_changes
    
    async def _check_schema_changes(self, api_changes: Dict) -> List[Dict]:
        """Check for breaking schema changes"""
        schema_changes = []
        
        # This would require more sophisticated analysis of the actual code
        # For now, we'll provide a basic framework
        
        for model in api_changes.get("models", []):
            if model["type"] == "added":
                # New models are generally not breaking
                continue
            
            # Check for field changes (this would require deeper analysis)
            # For now, we'll flag model modifications as potential breaking changes
            if "class" in model["content"] and "model" in model["content"]:
                schema_changes.append({
                    "type": "model_modified",
                    "severity": "medium",
                    "file": model["file"],
                    "description": f"Data model modified: {model['content']}",
                    "impact": "API responses may have different structure",
                    "remediation": "Ensure backward compatibility or version the API"
                })
        
        return schema_changes
    
    async def _check_url_changes(self, api_changes: Dict) -> List[Dict]:
        """Check for breaking URL pattern changes"""
        url_changes = []
        
        for endpoint in api_changes.get("endpoints", []):
            if "route" in endpoint["content"]:
                # Extract URL pattern
                url_match = re.search(r'["\']([^"\']*)["\']', endpoint["content"])
                if url_match:
                    url_pattern = url_match.group(1)
                    
                    if endpoint["type"] == "removed":
                        url_changes.append({
                            "type": "url_removed",
                            "severity": "critical",
                            "file": endpoint["file"],
                            "description": f"URL pattern removed: {url_pattern}",
                            "impact": "Existing clients will receive 404 errors",
                            "remediation": "Maintain URL compatibility or provide redirects"
                        })
                    
                    elif endpoint["type"] == "added":
                        # Check if this might conflict with existing patterns
                        url_changes.append({
                            "type": "url_added",
                            "severity": "low",
                            "file": endpoint["file"],
                            "description": f"New URL pattern added: {url_pattern}",
                            "impact": "New functionality available",
                            "remediation": "Ensure no conflicts with existing patterns"
                        })
        
        return url_changes
    
    def _generate_compatibility_report(self, api_changes: Dict, breaking_changes: List[Dict]) -> Dict:
        """Generate API compatibility report"""
        total_changes = len(api_changes.get("endpoints", [])) + len(api_changes.get("models", []))
        breaking_count = len(breaking_changes)
        
        # Calculate compatibility score (0-100)
        if total_changes == 0:
            compatibility_score = 100
        else:
            compatibility_score = max(0, 100 - (breaking_count * 20))
        
        # Determine compatibility level
        if compatibility_score >= 90:
            compatibility_level = "excellent"
        elif compatibility_score >= 75:
            compatibility_level = "good"
        elif compatibility_score >= 50:
            compatibility_level = "fair"
        else:
            compatibility_level = "poor"
        
        # Generate recommendations
        recommendations = []
        
        if breaking_count > 0:
            recommendations.append(f"Address {breaking_count} breaking changes to maintain API compatibility")
        
        if len(api_changes.get("endpoints", [])) > 0:
            recommendations.append("Consider API versioning for major changes")
        
        if len(api_changes.get("models", [])) > 0:
            recommendations.append("Ensure data model changes maintain backward compatibility")
        
        if total_changes > 10:
            recommendations.append("Consider a comprehensive API compatibility review")
        
        return {
            "score": compatibility_score,
            "level": compatibility_level,
            "total_changes": total_changes,
            "breaking_changes": breaking_count,
            "recommendations": recommendations,
            "compatibility_notes": self._generate_compatibility_notes(api_changes, breaking_changes)
        }
    
    def _generate_compatibility_notes(self, api_changes: Dict, breaking_changes: List[Dict]) -> List[str]:
        """Generate detailed compatibility notes"""
        notes = []
        
        # Endpoint changes
        endpoint_changes = api_changes.get("endpoints", [])
        if endpoint_changes:
            added_endpoints = [e for e in endpoint_changes if e["type"] == "added"]
            removed_endpoints = [e for e in endpoint_changes if e["type"] == "removed"]
            
            if added_endpoints:
                notes.append(f"Added {len(added_endpoints)} new API endpoints")
            
            if removed_endpoints:
                notes.append(f"Removed {len(removed_endpoints)} API endpoints (breaking change)")
        
        # Model changes
        model_changes = api_changes.get("models", [])
        if model_changes:
            added_models = [m for m in model_changes if m["type"] == "added"]
            removed_models = [m for m in model_changes if m["type"] == "removed"]
            
            if added_models:
                notes.append(f"Added {len(added_models)} new data models")
            
            if removed_models:
                notes.append(f"Removed {len(removed_models)} data models (breaking change)")
        
        # Breaking change details
        if breaking_changes:
            notes.append("Breaking changes detected - API versioning recommended")
            
            # Categorize breaking changes
            endpoint_removals = [b for b in breaking_changes if b["type"] == "removed_endpoint"]
            model_removals = [b for b in breaking_changes if b["type"] == "removed_model"]
            
            if endpoint_removals:
                notes.append(f"{len(endpoint_removals)} endpoint removals require client updates")
            
            if model_removals:
                notes.append(f"{len(model_removals)} model removals may break data parsing")
        
        return notes
    
    async def generate_api_documentation(self, branch: str = "main") -> Dict:
        """Generate current API documentation"""
        try:
            # Get current API structure
            api_files = []
            for pattern in ["**/*.py", "**/*.js", "**/*.ts"]:
                api_files.extend(self.repo_path.rglob(pattern))
            
            api_structure = {
                "endpoints": [],
                "models": [],
                "frameworks_detected": [],
                "total_files": len(api_files)
            }
            
            # Analyze each API file
            for file_path in api_files:
                if self._is_api_file(str(file_path)):
                    file_analysis = await self._analyze_api_file(file_path)
                    api_structure["endpoints"].extend(file_analysis.get("endpoints", []))
                    api_structure["models"].extend(file_analysis.get("models", []))
                    
                    # Detect frameworks
                    framework = self._detect_framework(file_path)
                    if framework and framework not in api_structure["frameworks_detected"]:
                        api_structure["frameworks_detected"].append(framework)
            
            return api_structure
            
        except Exception as e:
            return {
                "error": f"Failed to generate API documentation: {str(e)}",
                "endpoints": [],
                "models": [],
                "frameworks_detected": []
            }
    
    async def _analyze_api_file(self, file_path: Path) -> Dict:
        """Analyze a single API file"""
        analysis = {
            "endpoints": [],
            "models": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                # Look for endpoints
                for framework, patterns in self.api_patterns["python_frameworks"].items():
                    for pattern in patterns["route_patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            analysis["endpoints"].append({
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": line_num,
                                "framework": framework,
                                "pattern": line.strip()
                            })
                
                # Look for models
                for framework, patterns in self.api_patterns["python_frameworks"].items():
                    for pattern in patterns["model_patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            analysis["models"].append({
                                "file": str(file_path.relative_to(self.repo_path)),
                                "line": line_num,
                                "framework": framework,
                                "pattern": line.strip()
                            })
            
        except Exception:
            pass
        
        return analysis
    
    def _detect_framework(self, file_path: Path) -> Optional[str]:
        """Detect the web framework used in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for framework indicators
            if "fastapi" in content.lower() or "@app." in content:
                return "fastapi"
            elif "flask" in content.lower() or "@app.route" in content:
                return "flask"
            elif "django" in content.lower() or "models.Model" in content:
                return "django"
            elif "express" in content.lower() or "app.get" in content:
                return "express"
            elif "koa" in content.lower() or "router.get" in content:
                return "koa"
            
        except Exception:
            pass
        
        return None

async def analyze_api_changes_for_branches(repo_path: Path, base_branch: str, head_branch: str) -> Dict:
    """Analyze API changes between two branches"""
    analyzer = APIAnalyzer(repo_path)
    return await analyzer.analyze_api_changes(base_branch, head_branch)

async def generate_api_documentation_for_branch(repo_path: Path, branch: str = "main") -> Dict:
    """Generate API documentation for a specific branch"""
    analyzer = APIAnalyzer(repo_path)
    return await analyzer.generate_api_documentation(branch)
