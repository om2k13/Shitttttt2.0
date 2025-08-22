import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import asyncio
from ..core.settings import settings

class CICDIntegration:
    """Integration with various CI/CD platforms for automated code review"""
    
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.gitlab_token = os.getenv("GITLAB_TOKEN")
        self.jenkins_url = os.getenv("JENKINS_URL")
        self.jenkins_user = os.getenv("JENKINS_USER")
        self.jenkins_token = os.getenv("JENKINS_TOKEN")
    
    async def create_github_action_workflow(self, repo_path: Path, config: Dict) -> str:
        """Create GitHub Actions workflow for automated code review"""
        workflow_file = repo_path / ".github" / "workflows" / "code-review.yml"
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        
        workflow_content = self._generate_github_action_workflow_content(config)
        workflow_file.write_text(workflow_content)
        
        return str(workflow_file)
    
    async def execute_github_action_workflow(self, repo_path: Path, config: Dict) -> Dict[str, Any]:
        """Execute GitHub Actions workflow locally for testing"""
        try:
            print("🚀 Executing GitHub Actions workflow locally...")
            
            # Create workflow file
            workflow_file = await self.create_github_action_workflow(repo_path, config)
            
            # Execute the commands that would run in GitHub Actions
            results = {
                "workflow_created": True,
                "workflow_file": workflow_file,
                "execution_results": {}
            }
            
            # Run code review agent
            print("🔍 Running code review agent...")
            from ..review.enhanced_pipeline import run_standalone_code_review
            code_review_results = await run_standalone_code_review(str(repo_path))
            results["execution_results"]["code_review"] = code_review_results
            
            # Run security analysis
            print("🔒 Running security analysis...")
            from ..review.enhanced_pipeline import EnhancedPipeline
            pipeline = EnhancedPipeline()
            pipeline.repo_path = repo_path
            security_results = await pipeline._run_security_tools()
            results["execution_results"]["security_analysis"] = {
                "findings": security_results,
                "total_findings": len(security_results)
            }
            
            # Generate test plan
            print("🧪 Generating test plan...")
            from ..review.test_generator import TestGenerator
            test_generator = TestGenerator(repo_path)
            test_plan = await test_generator.generate_test_plan(
                changed_files=[{"path": str(repo_path), "type": "python"}],
                findings=security_results[:5]
            )
            results["execution_results"]["test_plan"] = test_plan
            
            print("✅ GitHub Actions workflow executed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error executing GitHub Actions workflow: {e}")
            return {
                "workflow_created": False,
                "error": str(e),
                "execution_results": {}
            }
    
    def _generate_github_action_command(self, config: Dict) -> str:
        """Generate the command to run the code review agent in GitHub Actions"""
        cmd_parts = ["python -m code_review_agent"]
        
        if config.get("analysis_type") == "pr":
            cmd_parts.append("pr-analysis")
        elif config.get("analysis_type") == "comprehensive":
            cmd_parts.append("comprehensive-analysis")
        else:
            cmd_parts.append("security-analysis")
        
        # Add configuration options
        if config.get("severity_threshold"):
            cmd_parts.extend(["--severity", config["severity_threshold"]])
        
        if config.get("enable_autofix", False):
            cmd_parts.append("--enable-autofix")
        
        if config.get("max_issues"):
            cmd_parts.extend(["--max-issues", str(config["max_issues"])])
        
        return " ".join(cmd_parts)
    
    def _generate_post_review_command(self, config: Dict) -> str:
        """Generate command to post review results back to GitHub"""
        return """
        if [ -f "review-results.json" ]; then
            python -m code_review_agent post-results \
                --repo $REPO_URL \
                --pr $PR_NUMBER \
                --results review-results.json
        fi
        """
    
    async def create_gitlab_ci_config(self, repo_path: Path, config: Dict) -> str:
        """Create a GitLab CI configuration for automated code review"""
        gitlab_ci_content = {
            "stages": ["code-review"],
            "variables": {
                "PYTHON_VERSION": "3.11"
            },
            "code-review": {
                "stage": "code-review",
                "image": f"python:{config.get('python_version', '3.11')}",
                "before_script": [
                    "pip install -r requirements.txt"
                ],
                "script": [
                    self._generate_gitlab_ci_command(config)
                ],
                "after_script": [
                    self._generate_gitlab_post_review_command(config)
                ],
                "rules": [
                    {
                        "if": "$CI_PIPELINE_SOURCE == 'merge_request_event'"
                    },
                    {
                        "if": "$CI_COMMIT_BRANCH == 'main'"
                    },
                    {
                        "if": "$CI_COMMIT_BRANCH == 'develop'"
                    }
                ],
                "artifacts": {
                    "reports": {
                        "junit": "review-results.xml"
                    },
                    "paths": ["review-results.json"],
                    "expire_in": "1 week"
                }
            }
        }
        
        # Add conditional stages based on config
        if config.get("enable_security_analysis", True):
            gitlab_ci_content["stages"].append("security-scan")
            gitlab_ci_content["security-scan"] = {
                "stage": "security-scan",
                "image": f"python:{config.get('python_version', '3.11')}",
                "script": [
                    "python -m code_review_agent security-analysis --level advanced"
                ],
                "rules": [
                    {
                        "if": "$CI_PIPELINE_SOURCE == 'merge_request_event'"
                    }
                ]
            }
        
        # Write .gitlab-ci.yml file
        gitlab_ci_file = repo_path / ".gitlab-ci.yml"
        with open(gitlab_ci_file, 'w') as f:
            yaml.dump(gitlab_ci_content, f, default_flow_style=False)
        
        return str(gitlab_ci_file)
    
    def _generate_gitlab_ci_command(self, config: Dict) -> str:
        """Generate the command to run the code review agent in GitLab CI"""
        cmd_parts = ["python -m code_review_agent"]
        
        if config.get("analysis_type") == "pr":
            cmd_parts.append("pr-analysis")
        elif config.get("analysis_type") == "comprehensive":
            cmd_parts.append("comprehensive-analysis")
        else:
            cmd_parts.append("security-analysis")
        
        # Add GitLab-specific options
        cmd_parts.extend([
            "--ci-platform", "gitlab",
            "--project-id", "$CI_PROJECT_ID",
            "--mr-id", "$CI_MERGE_REQUEST_ID"
        ])
        
        if config.get("severity_threshold"):
            cmd_parts.extend(["--severity", config["severity_threshold"]])
        
        return " ".join(cmd_parts)
    
    def _generate_gitlab_post_review_command(self, config: Dict) -> str:
        """Generate command to post review results back to GitLab"""
        return """
        if [ -f "review-results.json" ]; then
            python -m code_review_agent post-results \
                --ci-platform gitlab \
                --project-id $CI_PROJECT_ID \
                --mr-id $CI_MERGE_REQUEST_ID \
                --results review-results.json
        fi
        """
    
    async def create_jenkins_pipeline(self, repo_path: Path, config: Dict) -> str:
        """Create a Jenkins pipeline for automated code review"""
        jenkinsfile_content = f"""pipeline {{
    agent any
    
    environment {{
        PYTHON_VERSION = '{config.get("python_version", "3.11")}'
        REPO_URL = env.GIT_URL
        BRANCH_NAME = env.GIT_BRANCH
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Setup Python') {{
            steps {{
                sh 'python --version'
                sh 'pip install -r requirements.txt'
            }}
        }}
        
        stage('Code Review') {{
            steps {{
                script {{
                    def analysisType = '{config.get("analysis_type", "security")}'
                    def command = "python -m code_review_agent"
                    
                    if (analysisType == "pr") {{
                        command += " pr-analysis"
                    }} else if (analysisType == "comprehensive") {{
                        command += " comprehensive-analysis"
                    }} else {{
                        command += " security-analysis"
                    }}
                    
                    if ('{config.get("severity_threshold", "")}') {{
                        command += " --severity {config.get("severity_threshold", "medium")}"
                    }}
                    
                    sh command
                }}
            }}
            post {{
                always {{
                    archiveArtifacts artifacts: 'review-results.json', fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'review-reports',
                        reportFiles: 'index.html',
                        reportName: 'Code Review Report'
                    ])
                }}
            }}
        }}
"""
        
        # Add conditional stages based on config
        if config.get("enable_security_analysis", True):
            jenkinsfile_content += """
        stage('Security Analysis') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    changeRequest()
                }
            }
            steps {
                sh 'python -m code_review_agent security-analysis --level advanced'
            }
        }
"""
        
        if config.get("enable_performance_analysis", True):
            jenkinsfile_content += """
        stage('Performance Analysis') {
            when {
                anyOf {
                    branch 'main'
                    changeRequest()
                }
            }
            steps {
                sh 'python -m code_review_agent performance-analysis'
            }
        }
"""
        
        jenkinsfile_content += """
    }}
    
    post {{
        always {{
            script {{
                if (fileExists('review-results.json')) {{
                    sh '''
                        python -m code_review_agent post-results \\
                            --ci-platform jenkins \\
                            --results review-results.json \\
                            --build-url ${env.BUILD_URL}
                    '''
                }}
            }}
        }}
    }}
}}"""
        
        # Write Jenkinsfile
        jenkinsfile = repo_path / "Jenkinsfile"
        with open(jenkinsfile, 'w') as f:
            f.write(jenkinsfile_content)
        
        return str(jenkinsfile)
    
    async def create_circleci_config(self, repo_path: Path, config: Dict) -> str:
        """Create a CircleCI configuration for automated code review"""
        circleci_content = {
            "version": 2.1,
            "jobs": {
                "code-review": {
                    "docker": [
                        {
                            "image": f"python:{config.get('python_version', '3.11')}"
                        }
                    ],
                    "steps": [
                        "checkout",
                        {
                            "run": {
                                "name": "Install dependencies",
                                "command": "pip install -r requirements.txt"
                            }
                        },
                        {
                            "run": {
                                "name": "Run Code Review",
                                "command": self._generate_circleci_command(config)
                            }
                        },
                        {
                            "run": {
                                "name": "Post Results",
                                "command": self._generate_circleci_post_command(config),
                                "when": "always"
                            }
                        },
                        {
                            "store_artifacts": {
                                "path": "review-results.json",
                                "destination": "code-review-results"
                            }
                        }
                    ]
                }
            },
            "workflows": {
                "version": 2,
                "code-review": {
                    "jobs": [
                        {
                            "code-review": {
                                "filters": {
                                    "branches": {
                                        "only": ["main", "develop"]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Add conditional jobs based on config
        if config.get("enable_security_analysis", True):
            circleci_content["jobs"]["security-analysis"] = {
                "docker": [
                    {
                        "image": f"python:{config.get('python_version', '3.11')}"
                    }
                ],
                "steps": [
                    "checkout",
                    {
                        "run": {
                            "name": "Install dependencies",
                            "command": "pip install -r requirements.txt"
                        }
                    },
                    {
                        "run": {
                            "name": "Security Analysis",
                            "command": "python -m code_review_agent security-analysis --level advanced"
                        }
                    }
                ]
            }
            
            # Add to workflow
            circleci_content["workflows"]["code-review"]["jobs"].append({
                "security-analysis": {
                    "filters": {
                        "branches": {
                            "only": ["main", "develop"]
                        }
                    }
                }
            })
        
        # Write .circleci/config.yml
        circleci_dir = repo_path / ".circleci"
        circleci_dir.mkdir(parents=True, exist_ok=True)
        
        circleci_file = circleci_dir / "config.yml"
        with open(circleci_file, 'w') as f:
            yaml.dump(circleci_content, f, default_flow_style=False)
        
        return str(circleci_file)
    
    def _generate_circleci_command(self, config: Dict) -> str:
        """Generate the command to run the code review agent in CircleCI"""
        cmd_parts = ["python -m code_review_agent"]
        
        if config.get("analysis_type") == "pr":
            cmd_parts.append("pr-analysis")
        elif config.get("analysis_type") == "comprehensive":
            cmd_parts.append("comprehensive-analysis")
        else:
            cmd_parts.append("security-analysis")
        
        # Add CircleCI-specific options
        cmd_parts.extend([
            "--ci-platform", "circleci",
            "--build-num", "$CIRCLE_BUILD_NUM"
        ])
        
        if config.get("severity_threshold"):
            cmd_parts.extend(["--severity", config["severity_threshold"]])
        
        return " ".join(cmd_parts)
    
    def _generate_circleci_post_command(self, config: Dict) -> str:
        """Generate command to post review results in CircleCI"""
        return """
        if [ -f "review-results.json" ]; then
            python -m code_review_agent post-results \\
                --ci-platform circleci \\
                --results review-results.json \\
                --build-num $CIRCLE_BUILD_NUM
        fi
        """
    
    async def create_azure_pipelines_config(self, repo_path: Path, config: Dict) -> str:
        """Create an Azure Pipelines configuration for automated code review"""
        azure_pipelines_content = f"""trigger:
- main
- develop

pr:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  python.version: '{config.get("python_version", "3.11")}'

stages:
- stage: CodeReview
  displayName: 'Code Review Stage'
  jobs:
  - job: CodeReview
    displayName: 'Code Review Job'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(python.version)'
        addToPath: true
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
      displayName: 'Install dependencies'
    
    - script: |
        python -m code_review_agent {config.get("analysis_type", "security")}-analysis
      displayName: 'Run Code Review'
    
    - script: |
        if [ -f "review-results.json" ]; then
          python -m code_review_agent post-results \\
            --ci-platform azure \\
            --results review-results.json \\
            --build-id $(Build.BuildId)
        fi
      displayName: 'Post Results'
      condition: always()
    
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: 'review-results.json'
        artifactName: 'CodeReviewResults'
      condition: succeededOrFailed()"""
        
        # Add conditional stages based on config
        if config.get("enable_security_analysis", True):
            azure_pipelines_content += """

- stage: SecurityAnalysis
  displayName: 'Security Analysis Stage'
  dependsOn: CodeReview
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - job: SecurityAnalysis
    displayName: 'Security Analysis Job'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(python.version)'
        addToPath: true
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
      displayName: 'Install dependencies'
    
    - script: |
        python -m code_review_agent security-analysis --level advanced
      displayName: 'Run Security Analysis'"""
        
        # Write azure-pipelines.yml
        azure_pipelines_file = repo_path / "azure-pipelines.yml"
        with open(azure_pipelines_file, 'w') as f:
            f.write(azure_pipelines_content)
        
        return str(azure_pipelines_file)
    
    async def get_ci_platform_info(self, repo_path: Path) -> Dict[str, Any]:
        """Detect which CI/CD platform is configured in the repository"""
        ci_info = {
            "platform": "unknown",
            "config_files": [],
            "configured": False
        }
        
        # Check for GitHub Actions
        github_workflows = repo_path / ".github" / "workflows"
        if github_workflows.exists():
            ci_info["platform"] = "github_actions"
            ci_info["configured"] = True
            ci_info["config_files"] = [str(f) for f in github_workflows.glob("*.yml")]
        
        # Check for GitLab CI
        gitlab_ci = repo_path / ".gitlab-ci.yml"
        if gitlab_ci.exists():
            ci_info["platform"] = "gitlab_ci"
            ci_info["configured"] = True
            ci_info["config_files"].append(str(gitlab_ci))
        
        # Check for Jenkins
        jenkinsfile = repo_path / "Jenkinsfile"
        if jenkinsfile.exists():
            ci_info["platform"] = "jenkins"
            ci_info["configured"] = True
            ci_info["config_files"].append(str(jenkinsfile))
        
        # Check for CircleCI
        circleci_config = repo_path / ".circleci" / "config.yml"
        if circleci_config.exists():
            ci_info["platform"] = "circleci"
            ci_info["configured"] = True
            ci_info["config_files"].append(str(circleci_config))
        
        # Check for Azure Pipelines
        azure_pipelines = repo_path / "azure-pipelines.yml"
        if azure_pipelines.exists():
            ci_info["platform"] = "azure_pipelines"
            ci_info["configured"] = True
            ci_info["config_files"].append(str(azure_pipelines))
        
        return ci_info
    
    async def create_ci_config(self, repo_path: Path, platform: str, config: Dict) -> str:
        """Create CI/CD configuration for the specified platform"""
        if platform == "github_actions":
            return await self.create_github_action_workflow(repo_path, config)
        elif platform == "gitlab_ci":
            return await self.create_gitlab_ci_config(repo_path, config)
        elif platform == "jenkins":
            return await self.create_jenkins_pipeline(repo_path, config)
        elif platform == "circleci":
            return await self.create_circleci_config(repo_path, config)
        elif platform == "azure_pipelines":
            return await self.create_azure_pipelines_config(repo_path, config)
        else:
            raise ValueError(f"Unsupported CI/CD platform: {platform}")

    def _generate_github_action_workflow_content(self, config: Dict) -> str:
        """Generate the GitHub Actions workflow YAML content"""
        workflow_content = f"""name: Automated Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run Code Review Agent
        run: {self._generate_github_action_command(config)}
        env:
          GITHUB_TOKEN: ${{{{ secrets.GITHUB_TOKEN }}}}
          REPO_URL: ${{{{ github.repository }}}}
          PR_NUMBER: ${{{{ github.event.pull_request.number }}}}
          BRANCH_NAME: ${{{{ github.head_ref }}}}
      
      - name: Security Analysis
        run: |
          python -m app.cli.code_review_cli security-analysis ${{{{ github.repository }}}}
        if: {str(config.get("enable_security_analysis", True)).lower()}
      
      - name: Generate Test Plan
        run: |
          python -m app.cli.code_review_cli generate-test-plan ${{{{ github.repository }}}}
        if: {str(config.get("enable_test_generation", True)).lower()}
      
      - name: Post Review Results
        run: {self._generate_post_review_command(config)}
        if: always()
"""
        return workflow_content

# Global CI/CD integration instance
cicd_integration = CICDIntegration()

# Convenience functions
async def create_ci_config_for_platform(repo_path: Path, platform: str, config: Dict) -> str:
    """Create CI/CD configuration for the specified platform"""
    return await cicd_integration.create_ci_config(repo_path, platform, config)

async def detect_ci_platform(repo_path: Path) -> Dict[str, Any]:
    """Detect which CI/CD platform is configured in the repository"""
    return await cicd_integration.get_ci_platform_info(repo_path)
