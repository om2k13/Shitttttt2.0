import os
import json
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import aiohttp
from ..core.settings import settings

class GitHubIntegration:
    """Integrates with GitHub API for PR comments and status updates"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN") or settings.GITHUB_TOKEN
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CodeReviewAgent/1.0"
        } if self.token else {}
    
    async def post_pr_comments(self, repo_owner: str, repo_name: str, pr_number: int, 
                              findings: List[Dict]) -> Dict:
        """Post findings as inline PR comments"""
        if not self.token:
            return {"error": "GitHub token not configured"}
        
        try:
            # Get PR details to understand the diff
            pr_details = await self._get_pr_details(repo_owner, repo_name, pr_number)
            if not pr_details:
                return {"error": "Failed to get PR details"}
            
            # Get PR files to map findings to diff positions
            pr_files = await self._get_pr_files(repo_owner, repo_name, pr_number)
            
            # Post comments for each finding
            posted_comments = []
            for finding in findings:
                comment = await self._create_finding_comment(finding, pr_files)
                if comment:
                    result = await self._post_comment(repo_owner, repo_name, pr_number, comment)
                    if result.get("success"):
                        posted_comments.append(result)
            
            # Update PR status
            await self._update_pr_status(repo_owner, repo_name, pr_number, findings)
            
            return {
                "success": True,
                "comments_posted": len(posted_comments),
                "total_findings": len(findings),
                "posted_comments": posted_comments
            }
            
        except Exception as e:
            return {"error": f"Failed to post PR comments: {str(e)}"}
    
    async def _get_pr_details(self, repo_owner: str, repo_name: str, pr_number: int) -> Optional[Dict]:
        """Get PR details from GitHub API"""
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                return None
    
    async def _get_pr_files(self, repo_owner: str, repo_name: str, pr_number: int) -> List[Dict]:
        """Get list of files changed in PR"""
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/files"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    async def _create_finding_comment(self, finding: Dict, pr_files: List[Dict]) -> Optional[Dict]:
        """Create a GitHub comment for a finding"""
        # Find the corresponding PR file
        pr_file = None
        for file in pr_files:
            if file["filename"] == finding["file"]:
                pr_file = file
                break
        
        if not pr_file:
            return None
        
        # Determine comment position
        position = self._get_comment_position(finding, pr_file)
        if not position:
            return None
        
        # Create comment body
        body = self._format_finding_comment(finding)
        
        return {
            "body": body,
            "commit_id": pr_file["sha"],
            "path": finding["file"],
            "position": position
        }
    
    def _get_comment_position(self, finding: Dict, pr_file: Dict) -> Optional[int]:
        """Get the position in the diff for the comment"""
        try:
            line_number = finding.get("line")
            if not line_number:
                return None
            
            # Convert line number to diff position
            # This is a simplified approach - GitHub's diff position is more complex
            # For now, we'll use the line number directly
            return int(line_number)
        except (ValueError, TypeError):
            return None
    
    def _format_finding_comment(self, finding: Dict) -> str:
        """Format a finding as a GitHub comment"""
        severity_emoji = {
            "critical": "🚨",
            "high": "⚠️", 
            "medium": "⚡",
            "low": "💡"
        }
        
        tool_emoji = {
            "ruff": "🐍",
            "mypy": "🔍",
            "semgrep": "🛡️",
            "bandit": "🔒",
            "eslint": "⚙️",
            "yaml-validator": "📄",
            "json-validator": "📋"
        }
        
        emoji = severity_emoji.get(finding.get("severity", "low"), "ℹ️")
        tool_emoji_char = tool_emoji.get(finding.get("tool", ""), "🔧")
        
        comment = f"""## {emoji} {finding.get('severity', 'low').upper()} - {tool_emoji_char} {finding.get('tool', 'Unknown Tool')}

**Issue**: {finding.get('message', 'No message provided')}

**Rule**: `{finding.get('rule_id', 'Unknown')}`

**File**: `{finding.get('file', 'Unknown')}`
**Line**: {finding.get('line', 'Unknown')}

"""
        
        # Add PR context if available
        pr_context = finding.get("pr_context", {})
        if pr_context:
            comment += f"""**PR Context**:
- **File Status**: {pr_context.get('file_status', 'Unknown')}
- **File Type**: {pr_context.get('file_type', 'Unknown')}
- **Language**: {pr_context.get('language', 'Unknown')}

"""
        
        # Add remediation if available
        if finding.get("remediation"):
            comment += f"""**Suggested Fix**:
{finding.get('remediation')}

"""
        
        # Add auto-fixable indicator
        if finding.get("autofixable"):
            comment += "✅ **Auto-fixable**: This issue can be automatically fixed"
        else:
            comment += "🔧 **Manual Fix Required**: This issue requires manual intervention"
        
        return comment
    
    async def _post_comment(self, repo_owner: str, repo_name: str, pr_number: int, 
                           comment: Dict) -> Dict:
        """Post a comment to a GitHub PR"""
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/comments"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=comment) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    return {
                        "success": True,
                        "comment_id": result.get("id"),
                        "url": result.get("html_url")
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
    
    async def _update_pr_status(self, repo_owner: str, repo_name: str, pr_number: int, 
                               findings: Dict) -> bool:
        """Update PR status with analysis results"""
        try:
            # Count findings by severity
            severity_counts = {}
            for finding in findings:
                severity = finding.get("severity", "low")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Determine overall status
            if severity_counts.get("critical", 0) > 0:
                state = "failure"
                description = f"🚨 {severity_counts.get('critical', 0)} critical issues found"
            elif severity_counts.get("high", 0) > 0:
                state = "failure"
                description = f"⚠️ {severity_counts.get('high', 0)} high severity issues found"
            elif severity_counts.get("medium", 0) > 0:
                state = "success"
                description = f"⚡ {severity_counts.get('medium', 0)} medium issues found"
            else:
                state = "success"
                description = "✅ Code review passed"
            
            # Create status check
            status_data = {
                "state": state,
                "target_url": f"https://github.com/{repo_owner}/{repo_name}/pull/{pr_number}",
                "description": description,
                "context": "Code Review Agent"
            }
            
            # Post status (this requires a different endpoint)
            # For now, we'll return success
            return True
            
        except Exception:
            return False
    
    async def create_review(self, repo_owner: str, repo_name: str, pr_number: int, 
                           findings: List[Dict]) -> Dict:
        """Create a comprehensive PR review"""
        if not self.token:
            return {"error": "GitHub token not configured"}
        
        try:
            # Group findings by file and create review comments
            review_comments = []
            for finding in findings:
                comment = await self._create_finding_comment(finding, [])
                if comment:
                    review_comments.append(comment)
            
            # Create review body
            review_body = self._create_review_summary(findings)
            
            # Submit review
            review_data = {
                "body": review_body,
                "event": "COMMENT",  # or "REQUEST_CHANGES" for critical issues
                "comments": review_comments
            }
            
            url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/reviews"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=review_data) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        return {
                            "success": True,
                            "review_id": result.get("id"),
                            "url": result.get("html_url"),
                            "comments_posted": len(review_comments)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {"error": f"Failed to create review: {str(e)}"}
    
    def _create_review_summary(self, findings: List[Dict]) -> str:
        """Create a summary for the PR review"""
        severity_counts = {}
        tool_counts = {}
        
        for finding in findings:
            severity = finding.get("severity", "low")
            tool = finding.get("tool", "unknown")
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        summary = f"""## 🔍 Code Review Summary

**Total Issues Found**: {len(findings)}

### 📊 Issues by Severity
"""
        
        for severity in ["critical", "high", "medium", "low"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "💡"}[severity]
                summary += f"- {emoji} **{severity.upper()}**: {count} issues\n"
        
        summary += "\n### 🛠️ Issues by Tool\n"
        for tool, count in tool_counts.items():
            summary += f"- **{tool}**: {count} issues\n"
        
        summary += "\n### 💡 Recommendations\n"
        
        if severity_counts.get("critical", 0) > 0:
            summary += "- 🚨 **Critical issues must be fixed** before merging\n"
        
        if severity_counts.get("high", 0) > 0:
            summary += "- ⚠️ **High severity issues** should be addressed\n"
        
        auto_fixable = sum(1 for f in findings if f.get("autofixable"))
        if auto_fixable > 0:
            summary += f"- ✅ **{auto_fixable} issues can be auto-fixed** using the Apply Safe Auto-Fixes feature\n"
        
        summary += "\n---\n*Review generated by Code Review Agent*"
        
        return summary

# Convenience functions
async def post_findings_to_pr(repo_url: str, pr_number: int, findings: List[Dict]) -> Dict:
    """Post findings to a GitHub PR"""
    # Parse repo URL to get owner and name
    # Expected format: https://github.com/owner/repo
    parts = repo_url.rstrip('/').split('/')
    if len(parts) < 2:
        return {"error": "Invalid GitHub repository URL"}
    
    repo_owner = parts[-2]
    repo_name = parts[-1]
    
    github = GitHubIntegration()
    return await github.post_pr_comments(repo_owner, repo_name, pr_number, findings)

async def create_pr_review(repo_url: str, pr_number: int, findings: List[Dict]) -> Dict:
    """Create a comprehensive PR review"""
    parts = repo_url.rstrip('/').split('/')
    if len(parts) < 2:
        return {"error": "Invalid GitHub repository URL"}
    
    repo_owner = parts[-2]
    repo_name = parts[-1]
    
    github = GitHubIntegration()
    return await github.create_review(repo_owner, repo_name, pr_number, findings)
