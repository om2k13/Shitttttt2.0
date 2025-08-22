import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ..core.vcs import run_cmd

class GitAutomation:
    """Automates Git operations for committing and pushing fixes"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.git_path = repo_path / ".git"
    
    async def is_git_repository(self) -> bool:
        """Check if the directory is a Git repository"""
        return self.git_path.exists() and self.git_path.is_dir()
    
    async def get_current_branch(self) -> str:
        """Get the current Git branch"""
        try:
            rc, output, error = run_cmd(["git", "branch", "--show-current"], self.repo_path)
            if rc == 0 and output:
                return output.strip()
            return "main"  # Default fallback
        except Exception:
            return "main"
    
    async def get_remote_origin(self) -> Optional[str]:
        """Get the remote origin URL"""
        try:
            rc, output, error = run_cmd(["git", "remote", "get-url", "origin"], self.repo_path)
            if rc == 0 and output:
                return output.strip()
            return None
        except Exception:
            return None
    
    async def get_status(self) -> Dict:
        """Get the current Git status"""
        try:
            rc, output, error = run_cmd(["git", "status", "--porcelain"], self.repo_path)
            
            if rc != 0:
                return {"error": error or "Failed to get status"}
            
            # Parse status output
            status_lines = output.strip().split('\n') if output else []
            
            modified_files = []
            untracked_files = []
            staged_files = []
            
            for line in status_lines:
                if line:
                    status_code = line[:2]
                    file_path = line[3:]
                    
                    if status_code == 'M ' or status_code == 'MM':
                        modified_files.append(file_path)
                    elif status_code == '??':
                        untracked_files.append(file_path)
                    elif status_code == 'A ' or status_code == 'M ':
                        staged_files.append(file_path)
            
            return {
                "modified_files": modified_files,
                "untracked_files": untracked_files,
                "staged_files": staged_files,
                "has_changes": len(modified_files) > 0 or len(untracked_files) > 0,
                "has_staged": len(staged_files) > 0
            }
            
        except Exception as e:
            return {"error": f"Failed to get status: {str(e)}"}
    
    async def stage_all_changes(self) -> Dict:
        """Stage all modified and untracked files"""
        try:
            # Stage modified files
            rc, output, error = run_cmd(["git", "add", "."], self.repo_path)
            
            if rc != 0:
                return {"error": f"Failed to stage changes: {error}"}
            
            # Get status to confirm
            status = await self.get_status()
            
            return {
                "success": True,
                "message": "All changes staged successfully",
                "staged_files": status.get("staged_files", []),
                "total_staged": len(status.get("staged_files", []))
            }
            
        except Exception as e:
            return {"error": f"Failed to stage changes: {str(e)}"}
    
    async def stage_specific_files(self, file_paths: List[str]) -> Dict:
        """Stage specific files"""
        try:
            if not file_paths:
                return {"error": "No files specified"}
            
            # Stage specific files
            rc, output, error = run_cmd(["git", "add"] + file_paths, self.repo_path)
            
            if rc != 0:
                return {"error": f"Failed to stage files: {error}"}
            
            return {
                "success": True,
                "message": f"Staged {len(file_paths)} files successfully",
                "staged_files": file_paths
            }
            
        except Exception as e:
            return {"error": f"Failed to stage files: {str(e)}"}
    
    async def create_commit(self, message: str, author: str = "Code Review Agent") -> Dict:
        """Create a commit with the given message"""
        try:
            # Set author if not configured
            if not await self._is_author_configured():
                await self._configure_author(author)
            
            # Create commit
            rc, output, error = run_cmd(["git", "commit", "-m", message], self.repo_path)
            
            if rc != 0:
                return {"error": f"Failed to create commit: {error}"}
            
            # Get commit hash
            rc, commit_hash, error = run_cmd(["git", "rev-parse", "HEAD"], self.repo_path)
            
            return {
                "success": True,
                "message": "Commit created successfully",
                "commit_hash": commit_hash.strip() if commit_hash else None,
                "commit_message": message
            }
            
        except Exception as e:
            return {"error": f"Failed to create commit: {str(e)}"}
    
    async def _is_author_configured(self) -> bool:
        """Check if Git author is configured"""
        try:
            rc, output, error = run_cmd(["git", "config", "user.name"], self.repo_path)
            return rc == 0 and output.strip()
        except Exception:
            return False
    
    async def _configure_author(self, author: str):
        """Configure Git author"""
        try:
            run_cmd(["git", "config", "user.name", author], self.repo_path)
            run_cmd(["git", "config", "user.email", f"{author.lower().replace(' ', '.')}@codereview.agent"], self.repo_path)
        except Exception:
            pass  # Ignore configuration errors
    
    async def push_changes(self, branch: Optional[str] = None, force: bool = False) -> Dict:
        """Push changes to remote repository"""
        try:
            if not branch:
                branch = await self.get_current_branch()
            
            remote_origin = await self.get_remote_origin()
            if not remote_origin:
                return {"error": "No remote origin configured"}
            
            # Push command
            push_cmd = ["git", "push", "origin", branch]
            if force:
                push_cmd.append("--force")
            
            rc, output, error = run_cmd(push_cmd, self.repo_path)
            
            if rc != 0:
                return {"error": f"Failed to push changes: {error}"}
            
            return {
                "success": True,
                "message": f"Changes pushed successfully to {branch}",
                "branch": branch,
                "remote": remote_origin
            }
            
        except Exception as e:
            return {"error": f"Failed to push changes: {str(e)}"}
    
    async def create_branch(self, branch_name: str, base_branch: str = "main") -> Dict:
        """Create a new branch for fixes"""
        try:
            # Checkout base branch first
            rc, output, error = run_cmd(["git", "checkout", base_branch], self.repo_path)
            if rc != 0:
                return {"error": f"Failed to checkout base branch: {error}"}
            
            # Pull latest changes
            rc, output, error = run_cmd(["git", "pull", "origin", base_branch], self.repo_path)
            if rc != 0:
                return {"error": f"Failed to pull latest changes: {error}"}
            
            # Create and checkout new branch
            rc, output, error = run_cmd(["git", "checkout", "-b", branch_name], self.repo_path)
            if rc != 0:
                return {"error": f"Failed to create branch: {error}"}
            
            return {
                "success": True,
                "message": f"Branch {branch_name} created successfully",
                "branch_name": branch_name,
                "base_branch": base_branch
            }
            
        except Exception as e:
            return {"error": f"Failed to create branch: {str(e)}"}
    
    async def commit_fixes(self, fixes_applied: List[Dict], commit_message: Optional[str] = None) -> Dict:
        """Commit all applied fixes with a descriptive message"""
        try:
            # Get current status
            status = await self.get_status()
            if not status.get("has_changes", False):
                return {"message": "No changes to commit"}
            
            # Generate commit message if not provided
            if not commit_message:
                commit_message = self._generate_commit_message(fixes_applied)
            
            # Stage all changes
            stage_result = await self.stage_all_changes()
            if "error" in stage_result:
                return stage_result
            
            # Create commit
            commit_result = await self.create_commit(commit_message)
            if "error" in commit_result:
                return commit_result
            
            return {
                "success": True,
                "message": "Fixes committed successfully",
                "commit_hash": commit_result.get("commit_hash"),
                "commit_message": commit_message,
                "files_committed": status.get("modified_files", []) + status.get("untracked_files", []),
                "total_fixes": len(fixes_applied)
            }
            
        except Exception as e:
            return {"error": f"Failed to commit fixes: {str(e)}"}
    
    def _generate_commit_message(self, fixes_applied: List[Dict]) -> str:
        """Generate a descriptive commit message for the fixes"""
        if not fixes_applied:
            return "Apply code review fixes"
        
        # Count fixes by type
        security_fixes = len([f for f in fixes_applied if f.get('fix_type') == 'security'])
        quality_fixes = len([f for f in fixes_applied if f.get('fix_type') == 'code_quality'])
        type_fixes = len([f for f in fixes_applied if f.get('fix_type') == 'type_checking'])
        
        # Generate message
        parts = []
        if security_fixes > 0:
            parts.append(f"🔒 Fix {security_fixes} security issue(s)")
        if quality_fixes > 0:
            parts.append(f"✨ Fix {quality_fixes} code quality issue(s)")
        if type_fixes > 0:
            parts.append(f"🔍 Fix {type_fixes} type checking issue(s)")
        
        if not parts:
            parts.append("Apply code review fixes")
        
        message = " | ".join(parts)
        message += f"\n\nApplied {len(fixes_applied)} automatic fixes from code review analysis.\n"
        message += f"Generated by Code Review Agent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return message
    
    async def create_pull_request(self, branch_name: str, title: str, description: str) -> Dict:
        """Create a pull request using GitHub CLI or API"""
        try:
            # Try using GitHub CLI first
            rc, output, error = run_cmd(["gh", "pr", "create", "--title", title, "--body", description, "--head", branch_name], self.repo_path)
            
            if rc == 0 and output:
                # Extract PR URL from output
                pr_url_match = re.search(r'https://github\.com/[^/]+/[^/]+/pull/\d+', output)
                pr_url = pr_url_match.group(0) if pr_url_match else None
                
                return {
                    "success": True,
                    "message": "Pull request created successfully",
                    "pr_url": pr_url,
                    "branch": branch_name,
                    "method": "github_cli"
                }
            
            # Fallback: return instructions for manual PR creation
            return {
                "success": False,
                "message": "GitHub CLI not available. Please create PR manually.",
                "instructions": [
                    f"1. Push your branch: git push origin {branch_name}",
                    f"2. Go to the repository on GitHub",
                    f"3. Click 'Compare & pull request'",
                    f"4. Use title: {title}",
                    f"5. Use description: {description}"
                ],
                "branch": branch_name,
                "title": title,
                "description": description
            }
            
        except Exception as e:
            return {"error": f"Failed to create pull request: {str(e)}"}
    
    async def automate_fix_workflow(self, fixes_applied: List[Dict], create_pr: bool = True) -> Dict:
        """Complete workflow: commit fixes, push, and optionally create PR"""
        try:
            workflow_steps = []
            
            # Step 1: Check if it's a Git repository
            if not await self.is_git_repository():
                return {"error": "Not a Git repository"}
            
            # Step 2: Get current branch
            current_branch = await self.get_current_branch()
            workflow_steps.append(f"Current branch: {current_branch}")
            
            # Step 3: Create fix branch
            fix_branch_name = f"fix/code-review-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            branch_result = await self.create_branch(fix_branch_name, current_branch)
            if "error" in branch_result:
                return branch_result
            
            workflow_steps.append(f"Created branch: {fix_branch_name}")
            
            # Step 4: Commit fixes
            commit_result = await self.commit_fixes(fixes_applied)
            if "error" in commit_result:
                return commit_result
            
            workflow_steps.append("Committed fixes")
            
            # Step 5: Push changes
            push_result = await self.push_changes(fix_branch_name)
            if "error" in push_result:
                return push_result
            
            workflow_steps.append("Pushed changes")
            
            # Step 6: Create pull request (optional)
            pr_result = None
            if create_pr:
                title = f"🔧 Apply Code Review Fixes ({len(fixes_applied)} issues)"
                description = self._generate_pr_description(fixes_applied)
                
                pr_result = await self.create_pull_request(fix_branch_name, title, description)
                if pr_result.get("success"):
                    workflow_steps.append("Created pull request")
                else:
                    workflow_steps.append("PR creation instructions provided")
            
            return {
                "success": True,
                "message": "Fix workflow completed successfully",
                "workflow_steps": workflow_steps,
                "branch_name": fix_branch_name,
                "commit_hash": commit_result.get("commit_hash"),
                "pull_request": pr_result,
                "total_fixes": len(fixes_applied)
            }
            
        except Exception as e:
            return {"error": f"Workflow failed: {str(e)}"}
    
    def _generate_pr_description(self, fixes_applied: List[Dict]) -> str:
        """Generate a detailed PR description"""
        description = f"## 🔧 Code Review Fixes\n\n"
        description += f"This PR automatically fixes **{len(fixes_applied)}** issues found during code review.\n\n"
        
        # Group fixes by type
        fixes_by_type = {}
        for fix in fixes_applied:
            fix_type = fix.get('fix_type', 'unknown')
            if fix_type not in fixes_by_type:
                fixes_by_type[fix_type] = []
            fixes_by_type[fix_type].append(fix)
        
        # Add details for each type
        for fix_type, fixes in fixes_by_type.items():
            type_name = fix_type.replace('_', ' ').title()
            description += f"### {type_name} Fixes ({len(fixes)})\n\n"
            
            for fix in fixes:
                if fix.get('file'):
                    description += f"- **{fix.get('file', 'Unknown file')}**: "
                    if fix.get('details', {}).get('action'):
                        description += f"{fix['details']['action']}\n"
                    else:
                        description += "Issue resolved\n"
            
            description += "\n"
        
        description += "## 🤖 Automation Details\n\n"
        description += "- **Generated by**: Code Review Agent\n"
        description += f"- **Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        description += "- **Fix method**: Automatic fixes applied\n\n"
        
        description += "## ✅ Verification\n\n"
        description += "Please review the changes and ensure they meet your requirements.\n"
        description += "All fixes are automatically generated and should be safe to apply.\n\n"
        
        description += "## 🔍 What Changed\n\n"
        description += "The following types of issues were automatically fixed:\n"
        
        for fix_type in fixes_by_type.keys():
            type_name = fix_type.replace('_', ' ').title()
            description += f"- {type_name}\n"
        
        return description
