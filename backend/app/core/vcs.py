import tempfile, shutil, os, subprocess, pathlib, time
import json
from pathlib import Path
from git import Repo
from .settings import settings

# Track repository information for smart cleanup
REPO_TRACKER_FILE = Path(settings.WORK_DIR) / ".repo_tracker.json"
CURRENT_REPO_FILE = Path(settings.WORK_DIR) / ".current_repo.json"

def _load_repo_tracker() -> dict:
    """Load repository tracking information"""
    if REPO_TRACKER_FILE.exists():
        try:
            with open(REPO_TRACKER_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_repo_tracker(tracker: dict):
    """Save repository tracking information"""
    REPO_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(REPO_TRACKER_FILE, 'w') as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving repo tracker: {e}")

def _load_current_repo() -> dict:
    """Load current active repository information"""
    if CURRENT_REPO_FILE.exists():
        try:
            with open(CURRENT_REPO_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_current_repo(current_repo: dict):
    """Save current active repository information"""
    CURRENT_REPO_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CURRENT_REPO_FILE, 'w') as f:
            json.dump(current_repo, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving current repo: {e}")

def _cleanup_previous_repo_for_new_work():
    """Clean up the previous repository when starting work on a new one"""
    if not settings.CLEANUP_SAME_URL_REPOS:
        return
        
    current_repo = _load_current_repo()
    if not current_repo:
        return  # No previous repo to clean up
    
    previous_job_id = current_repo.get('job_id')
    previous_repo_url = current_repo.get('repo_url')
    
    if previous_job_id and previous_repo_url:
        previous_path = Path(settings.WORK_DIR) / previous_job_id
        if previous_path.exists():
            try:
                print(f"🧹 Cleaning up previous repository for new work: {previous_repo_url}")
                shutil.rmtree(previous_path)
                
                # Remove from tracker
                tracker = _load_repo_tracker()
                if previous_job_id in tracker:
                    del tracker[previous_job_id]
                _save_repo_tracker(tracker)
                
                print(f"✅ Previous repository {previous_job_id} cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning up previous repo {previous_job_id}: {e}")

def prepare_workspace(job_id: str) -> Path:
    base = Path(settings.WORK_DIR)
    base.mkdir(parents=True, exist_ok=True)
    path = base / job_id
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def cleanup_old_workspaces(max_age_hours: int = None):
    """Clean up old workspace directories to save disk space"""
    if not settings.AUTO_CLEANUP_OLD_REPOS:
        return
        
    max_age_hours = max_age_hours or settings.CLEANUP_MAX_AGE_HOURS
    base = Path(settings.WORK_DIR)
    if not base.exists():
        return
    
    current_time = time.time()
    tracker = _load_repo_tracker()
    cleaned_count = 0
    
    for workspace_dir in base.iterdir():
        if workspace_dir.is_dir() and not workspace_dir.name.startswith('.'):
            try:
                # Check if directory is older than max_age_hours
                dir_age = current_time - workspace_dir.stat().st_mtime
                if dir_age > (max_age_hours * 3600):
                    print(f"🧹 Cleaning up old workspace: {workspace_dir}")
                    shutil.rmtree(workspace_dir)
                    # Remove from tracker
                    if workspace_dir.name in tracker:
                        del tracker[workspace_dir.name]
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Error cleaning up {workspace_dir}: {e}")
    
    if cleaned_count > 0:
        _save_repo_tracker(tracker)
        print(f"✅ Cleaned up {cleaned_count} old workspaces")

def cleanup_repo_after_review(job_id: str):
    """Clean up repository after review completion to save space"""
    # NOTE: This function is now DEPRECATED in favor of the new workflow
    # Repositories are kept until a new one is requested
    print(f"ℹ️ Repository {job_id} kept in workspace for potential further work")
    print(f"ℹ️ It will be automatically cleaned up when starting work on a new repository")

def cleanup_current_repo():
    """Clean up the current active repository"""
    current_repo = _load_current_repo()
    
    if not current_repo:
        return {"message": "No active repository found", "cleaned_count": 0}
    
    job_id = current_repo.get('job_id')
    repo_url = current_repo.get('repo_url')
    
    if not job_id:
        return {"message": "Invalid current repository data", "cleaned_count": 0}
    
    repo_path = Path(settings.WORK_DIR) / job_id
    if not repo_path.exists():
        return {"message": f"Repository path {repo_path} not found", "cleaned_count": 0}
    
    try:
        print(f"🧹 Cleaning up current active repository: {repo_url}")
        shutil.rmtree(repo_path)
        
        # Remove from tracker
        tracker = _load_repo_tracker()
        if job_id in tracker:
            del tracker[job_id]
        _save_repo_tracker(tracker)
        
        # Clear current repo
        _save_current_repo({})
        
        print(f"✅ Current repository {job_id} cleaned up")
        return {"message": f"Current repository {job_id} cleaned up", "cleaned_count": 1, "repo_url": repo_url}
        
    except Exception as e:
        error_msg = f"Error cleaning up current repo {job_id}: {e}"
        print(f"⚠️ {error_msg}")
        return {"message": error_msg, "cleaned_count": 0, "error": str(e)}

def cleanup_tracking_files():
    """Clear tracking files content (but keep the files for system functionality)"""
    print(f"🧹 Clearing tracking files content")
    _save_repo_tracker({})
    _save_current_repo({})
    
    print(f"✅ Tracking files content cleared")
    return {
        "message": "Tracking files content cleared", 
        "cleaned_count": 0,
        "cleaned_files": [".repo_tracker.json", ".current_repo.json"]
    }

def cleanup_all_repos():
    """Clean up all repositories in .workspaces directory"""
    base = Path(settings.WORK_DIR)
    if not base.exists():
        return {"message": "No workspaces directory found", "cleaned_count": 0}
    
    tracker = _load_repo_tracker()
    cleaned_count = 0
    cleaned_repos = []
    
    # First, clean up all repository directories
    for workspace_dir in base.iterdir():
        if workspace_dir.is_dir() and not workspace_dir.name.startswith('.'):
            try:
                repo_info = tracker.get(workspace_dir.name, {})
                repo_url = repo_info.get('repo_url', 'Unknown')
                print(f"🧹 Cleaning up repository: {workspace_dir} ({repo_url})")
                shutil.rmtree(workspace_dir)
                cleaned_count += 1
                cleaned_repos.append({
                    "job_id": workspace_dir.name,
                    "repo_url": repo_url
                })
            except Exception as e:
                print(f"⚠️ Error cleaning up {workspace_dir}: {e}")
    
    # Clear tracking files content (but keep the files for system functionality)
    print(f"🧹 Clearing tracking files content")
    _save_repo_tracker({})
    _save_current_repo({})
    
    print(f"✅ Cleaned up {cleaned_count} repositories and cleared tracking data")
    return {
        "message": f"Cleaned up {cleaned_count} repositories and cleared tracking data", 
        "cleaned_count": cleaned_count,
        "cleaned_repos": cleaned_repos
    }

def clone_repo(job_id: str, repo_url: str, branch: str | None = None) -> Path:
    # Clean up previous repository when starting work on a new one
    _cleanup_previous_repo_for_new_work()
    
    path = prepare_workspace(job_id)
    Repo.clone_from(repo_url, path)
    
    if branch:
        repo = Repo(path)
        try:
            # Try to checkout the specified branch
            repo.git.checkout(branch)
        except Exception:
            # If branch doesn't exist, try common alternatives
            try:
                # Try 'master' if 'main' fails
                if branch == 'main':
                    repo.git.checkout('master')
                elif branch == 'master':
                    repo.git.checkout('main')
                else:
                    # Just use the default branch
                    pass
            except Exception:
                # If all else fails, just use the default branch
                pass
    
    # Track this repository
    tracker = _load_repo_tracker()
    tracker[job_id] = {
        'repo_url': repo_url,
        'branch': branch,
        'cloned_at': time.time(),
        'path': str(path)
    }
    _save_repo_tracker(tracker)
    
    # Set this as the current active repository
    current_repo = {
        'job_id': job_id,
        'repo_url': repo_url,
        'branch': branch,
        'cloned_at': time.time(),
        'path': str(path),
        'status': 'active'
    }
    _save_current_repo(current_repo)
    
    print(f"📁 Cloned {repo_url} to {path} (job_id: {job_id})")
    print(f"📁 This is now the active repository - previous repos have been cleaned up")
    return path

def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def get_workspace_stats() -> dict:
    """Get statistics about current workspaces"""
    base = Path(settings.WORK_DIR)
    if not base.exists():
        return {"total_repos": 0, "total_size": "0B", "repos": [], "current_repo": None}
    
    tracker = _load_repo_tracker()
    current_repo = _load_current_repo()
    total_size = 0
    repos_info = []
    
    for workspace_dir in base.iterdir():
        if workspace_dir.is_dir() and not workspace_dir.name.startswith('.'):
            try:
                size = sum(f.stat().st_size for f in workspace_dir.rglob('*') if f.is_file())
                total_size += size
                
                repo_info = {
                    "job_id": workspace_dir.name,
                    "size": size,
                    "size_human": _format_size(size),
                    "age_hours": (time.time() - workspace_dir.stat().st_mtime) / 3600
                }
                
                # Add tracker info if available
                if workspace_dir.name in tracker:
                    repo_info.update({
                        "repo_url": tracker[workspace_dir.name].get("repo_url"),
                        "branch": tracker[workspace_dir.name].get("branch"),
                        "cloned_at": tracker[workspace_dir.name].get("cloned_at")
                    })
                
                # Mark if this is the current active repo
                if current_repo and workspace_dir.name == current_repo.get('job_id'):
                    repo_info['is_current'] = True
                    repo_info['status'] = 'active'
                else:
                    repo_info['is_current'] = False
                    repo_info['status'] = 'previous'
                
                repos_info.append(repo_info)
                
            except Exception as e:
                print(f"⚠️ Error getting stats for {workspace_dir}: {e}")
    
    return {
        "total_repos": len(repos_info),
        "total_size": _format_size(total_size),
        "total_size_bytes": total_size,
        "repos": repos_info,
        "current_repo": current_repo
    }

def get_current_repo_info() -> dict:
    """Get information about the current active repository"""
    return _load_current_repo()

def _format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"
