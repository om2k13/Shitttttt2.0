#!/usr/bin/env python3
"""
Simple CLI for workspace management and repository cleanup.
This CLI focuses only on workspace operations to avoid import dependencies.
"""

import click
import sys
from pathlib import Path
import time

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.vcs import (
    cleanup_all_repos, 
    get_workspace_stats, 
    cleanup_old_workspaces, 
    cleanup_repo_after_review
)
from app.core.settings import settings

@click.group()
def cli():
    """Workspace Management CLI - Repository Cleanup and Monitoring"""
    pass

@cli.group()
def workspaces():
    """Manage workspace directories and repositories"""
    pass

@workspaces.command()
@click.option('--max-age', default=1, help='Maximum age in hours for cleanup (default: 1)')
def cleanup(max_age):
    """Clean up old workspace directories"""
    click.echo(f"🧹 Cleaning up workspaces older than {max_age} hours...")
    cleanup_old_workspaces(max_age_hours=max_age)
    click.echo("✅ Workspace cleanup completed")

@workspaces.command()
def stats():
    """Show workspace statistics"""
    stats = get_workspace_stats()
    click.echo(f"📊 Workspace Statistics:")
    click.echo(f"   Total repositories: {stats['total_repos']}")
    click.echo(f"   Total size: {stats['total_size']}")
    
    # Show current active repository
    if stats.get('current_repo'):
        current = stats['current_repo']
        click.echo(f"\n🎯 Current Active Repository:")
        click.echo(f"   Job ID: {current.get('job_id')}")
        click.echo(f"   URL: {current.get('repo_url')}")
        click.echo(f"   Branch: {current.get('branch', 'default')}")
        click.echo(f"   Status: {current.get('status', 'active')}")
        click.echo(f"   Cloned: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current.get('cloned_at', 0)))}")
    else:
        click.echo(f"\n🎯 Current Active Repository: None")
    
    if stats['repos']:
        click.echo("\n📁 Repository Details:")
        for repo in stats['repos']:
            status_emoji = "🎯" if repo.get('is_current') else "📁"
            status_text = "ACTIVE" if repo.get('is_current') else "PREVIOUS"
            click.echo(f"   {status_emoji} {repo['job_id']}: {repo['size_human']} (age: {repo['age_hours']:.1f}h) - {status_text}")
            if 'repo_url' in repo:
                click.echo(f"     URL: {repo['repo_url']}")
                if 'branch' in repo:
                    click.echo(f"     Branch: {repo['branch']}")
    else:
        click.echo("   No repositories found")
    
    # Show workflow information
    click.echo(f"\n🔄 Workflow Information:")
    click.echo(f"   • Current repository stays in workspace until new work is requested")
    click.echo(f"   • Previous repositories are automatically cleaned up when starting new work")
    click.echo(f"   • Manual cleanup available via 'cleanup-all' command")

@workspaces.command()
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
def cleanup_all(force):
    """Clean up all repositories in .workspaces directory"""
    if not force:
        if not click.confirm("⚠️ This will delete ALL repositories. Are you sure?"):
            click.echo("❌ Cleanup cancelled")
            return
    
    click.echo("🧹 Cleaning up all repositories...")
    result = cleanup_all_repos()
    click.echo(f"✅ {result['message']}")
    
    # Show details of what was cleaned up
    if result.get('cleaned_repos'):
        click.echo(f"\n📋 Cleaned up repositories:")
        for repo in result['cleaned_repos']:
            click.echo(f"   • {repo['job_id']}: {repo['repo_url']}")

@workspaces.command()
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
def cleanup_current(force):
    """Clean up the current active repository"""
    from app.core.vcs import get_current_repo_info, cleanup_current_repo
    
    current_repo = get_current_repo_info()
    if not current_repo:
        click.echo("📭 No active repository found to clean up")
        return
    
    click.echo(f"🎯 Current Active Repository:")
    click.echo(f"   Job ID: {current_repo.get('job_id')}")
    click.echo(f"   URL: {current_repo.get('repo_url')}")
    click.echo(f"   Branch: {current_repo.get('branch', 'default')}")
    
    if not force:
        if not click.confirm(f"⚠️ Clean up the current repository '{current_repo.get('repo_url')}'?"):
            click.echo("❌ Cleanup cancelled")
            return
    
    click.echo("🧹 Cleaning up current repository...")
    result = cleanup_current_repo()
    click.echo(f"✅ {result['message']}")

@workspaces.command()
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
def cleanup_tracking(force):
    """Clear tracking files content (but keep the files for system functionality)"""
    from app.core.vcs import cleanup_tracking_files
    
    if not force:
        if not click.confirm("⚠️ Clear tracking files content? This will reset repository tracking state."):
            click.echo("❌ Cleanup cancelled")
            return
    
    click.echo("🧹 Clearing tracking files content...")
    result = cleanup_tracking_files()
    click.echo(f"✅ {result['message']}")
    
    if result.get('cleaned_files'):
        click.echo(f"\n📋 Tracking files content cleared:")
        for file in result['cleaned_files']:
            click.echo(f"   • {file}")

@workspaces.command()
@click.argument('job_id')
def cleanup_repo(job_id):
    """Clean up a specific repository by job ID"""
    click.echo(f"🧹 Cleaning up repository {job_id}...")
    cleanup_repo_after_review(job_id)
    click.echo(f"✅ Repository {job_id} cleaned up")

@workspaces.command()
def monitor():
    """Monitor workspace directory in real-time"""
    import time
    click.echo("📊 Monitoring workspace directory... (Press Ctrl+C to stop)")
    
    try:
        while True:
            stats = get_workspace_stats()
            click.echo(f"\r📁 Repos: {stats['total_repos']} | 💾 Size: {stats['total_size']} | ⏰ {time.strftime('%H:%M:%S')}", nl=False)
            time.sleep(5)
    except KeyboardInterrupt:
        click.echo("\n✅ Monitoring stopped")

@workspaces.command()
def current():
    """Show information about the current active repository"""
    from app.core.vcs import get_current_repo_info
    
    current_repo = get_current_repo_info()
    if current_repo:
        click.echo("🎯 Current Active Repository:")
        click.echo(f"   Job ID: {current_repo.get('job_id')}")
        click.echo(f"   Repository URL: {current_repo.get('repo_url')}")
        click.echo(f"   Branch: {current_repo.get('branch', 'default')}")
        click.echo(f"   Status: {current_repo.get('status', 'active')}")
        click.echo(f"   Cloned: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_repo.get('cloned_at', 0)))}")
        click.echo(f"   Path: {current_repo.get('path')}")
        
        click.echo(f"\n🔄 Workflow Status:")
        click.echo(f"   • This repository will stay in workspace until new work is requested")
        click.echo(f"   • You can continue working on it, make changes, push updates, etc.")
        click.echo(f"   • It will be automatically cleaned up when you start work on a new repository")
    else:
        click.echo("📭 No active repository found")
        click.echo("   • Clone a repository to start working")
        click.echo("   • Use 'workspaces stats' to see workspace status")

@workspaces.command()
def workflow():
    """Show the current workflow information"""
    click.echo("🔄 Repository Workflow Information:")
    click.echo("")
    click.echo("📋 Current Workflow:")
    click.echo("   1. Agent gets a GitHub repo to review")
    click.echo("   2. Agent reviews, generates report, makes changes if needed")
    click.echo("   3. Agent pushes changes to GitHub if modifications were made")
    click.echo("   4. Repository stays in workspace for potential further work")
    click.echo("   5. When agent gets a NEW repo, previous repo is automatically cleaned up")
    click.echo("")
    click.echo("✅ Benefits:")
    click.echo("   • No premature cleanup - repo available for further work")
    click.echo("   • Automatic space management when switching to new repos")
    click.echo("   • Smart cleanup - only one repo at a time")
    click.echo("   • User control - cleanup happens when deciding to work on something new")
    click.echo("")
    click.echo("🧹 Manual Cleanup Options:")
    click.echo("   • 'workspaces cleanup-current' - Clean up current active repository")
    click.echo("   • 'workspaces cleanup-all' - Clean up all repositories")
    click.echo("   • 'workspaces cleanup --max-age N' - Age-based cleanup")
    click.echo("   • 'workspaces cleanup-options' - Show all cleanup options")
    click.echo("   • 'workspaces status' - Comprehensive status and cleanup info")
    click.echo("")
    click.echo("🛠️ Commands:")
    click.echo("   • 'workspaces current' - Show current repository")
    click.echo("   • 'workspaces stats' - Show workspace statistics")
    click.echo("   • 'workspaces monitor' - Real-time monitoring")

@workspaces.command()
def cleanup_options():
    """Show all available cleanup options and their current status"""
    from app.core.vcs import get_current_repo_info, get_workspace_stats
    
    current_repo = get_current_repo_info()
    stats = get_workspace_stats()
    
    click.echo("🧹 Cleanup Options and Status:")
    click.echo("=" * 40)
    
    # Current repository status
    if current_repo:
        click.echo(f"🎯 Current Active Repository:")
        click.echo(f"   • {current_repo.get('repo_url')} (Job ID: {current_repo.get('job_id')})")
        click.echo(f"   • Status: Active - available for cleanup")
        click.echo(f"   • Command: 'workspaces cleanup-current'")
    else:
        click.echo(f"📭 Current Active Repository: None")
        click.echo(f"   • No repository to clean up")
    
    click.echo()
    
    # All repositories status
    if stats['repos']:
        click.echo(f"📁 All Repositories ({stats['total_repos']}):")
        click.echo(f"   • Total size: {stats['total_size']}")
        click.echo(f"   • Command: 'workspaces cleanup-all'")
        
        for repo in stats['repos']:
            status = "🎯 ACTIVE" if repo.get('is_current') else "📁 PREVIOUS"
            click.echo(f"   • {repo['job_id']}: {repo['size_human']} - {status}")
    else:
        click.echo(f"📁 All Repositories: None")
        click.echo(f"   • No repositories to clean up")
    
    click.echo()
    
    # Available commands
    click.echo("🛠️ Available Cleanup Commands:")
    click.echo("   • 'workspaces cleanup-current'     - Clean up current active repository")
    click.echo("   • 'workspaces cleanup-all'         - Clean up all repositories")
    click.echo("   • 'workspaces cleanup --max-age N' - Clean up old workspaces (age-based)")
    click.echo("   • 'workspaces cleanup-repo <id>'   - Clean up specific repository by ID")
    click.echo("   • 'workspaces cleanup-tracking'    - Clear tracking files content")
    
    click.echo()
    
    # Workflow information
    click.echo("🔄 Workflow Information:")
    click.echo("   • Current repository stays until new work is requested")
    click.echo("   • Previous repositories auto-cleaned when starting new work")
    click.echo("   • Manual cleanup available for immediate space management")
    click.echo("   • Tracking files content can be cleared separately if needed")

@workspaces.command()
def status():
    """Show comprehensive workspace status and cleanup options"""
    from app.core.vcs import get_current_repo_info, get_workspace_stats
    
    current_repo = get_current_repo_info()
    stats = get_workspace_stats()
    
    click.echo("📊 Comprehensive Workspace Status:")
    click.echo("=" * 40)
    
    # Basic stats
    click.echo(f"📈 Statistics:")
    click.echo(f"   • Total repositories: {stats['total_repos']}")
    click.echo(f"   • Total disk usage: {stats['total_size']}")
    click.echo(f"   • Workspace path: {Path(settings.WORK_DIR).absolute()}")
    
    click.echo()
    
    # Current repository
    if current_repo:
        click.echo(f"🎯 Current Active Repository:")
        click.echo(f"   • URL: {current_repo.get('repo_url')}")
        click.echo(f"   • Job ID: {current_repo.get('job_id')}")
        click.echo(f"   • Branch: {current_repo.get('branch', 'default')}")
        click.echo(f"   • Cloned: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_repo.get('cloned_at', 0)))}")
        click.echo(f"   • Path: {current_repo.get('path')}")
        click.echo(f"   • Status: Active - can be cleaned up manually")
    else:
        click.echo(f"🎯 Current Active Repository: None")
        click.echo(f"   • Ready for first repository")
    
    click.echo()
    
    # Repository details
    if stats['repos']:
        click.echo(f"📁 Repository Details:")
        for repo in stats['repos']:
            status_emoji = "🎯" if repo.get('is_current') else "📁"
            status_text = "ACTIVE" if repo.get('is_current') else "PREVIOUS"
            click.echo(f"   {status_emoji} {repo['job_id']}: {repo['size_human']} (age: {repo['age_hours']:.1f}h) - {status_text}")
            if 'repo_url' in repo:
                click.echo(f"     URL: {repo['repo_url']}")
                if 'branch' in repo:
                    click.echo(f"     Branch: {repo['branch']}")
    else:
        click.echo(f"📁 Repository Details: None")
    
    click.echo()
    
    # Cleanup options
    click.echo("🧹 Cleanup Options:")
    if current_repo:
        click.echo(f"   • Clean current repo: 'workspaces cleanup-current'")
    if stats['total_repos'] > 0:
        click.echo(f"   • Clean all repos: 'workspaces cleanup-all'")
    click.echo(f"   • Age-based cleanup: 'workspaces cleanup --max-age 24'")
    
    click.echo()
    
    # Workflow status
    click.echo("🔄 Workflow Status:")
    click.echo(f"   • Auto cleanup after review: {settings.AUTO_CLEANUP_AFTER_REVIEW}")
    click.echo(f"   • Auto cleanup old repos: {settings.AUTO_CLEANUP_OLD_REPOS}")
    click.echo(f"   • Cleanup max age hours: {settings.CLEANUP_MAX_AGE_HOURS}")
    click.echo(f"   • Cleanup same URL repos: {settings.CLEANUP_SAME_URL_REPOS}")

@cli.group()
def system():
    """System management commands"""
    pass

@system.command()
def info():
    """Show system information"""
    click.echo("🔧 System Information:")
    click.echo(f"   Workspace directory: {Path(settings.WORK_DIR).absolute()}")
    click.echo(f"   Database URL: {settings.DATABASE_URL}")
    click.echo(f"   LLM Provider: {settings.LLM_PROVIDER}")
    click.echo(f"   Environment: {settings.ENV}")
    
    # Check workspace directory
    workspace_path = Path(settings.WORK_DIR)
    if workspace_path.exists():
        click.echo(f"   Workspace exists: ✅")
        click.echo(f"   Workspace path: {workspace_path.absolute()}")
    else:
        click.echo(f"   Workspace exists: ❌")

@system.command()
def config():
    """Show cleanup configuration"""
    click.echo("⚙️ Cleanup Configuration:")
    click.echo(f"   Auto cleanup after review: {settings.AUTO_CLEANUP_AFTER_REVIEW}")
    click.echo(f"   Auto cleanup old repos: {settings.AUTO_CLEANUP_OLD_REPOS}")
    click.echo(f"   Cleanup max age hours: {settings.CLEANUP_MAX_AGE_HOURS}")
    click.echo(f"   Cleanup same URL repos: {settings.CLEANUP_SAME_URL_REPOS}")

if __name__ == '__main__':
    cli()
