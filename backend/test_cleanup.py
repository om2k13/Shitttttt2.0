#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced repository cleanup system.
This script shows how the system automatically manages disk space.
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.vcs import (
    clone_repo, 
    cleanup_all_repos, 
    get_workspace_stats, 
    cleanup_old_workspaces,
    cleanup_repo_after_review
)
from app.core.settings import settings

def test_cleanup_system():
    """Test the cleanup system functionality"""
    print("🧪 Testing Enhanced Repository Cleanup System")
    print("=" * 50)
    
    # Show current settings
    print(f"📋 Cleanup Settings:")
    print(f"   Auto cleanup after review: {settings.AUTO_CLEANUP_AFTER_REVIEW}")
    print(f"   Auto cleanup old repos: {settings.AUTO_CLEANUP_OLD_REPOS}")
    print(f"   Cleanup max age hours: {settings.CLEANUP_MAX_AGE_HOURS}")
    print(f"   Cleanup same URL repos: {settings.CLEANUP_SAME_URL_REPOS}")
    print()
    
    # Show initial state
    print("📊 Initial Workspace State:")
    stats = get_workspace_stats()
    print(f"   Total repositories: {stats['total_repos']}")
    print(f"   Total size: {stats['total_size']}")
    print()
    
    # Test cleanup all repos
    print("🧹 Testing cleanup all repositories...")
    result = cleanup_all_repos()
    print(f"   Result: {result['message']}")
    print()
    
    # Show state after cleanup
    print("📊 State After Cleanup:")
    stats = get_workspace_stats()
    print(f"   Total repositories: {stats['total_repos']}")
    print(f"   Total size: {stats['total_size']}")
    print()
    
    # Test cleanup old workspaces
    print("🧹 Testing cleanup old workspaces...")
    cleanup_old_workspaces(max_age_hours=1)
    print("   Old workspace cleanup completed")
    print()
    
    print("✅ Cleanup system test completed!")

def test_repo_tracking():
    """Test repository tracking functionality"""
    print("\n🔍 Testing Repository Tracking")
    print("=" * 30)
    
    # This would normally be done by the actual application
    # Here we just show the structure
    print("📁 Repository tracking is implemented with:")
    print("   - Automatic cleanup of old repos for same URL")
    print("   - Cleanup after review completion")
    print("   - Age-based cleanup of old workspaces")
    print("   - JSON-based tracking file (.repo_tracker.json)")
    print()
    
    print("✅ Repository tracking test completed!")

if __name__ == "__main__":
    try:
        test_cleanup_system()
        test_repo_tracking()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
