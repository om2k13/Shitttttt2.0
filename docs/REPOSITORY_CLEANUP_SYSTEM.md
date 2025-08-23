# Repository Cleanup System

## Overview

The Enhanced Code Review Agent now includes an intelligent repository cleanup system that automatically manages disk space by implementing a **smart workflow-based cleanup strategy**.

## 🎯 **New Workflow Strategy**

### **How It Works Now:**
1. **Agent gets a GitHub repo to review**
2. **Agent reviews, generates report, makes changes if needed**
3. **Agent pushes changes to GitHub if modifications were made**
4. **Repository stays in workspace for potential further work**
5. **When agent gets a NEW repo, previous repo is automatically cleaned up**

### **Key Benefits:**
- ✅ **No premature cleanup** - repo available for further work
- ✅ **Automatic space management** when switching to new repos
- ✅ **Smart cleanup** - only one repo at a time
- ✅ **User control** - cleanup happens when deciding to work on something new

## Features

### 🧹 Smart Cleanup
- **Workflow-based Cleanup**: Repositories are kept until new work is requested
- **Automatic Previous Repo Cleanup**: When starting work on a new repo, previous one is automatically removed
- **Age-based Cleanup**: Old repositories are automatically removed based on configurable age limits
- **Smart Tracking**: All repositories are tracked with metadata for intelligent cleanup decisions

### 📊 Monitoring & Statistics
- Real-time workspace statistics
- Current active repository tracking
- Disk usage monitoring
- Repository age tracking
- Workflow status information

### ⚙️ Configurable Behavior
- Enable/disable automatic cleanup features
- Configurable age limits for cleanup
- Granular control over cleanup behavior

## Configuration

### Environment Variables

Add these to your `.env` file to customize cleanup behavior:

```bash
# Repository cleanup settings - NEW WORKFLOW
AUTO_CLEANUP_AFTER_REVIEW=false         # Repos kept until new work requested
AUTO_CLEANUP_OLD_REPOS=true            # Age-based cleanup still available
CLEANUP_MAX_AGE_HOURS=24               # Maximum age in hours before cleanup
CLEANUP_SAME_URL_REPOS=true            # Cleanup previous repo when starting new work
```

### Default Settings

```python
# Default values if not specified in .env
AUTO_CLEANUP_AFTER_REVIEW = False      # Changed: repos kept until new work requested
AUTO_CLEANUP_OLD_REPOS = True          # Age-based cleanup still available
CLEANUP_MAX_AGE_HOURS = 24             # Increased: repos kept longer
CLEANUP_SAME_URL_REPOS = True          # Cleanup previous repo when starting new work
```

## API Endpoints

### Cleanup Endpoints

```bash
# Clean up old workspaces
POST /api/jobs/cleanup-workspaces?max_age_hours=24

# Clean up all repositories
POST /api/jobs/cleanup-all-repos

# Clean up current active repository
POST /api/jobs/cleanup-current-repo

# Get current repository information
GET /api/jobs/current-repo

# Get workspace statistics
GET /api/jobs/workspace-stats

# Clean up specific repository after review (DEPRECATED)
POST /api/jobs/cleanup-after-review/{job_id}
```

### Example Usage

```bash
# Clean up workspaces older than 24 hours
curl -X POST "http://localhost:8000/api/jobs/cleanup-workspaces?max_age_hours=24"

# Get current repository information
curl "http://localhost:8000/api/jobs/current-repo"

# Clean up current active repository
curl -X POST "http://localhost:8000/api/jobs/cleanup-current-repo"

# Get current workspace statistics
curl "http://localhost:8000/api/jobs/workspace-stats"

# Emergency cleanup of all repositories
curl -X POST "http://localhost:8000/api/jobs/cleanup-all-repos"
```

## CLI Commands

### Workspace Management

```bash
# Navigate to backend directory
cd backend

# Show current active repository
python workspace_cli.py workspaces current

# Show workspace statistics
python workspace_cli.py workspaces stats

# Show comprehensive workspace status
python workspace_cli.py workspaces status

# Show all cleanup options and their status
python workspace_cli.py workspaces cleanup-options

# Show workflow information
python workspace_cli.py workspaces workflow

# Clean up current active repository
python workspace_cli.py workspaces cleanup-current

# Clean up all repositories
python workspace_cli.py workspaces cleanup-all --force

# Clean up old workspaces (age-based)
python workspace_cli.py workspaces cleanup --max-age 24

# Monitor workspaces in real-time
python workspace_cli.py workspaces monitor
```

### System Information

```bash
# Show system configuration
python workspace_cli.py system info

# Show cleanup configuration
python workspace_cli.py system config
```

## Cleanup Options

### 🎯 **Current Repository Cleanup**
- **Command**: `workspaces cleanup-current`
- **Purpose**: Clean up the current active repository you're working on
- **Use Case**: When you're done with current work and want to free up space immediately
- **Safety**: Confirmation prompt (use `--force` to skip)

### 🧹 **All Repositories Cleanup**
- **Command**: `workspaces cleanup-all`
- **Purpose**: Clean up all repositories and clear tracking data
- **Use Case**: Emergency cleanup, starting fresh, or freeing up maximum space
- **Safety**: Confirmation prompt (use `--force` to skip)
- **Note**: Tracking files (`.repo_tracker.json`, `.current_repo.json`) are cleared but preserved for system functionality

### 📊 **Tracking Files Cleanup**
- **Command**: `workspaces cleanup-tracking`
- **Purpose**: Clear tracking files content (but keep the files for system functionality)
- **Use Case**: Reset repository tracking state without touching actual repositories
- **Safety**: Confirmation prompt (use `--force` to skip)
- **Note**: Files are preserved but content is cleared to `{}`

### ⏰ **Age-based Cleanup**
- **Command**: `workspaces cleanup --max-age <hours>`
- **Purpose**: Clean up repositories older than specified hours
- **Use Case**: Regular maintenance, cleaning up old unused repositories
- **Example**: `workspaces cleanup --max-age 48` (clean repos older than 2 days)

### 📊 **Status and Options**
- **Command**: `workspaces status`
- **Purpose**: Comprehensive workspace status with cleanup recommendations
- **Use Case**: Understanding current state and available cleanup options

- **Command**: `workspaces cleanup-options`
- **Purpose**: Show all cleanup options and their current status
- **Use Case**: Quick overview of what can be cleaned up and how

## How It Works

### 1. Repository Tracking
- Each cloned repository is tracked with metadata:
  - Repository URL
  - Branch name
  - Clone timestamp
  - Job ID
  - File path
  - Status (active/previous)

### 2. Smart Cleanup Triggers
- **New Work Request**: Previous repo is cleaned up when starting work on a new one
- **Age-based Cleanup**: Old workspaces are cleaned up based on age limits
- **Manual Cleanup**: API endpoints and CLI commands for immediate cleanup

### 3. Cleanup Logic
```python
# Example: Clean up previous repo when starting new work
def _cleanup_previous_repo_for_new_work():
    # Get current active repository
    # Clean it up before cloning new one
    # Set new repo as current active
```

### 4. Safety Features
- Cleanup only affects `.workspaces` directory
- Hidden files (starting with `.`) are preserved
- Error handling for failed cleanup operations
- Confirmation prompts for destructive operations

## File Structure

```
backend/
├── .workspaces/                    # Cloned repositories
│   ├── .repo_tracker.json         # Repository metadata
│   ├── .current_repo.json         # Current active repository
│   └── [job_id]/                  # Individual repositories
├── app/
│   ├── core/
│   │   ├── vcs.py                 # Core cleanup logic
│   │   └── settings.py            # Cleanup configuration
│   ├── api/
│   │   └── jobs.py                # Cleanup API endpoints
│   ├── cli/
│   │   └── workspace_cli.py       # CLI commands
│   └── review/
│       └── pipeline.py            # Workflow integration
```

## Best Practices

### 1. Workflow Management
- **Single Repository Focus**: Work on one repository at a time
- **Complete Work Cycles**: Finish all work on current repo before switching
- **Regular Monitoring**: Use CLI commands to track workspace status

### 2. Configuration Tuning
- Adjust `CLEANUP_MAX_AGE_HOURS` based on your workflow needs
- Keep `CLEANUP_SAME_URL_REPOS=true` for automatic space management
- Monitor cleanup performance and adjust settings accordingly

### 3. Manual Cleanup
- Use `cleanup-all-repos` sparingly and with caution
- Prefer workflow-based automatic cleanup
- Use age-based cleanup for regular maintenance

## Troubleshooting

### Common Issues

1. **Repository Not Being Cleaned Up**
   - Check if it's the current active repository
   - Verify configuration settings in `.env`
   - Use `workspaces current` to see active repo

2. **High Disk Usage**
   - Run `workspaces stats` to see current usage
   - Check if you have multiple repositories
   - Use `cleanup-all-repos` to free up space immediately

3. **Workflow Confusion**
   - Use `workspaces workflow` to understand the current system
   - Check `workspaces current` for active repository
   - Verify cleanup behavior with `system config`

### Debug Commands

```bash
# Check current repository status
python workspace_cli.py workspaces current

# Verify workflow information
python workspace_cli.py workspaces workflow

# Check workspace state
python workspace_cli.py workspaces stats

# Verify configuration
python workspace_cli.py system config
```

## Performance Impact

- **Minimal**: Cleanup operations are lightweight and run only when needed
- **Efficient**: Only necessary files are removed
- **Non-blocking**: Cleanup doesn't interfere with active work
- **Configurable**: Can be adjusted based on performance needs

## Migration from Old System

### What Changed:
- **Before**: Repositories were cleaned up immediately after review
- **After**: Repositories are kept until new work is requested

### Benefits of New System:
- ✅ **Better workflow support** - can continue working on same repo
- ✅ **Automatic space management** - no manual cleanup needed
- ✅ **Improved user experience** - repos available for further work
- ✅ **Smart resource management** - only one repo at a time

## Future Enhancements

- [ ] Workflow templates for different use cases
- [ ] Repository backup before cleanup
- [ ] Cleanup scheduling options
- [ ] Integration with external monitoring systems
- [ ] Cleanup analytics and reporting
- [ ] Selective cleanup based on repository size or type
