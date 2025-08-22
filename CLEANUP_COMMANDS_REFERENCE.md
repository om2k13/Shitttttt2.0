# Repository Cleanup Commands Quick Reference

## 🚀 **Quick Start**

```bash
cd backend
python workspace_cli.py workspaces status
```

## 🧹 **Cleanup Commands**

### **Current Repository Cleanup**
```bash
# Clean up the repository you're currently working on
python workspace_cli.py workspaces cleanup-current

# Force cleanup without confirmation
python workspace_cli.py workspaces cleanup-current --force
```

### **All Repositories Cleanup**
```bash
# Clean up all repositories (emergency cleanup)
python workspace_cli.py workspaces cleanup-all

# Force cleanup without confirmation
python workspace_cli.py workspaces cleanup-all --force
```

### **Age-based Cleanup**
```bash
# Clean up repositories older than 24 hours
python workspace_cli.py workspaces cleanup --max-age 24

# Clean up repositories older than 48 hours (2 days)
python workspace_cli.py workspaces cleanup --max-age 48

# Clean up repositories older than 168 hours (1 week)
python workspace_cli.py workspaces cleanup --max-age 168
```

## 📊 **Status and Information Commands**

### **Current Status**
```bash
# Show current active repository
python workspace_cli.py workspaces current

# Show workspace statistics
python workspace_cli.py workspaces stats

# Show comprehensive status with cleanup options
python workspace_cli.py workspaces status

# Show all cleanup options and their status
python workspace_cli.py workspaces cleanup-options
```

### **Workflow Information**
```bash
# Show workflow explanation
python workspace_cli.py workspaces workflow

# Show system configuration
python workspace_cli.py system config

# Show system information
python workspace_cli.py system info
```

### **Real-time Monitoring**
```bash
# Monitor workspace in real-time
python workspace_cli.py workspaces monitor
```

## 🔄 **API Endpoints**

### **Cleanup Endpoints**
```bash
# Clean up current repository
POST /api/jobs/cleanup-current-repo

# Clean up all repositories
POST /api/jobs/cleanup-all-repos

# Clean up old workspaces
POST /api/jobs/cleanup-workspaces?max_age_hours=24

# Get current repository info
GET /api/jobs/current-repo

# Get workspace statistics
GET /api/jobs/workspace-stats
```

## 📋 **Use Case Examples**

### **Scenario 1: Done with Current Work**
```bash
# Check what you're working on
python workspace_cli.py workspaces current

# Clean up current repository
python workspace_cli.py workspaces cleanup-current
```

### **Scenario 2: Emergency Cleanup**
```bash
# See what's taking up space
python workspace_cli.py workspaces status

# Clean up everything
python workspace_cli.py workspaces cleanup-all --force
```

### **Scenario 3: Regular Maintenance**
```bash
# Check for old repositories
python workspace_cli.py workspaces cleanup --max-age 48

# See cleanup options
python workspace_cli.py workspaces cleanup-options
```

### **Scenario 4: Starting Fresh**
```bash
# Check current state
python workspace_cli.py workspaces status

# Clean up everything
python workspace_cli.py workspaces cleanup-all

# Verify cleanup
python workspace_cli.py workspaces stats
```

## ⚙️ **Configuration**

### **Environment Variables** (`.env` file)
```bash
# Repository cleanup settings
AUTO_CLEANUP_AFTER_REVIEW=false         # Repos kept until new work requested
AUTO_CLEANUP_OLD_REPOS=true            # Age-based cleanup available
CLEANUP_MAX_AGE_HOURS=24               # Default max age for cleanup
CLEANUP_SAME_URL_REPOS=true            # Auto-cleanup previous repo when starting new work
```

### **Check Current Configuration**
```bash
python workspace_cli.py system config
```

## 🎯 **Workflow Summary**

### **Automatic Cleanup (Default Behavior)**
1. **Clone repository** → Becomes current active
2. **Work on repository** → Review, report, changes, push
3. **Repository stays** → Available for further work
4. **Clone NEW repository** → Previous repo automatically cleaned up

### **Manual Cleanup Options**
- **`cleanup-current`**: Clean up repository you're working on
- **`cleanup-all`**: Clean up all repositories
- **`cleanup --max-age N`**: Age-based cleanup
- **`cleanup-options`**: See what can be cleaned up

## 🚨 **Safety Features**

- **Confirmation prompts** for destructive operations
- **`--force` flag** to skip confirmations
- **Detailed feedback** on what's being cleaned up
- **Error handling** for failed cleanup operations
- **Repository tracking** to prevent accidental deletions

## 💡 **Best Practices**

1. **Use `status` command** to understand current state
2. **Use `cleanup-current`** when done with specific work
3. **Use `cleanup-all`** only for emergency cleanup
4. **Use age-based cleanup** for regular maintenance
5. **Monitor with `cleanup-options`** to see available actions
