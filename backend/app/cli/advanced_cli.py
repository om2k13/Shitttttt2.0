#!/usr/bin/env python3
"""
Advanced CLI interface for the Enhanced Code Review Agent
Provides command-line access to all advanced features
"""

import click
import asyncio
import json
from pathlib import Path
from typing import Optional, List
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.review.pipeline import run_review
from app.review.pr_analyzer import analyze_pr_changes
from app.review.security_analyzer import run_security_analysis
from app.review.performance_analyzer import analyze_code_performance
from app.review.api_analyzer import analyze_api_changes_for_branches, generate_api_documentation_for_branch
from app.review.test_generator import generate_test_plan_for_changes
from app.review.ml_analyzer import (
    predict_finding_severity, detect_false_positive, predict_risk_score,
    generate_intelligent_suggestions, train_ml_models, get_ml_model_performance
)
from app.review.advanced_reporter import generate_executive_report
from app.integrations.github import post_findings_to_pr, create_pr_review
from app.integrations.cicd import create_ci_config_for_platform, detect_ci_platform
from app.integrations.notifications import (
    send_critical_finding_notification, send_analysis_complete_notification,
    send_pr_review_notification, test_notification_system
)
from app.core.enhanced_config import (
    get_preset_config, merge_configs, validate_config, export_config, import_config
)
from app.core.vcs import cleanup_all_repos, get_workspace_stats, cleanup_old_workspaces, cleanup_repo_after_review
from app.core.settings import settings

@click.group()
@click.version_option(version="3.0.0")
def cli():
    """Enhanced Code Review Agent CLI - Enterprise-Grade Code Analysis"""
    pass

@cli.group()
def analyze():
    """Code analysis commands"""
    pass

@analyze.command()
@click.option('--repo-url', required=True, help='Repository URL to analyze')
@click.option('--analysis-type', 
              type=click.Choice(['security', 'performance', 'api', 'comprehensive']),
              default='comprehensive', help='Type of analysis to perform')
@click.option('--severity-threshold',
              type=click.Choice(['low', 'medium', 'high', 'critical']),
              default='low', help='Minimum severity to report')
@click.option('--enable-ml', is_flag=True, help='Enable ML-based analysis')
@click.option('--output-format', 
              type=click.Choice(['json', 'yaml', 'html']),
              default='json', help='Output format for results')
@click.option('--save-results', is_flag=True, help='Save results to file')
async def repository(repo_url: str, analysis_type: str, severity_threshold: str, 
                   enable_ml: bool, output_format: str, save_results: bool):
    """Analyze a complete repository"""
    click.echo(f"🔍 Starting {analysis_type} analysis of {repo_url}")
    
    try:
        # Run the appropriate analysis
        if analysis_type == 'security':
            results = await run_security_analysis(repo_url)
        elif analysis_type == 'performance':
            results = await analyze_code_performance(repo_url)
        elif analysis_type == 'api':
            results = await generate_api_documentation_for_branch(repo_url)
        else:  # comprehensive
            results = await run_review(repo_url)
        
        # Apply ML analysis if enabled
        if enable_ml and 'findings' in results:
            click.echo("🤖 Applying ML-based analysis...")
            for finding in results['findings']:
                # Predict severity
                predicted_severity = await predict_finding_severity(finding)
                finding['ml_predicted_severity'] = predicted_severity
                
                # Detect false positives
                fp_probability = await detect_false_positive(finding)
                finding['false_positive_probability'] = fp_probability
                
                # Predict risk score
                risk_score = await predict_risk_score(finding)
                finding['ml_risk_score'] = risk_score
                
                # Generate intelligent suggestions
                suggestions = await generate_intelligent_suggestions(finding)
                finding['ml_suggestions'] = suggestions
        
        # Format output
        if output_format == 'json':
            output = json.dumps(results, indent=2)
        elif output_format == 'yaml':
            import yaml
            output = yaml.dump(results, default_flow_style=False)
        else:
            output = str(results)
        
        # Display results
        click.echo(f"\n📊 Analysis Results ({analysis_type}):")
        click.echo(output)
        
        # Save results if requested
        if save_results:
            filename = f"analysis_results_{analysis_type}_{Path(repo_url).name}.{output_format}"
            with open(filename, 'w') as f:
                f.write(output)
            click.echo(f"\n💾 Results saved to {filename}")
        
        # Send notifications if configured
        if 'findings' in results:
            repo_info = {"name": Path(repo_url).name}
            await send_analysis_complete_notification(results, repo_info)
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {str(e)}", err=True)
        sys.exit(1)

@analyze.command()
@click.option('--repo-url', required=True, help='Repository URL')
@click.option('--pr-number', required=True, type=int, help='Pull request number')
@click.option('--base-branch', default='main', help='Base branch for comparison')
@click.option('--head-branch', help='Head branch (defaults to PR branch)')
@click.option('--enable-ml', is_flag=True, help='Enable ML-based analysis')
async def pr(repo_url: str, pr_number: int, base_branch: str, head_branch: Optional[str], enable_ml: bool):
    """Analyze a specific pull request"""
    click.echo(f"🔍 Analyzing PR #{pr_number} in {repo_url}")
    
    try:
        # Analyze PR changes
        results = await analyze_pr_changes(repo_url, base_branch, head_branch or f"pr-{pr_number}")
        
        # Apply ML analysis if enabled
        if enable_ml and 'findings' in results:
            click.echo("🤖 Applying ML-based analysis...")
            for finding in results['findings']:
                # Add ML insights
                predicted_severity = await predict_finding_severity(finding)
                finding['ml_predicted_severity'] = predicted_severity
                
                fp_probability = await detect_false_positive(finding)
                finding['false_positive_probability'] = fp_probability
                
                risk_score = await predict_risk_score(finding)
                finding['ml_risk_score'] = risk_score
                
                suggestions = await generate_intelligent_suggestions(finding)
                finding['ml_suggestions'] = suggestions
        
        # Display results
        click.echo(f"\n📊 PR Analysis Results:")
        click.echo(json.dumps(results, indent=2))
        
        # Send PR review notification
        pr_info = {
            "repo_name": Path(repo_url).name,
            "pr_number": pr_number,
            "branch_name": head_branch or f"pr-{pr_number}"
        }
        await send_pr_review_notification(pr_info, results.get('findings', []))
        
    except Exception as e:
        click.echo(f"❌ PR analysis failed: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def ml():
    """Machine Learning commands"""
    pass

@ml.command()
@click.option('--training-data', required=True, help='Path to training data file')
async def train(training_data: str):
    """Train ML models with historical data"""
    click.echo(f"🤖 Training ML models with {training_data}")
    
    try:
        with open(training_data, 'r') as f:
            training_data = json.load(f)
        
        await train_ml_models(training_data)
        click.echo("✅ ML models trained successfully!")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {str(e)}", err=True)
        sys.exit(1)

@ml.command()
async def status():
    """Check ML model status and performance"""
    click.echo("🤖 Checking ML model status...")
    
    try:
        performance = await get_ml_model_performance()
        click.echo("ML Model Status:")
        click.echo(json.dumps(performance, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def report():
    """Reporting commands"""
    pass

@report.command()
@click.option('--findings-file', required=True, help='Path to findings JSON file')
@click.option('--repo-info', required=True, help='Path to repository info JSON file')
@click.option('--report-type',
              type=click.Choice(['executive', 'technical', 'security', 'performance']),
              default='executive', help='Type of report to generate')
@click.option('--output-format',
              type=click.Choice(['json', 'yaml', 'html']),
              default='json', help='Output format')
async def generate(findings_file: str, repo_info: str, report_type: str, output_format: str):
    """Generate advanced reports"""
    click.echo(f"📊 Generating {report_type} report...")
    
    try:
        # Load data
        with open(findings_file, 'r') as f:
            findings = json.load(f)
        
        with open(repo_info, 'r') as f:
            repo_info_data = json.load(f)
        
        # Generate report
        if report_type == 'executive':
            report = await generate_executive_report(findings, repo_info_data)
        else:
            # For other report types, you would call specific generators
            report = {"type": report_type, "data": findings}
        
        # Format output
        if output_format == 'json':
            output = json.dumps(report, indent=2)
        elif output_format == 'yaml':
            import yaml
            output = yaml.dump(report, default_flow_style=False)
        else:
            output = str(report)
        
        # Display and save
        click.echo(f"\n📋 {report_type.title()} Report:")
        click.echo(output)
        
        filename = f"{report_type}_report.{output_format}"
        with open(filename, 'w') as f:
            f.write(output)
        click.echo(f"\n💾 Report saved to {filename}")
        
    except Exception as e:
        click.echo(f"❌ Report generation failed: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def ci():
    """CI/CD integration commands"""
    pass

@ci.command()
@click.option('--repo-path', required=True, help='Path to local repository')
@click.option('--platform',
              type=click.Choice(['github_actions', 'gitlab_ci', 'jenkins', 'circleci', 'azure_pipelines']),
              required=True, help='CI/CD platform')
@click.option('--config-file', help='Path to configuration file')
async def setup(repo_path: str, platform: str, config_file: Optional[str]):
    """Set up CI/CD integration for a repository"""
    click.echo(f"🔧 Setting up {platform} integration for {repo_path}")
    
    try:
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "analysis_type": "comprehensive",
                "enable_security_analysis": True,
                "enable_performance_analysis": True,
                "enable_api_analysis": True,
                "severity_threshold": "medium"
            }
        
        # Create CI/CD configuration
        config_file_path = await create_ci_config_for_platform(Path(repo_path), platform, config)
        click.echo(f"✅ {platform} configuration created: {config_file_path}")
        
    except Exception as e:
        click.echo(f"❌ CI/CD setup failed: {str(e)}", err=True)
        sys.exit(1)

@ci.command()
@click.option('--repo-path', required=True, help='Path to local repository')
async def detect(repo_path: str):
    """Detect existing CI/CD configuration"""
    click.echo(f"🔍 Detecting CI/CD configuration in {repo_path}")
    
    try:
        ci_info = await detect_ci_platform(Path(repo_path))
        click.echo("CI/CD Platform Information:")
        click.echo(json.dumps(ci_info, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Detection failed: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def github():
    """GitHub integration commands"""
    pass

@github.command()
@click.option('--repo-url', required=True, help='Repository URL')
@click.option('--pr-number', required=True, type=int, help='Pull request number')
@click.option('--findings-file', required=True, help='Path to findings JSON file')
async def comment(repo_url: str, pr_number: int, findings_file: str):
    """Post findings as comments to a GitHub PR"""
    click.echo(f"💬 Posting comments to PR #{pr_number} in {repo_url}")
    
    try:
        with open(findings_file, 'r') as f:
            findings = json.load(f)
        
        results = await post_findings_to_pr(repo_url, pr_number, findings)
        click.echo(f"✅ Posted {results.get('comments_posted', 0)} comments")
        
    except Exception as e:
        click.echo(f"❌ Failed to post comments: {str(e)}", err=True)
        sys.exit(1)

@github.command()
@click.option('--repo-url', required=True, help='Repository URL')
@click.option('--pr-number', required=True, type=int, help='Pull request number')
@click.option('--findings-file', required=True, help='Path to findings JSON file')
async def review(repo_url: str, pr_number: int, findings_file: str):
    """Create a comprehensive PR review on GitHub"""
    click.echo(f"🔍 Creating PR review for #{pr_number} in {repo_url}")
    
    try:
        with open(findings_file, 'r') as f:
            findings = json.load(f)
        
        results = await create_pr_review(repo_url, pr_number, findings)
        click.echo("✅ PR review created successfully")
        
    except Exception as e:
        click.echo(f"❌ Failed to create review: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def notify():
    """Notification commands"""
    pass

@notify.command()
async def test():
    """Test all notification channels"""
    click.echo("🔔 Testing notification system...")
    
    try:
        results = await test_notification_system()
        click.echo("Notification Test Results:")
        click.echo(json.dumps(results, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Notification test failed: {str(e)}", err=True)
        sys.exit(1)

@notify.command()
@click.option('--finding-file', required=True, help='Path to finding JSON file')
@click.option('--repo-info-file', required=True, help='Path to repository info JSON file')
async def critical(finding_file: str, repo_info_file: str):
    """Send critical finding notification"""
    click.echo("🚨 Sending critical finding notification...")
    
    try:
        with open(finding_file, 'r') as f:
            finding = json.load(f)
        
        with open(repo_info_file, 'r') as f:
            repo_info = json.load(f)
        
        results = await send_critical_finding_notification(finding, repo_info)
        click.echo("Notification Results:")
        click.echo(json.dumps(results, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Notification failed: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def config():
    """Configuration commands"""
    pass

@config.command()
@click.option('--preset',
              type=click.Choice(['basic', 'standard', 'advanced', 'enterprise']),
              required=True, help='Configuration preset')
@click.option('--output-file', help='Output file path')
async def preset(preset: str, output_file: Optional[str]):
    """Generate configuration preset"""
    click.echo(f"⚙️ Generating {preset} configuration preset...")
    
    try:
        config = get_preset_config(preset)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config.dict(), f, indent=2)
            click.echo(f"✅ Configuration saved to {output_file}")
        else:
            click.echo("Configuration:")
            click.echo(json.dumps(config.dict(), indent=2))
        
    except Exception as e:
        click.echo(f"❌ Configuration generation failed: {str(e)}", err=True)
        sys.exit(1)

@config.command()
@click.option('--config-file', required=True, help='Path to configuration file')
async def validate(config_file: str):
    """Validate configuration file"""
    click.echo(f"✅ Validating configuration file: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        config = import_config(json.dumps(config_data), "json")
        issues = validate_config(config)
        
        if issues:
            click.echo("❌ Configuration validation failed:")
            for issue in issues:
                click.echo(f"  - {issue}")
            sys.exit(1)
        else:
            click.echo("✅ Configuration is valid!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}", err=True)
        sys.exit(1)

@config.command()
@click.option('--config-file', required=True, help='Path to configuration file')
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json')
async def export(config_file: str, format: str):
    """Export configuration in different formats"""
    click.echo(f"📤 Exporting configuration in {format} format...")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        config = import_config(json.dumps(config_data), "json")
        exported = export_config(config, format)
        
        output_file = f"config_export.{format}"
        with open(output_file, 'w') as f:
            f.write(exported)
        
        click.echo(f"✅ Configuration exported to {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Export failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
async def version():
    """Show version information"""
    click.echo("Enhanced Code Review Agent v3.0.0")
    click.echo("Enterprise-Grade Code Analysis Platform")
    click.echo("Features: ML Analysis, CI/CD Integration, Advanced Reporting, Notifications")

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
    
    if stats['repos']:
        click.echo("\n📁 Repository Details:")
        for repo in stats['repos']:
            click.echo(f"   - {repo['job_id']}: {repo['size_human']} (age: {repo['age_hours']:.1f}h)")
            if 'repo_url' in repo:
                click.echo(f"     URL: {repo['repo_url']}")
                if 'branch' in repo:
                    click.echo(f"     Branch: {repo['branch']}")
    else:
        click.echo("   No repositories found")

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

if __name__ == "__main__":
    # Handle asyncio for async commands
    if len(sys.argv) > 1 and sys.argv[1] in ['analyze', 'ml', 'report', 'ci', 'github', 'notify']:
        # For async commands, we need to run the event loop
        asyncio.run(main())
    else:
        main()
