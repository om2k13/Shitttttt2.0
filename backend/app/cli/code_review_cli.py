#!/usr/bin/env python3
"""
Code Review Agent CLI

This CLI provides an interface to run the Code Review Agent both as part of the pipeline
and in standalone mode for analyzing code quality, refactoring opportunities, and more.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.review.code_review_agent import CodeReviewAgent
from app.review.enhanced_pipeline import EnhancedPipeline, run_enhanced_review, run_standalone_code_review


def print_banner():
    """Print the Code Review Agent banner"""
    print("""
🔍 Code Review Agent CLI
========================

Advanced code analysis for quality improvements, refactoring opportunities, 
and reusable method suggestions.

This agent works both as part of the pipeline and standalone.
    """)


def print_help():
    """Print detailed help information"""
    print("""
Usage Examples:

1. Standalone Code Review (analyze local repository):
   python -m app.cli.code_review_cli standalone /path/to/repo

2. Enhanced Pipeline Review (with job ID):
   python -m app.cli.code_review_cli pipeline <job_id>

3. Export results to different formats:
   python -m app.cli.code_review_cli standalone /path/to/repo --export-json report.json
   python -m app.cli.code_review_cli standalone /path/to/repo --export-markdown report.md

4. Run with specific analysis options:
   python -m app.cli.code_review_cli standalone /path/to/repo --no-llm --verbose

Available Analysis Categories:
- Code Quality: Style, imports, unused variables, etc.
- Refactoring: Long functions, large classes, nested conditionals
- Reusability: Code duplication, similar logic patterns
- Efficiency: Inefficient patterns, performance improvements
- Configuration: Hardcoded values, environment variables
- Complexity: Cyclomatic complexity analysis

For more information, visit: https://github.com/your-org/code-review-agent
    """)


async def run_standalone_analysis(repo_path: str, options: dict):
    """Run standalone code review analysis"""
    print(f"🔍 Starting standalone Code Review analysis on: {repo_path}")
    
    try:
        # Validate repository path
        repo_path_obj = Path(repo_path)
        if not repo_path_obj.exists():
            print(f"❌ Error: Repository path does not exist: {repo_path}")
            return 1
        
        if not repo_path_obj.is_dir():
            print(f"❌ Error: Path is not a directory: {repo_path}")
            return 1
        
        # Check if it looks like a repository
        if not any(repo_path_obj.glob("*.py")) and not any(repo_path_obj.glob("*.js")) and not any(repo_path_obj.glob("*.ts")):
            print(f"⚠️  Warning: Path doesn't appear to contain common code files: {repo_path}")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
        
        # Initialize and run the Code Review Agent
        agent = CodeReviewAgent(repo_path=repo_path_obj, standalone=True)
        
        print("📊 Running comprehensive code analysis...")
        results = await agent.run_code_review()
        
        if results.get("status") == "error":
            print(f"❌ Analysis failed: {results.get('error')}")
            return 1
        
        # Display results
        await display_results(results, options)
        
        # Export if requested
        if options.get("export_json"):
            await agent.export_findings_to_json(options["export_json"])
        
        if options.get("export_markdown"):
            await agent.export_findings_to_markdown(options["export_markdown"])
        
        print("✅ Standalone analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error during analysis: {str(e)}")
        if options.get("verbose"):
            import traceback
            traceback.print_exc()
        return 1


async def run_enhanced_pipeline(job_id: str):
    """Run enhanced pipeline with code review integration"""
    print(f"🚀 Running Enhanced Pipeline for job: {job_id}")
    
    try:
        from app.review.enhanced_pipeline import run_enhanced_review
        
        # Run the enhanced pipeline
        results = await run_enhanced_review(job_id, include_code_review=True)
        
        if results.get("status") == "error":
            print(f"❌ Pipeline failed: {results.get('error')}")
            return False
        
        # Display results
        print("\n📊 Pipeline Results:")
        print(f"   Status: {results.get('status', 'unknown')}")
        print(f"   Total Findings: {results.get('total_findings', 0)}")
        
        if results.get("findings"):
            print(f"\n🔍 Findings by Stage:")
            stages = {}
            for finding in results["findings"]:
                stage = finding.get("stage", "unknown")
                if stage not in stages:
                    stages[stage] = 0
                stages[stage] += 1
            
            for stage, count in stages.items():
                print(f"   • {stage}: {count}")
        
        if results.get("test_plan"):
            test_plan = results["test_plan"]
            print(f"\n🧪 Test Plan Generated:")
            print(f"   • Total Files: {test_plan.get('total_files', 0)}")
            print(f"   • Test Files: {len(test_plan.get('test_files', []))}")
            print(f"   • Priority Tests: {len(test_plan.get('priority_tests', []))}")
        
        print("\n✅ Enhanced pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running enhanced pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def display_results(results: dict, options: dict):
    """Display analysis results in a user-friendly format"""
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    # Summary statistics
    total_findings = results.get("total_findings", 0)
    print(f"\n🔍 Total Findings: {total_findings}")
    
    if total_findings == 0:
        print("🎉 No issues found! Your code looks great!")
        return
    
    # Findings by category
    if "findings_by_category" in results:
        print("\n📋 Findings by Category:")
        for category, count in results["findings_by_category"].items():
            category_name = category.replace("_", " ").title()
            print(f"   • {category_name}: {count}")
    
    # Findings by severity
    if "findings_by_severity" in results:
        print("\n⚠️  Findings by Severity:")
        severity_order = ["critical", "high", "medium", "low"]
        for severity in severity_order:
            count = results["findings_by_severity"].get(severity, 0)
            if count > 0:
                severity_icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                print(f"   {severity_icon} {severity.title()}: {count}")
    
    # Summary highlights
    if "summary" in results:
        summary = results["summary"]
        print("\n🎯 Key Insights:")
        
        if summary.get("refactoring_opportunities", 0) > 0:
            print(f"   🔄 {summary['refactoring_opportunities']} refactoring opportunities")
        
        if summary.get("reusability_improvements", 0) > 0:
            print(f"   ♻️  {summary['reusability_improvements']} reusability improvements")
        
        if summary.get("efficiency_gains", 0) > 0:
            print(f"   ⚡ {summary['efficiency_gains']} efficiency improvements")
        
        if summary.get("security_issues", 0) > 0:
            print(f"   🔒 {summary['security_issues']} security issues")
        
        if summary.get("dependency_vulnerabilities", 0) > 0:
            print(f"   📦 {summary['dependency_vulnerabilities']} dependency vulnerabilities")
    
    # Detailed findings (if verbose or if there are few findings)
    if options.get("verbose") or total_findings <= 10:
        print("\n📝 Detailed Findings:")
        for i, finding in enumerate(results.get("findings", [])[:20], 1):  # Limit to first 20
            print(f"\n{i}. {finding.get('file', 'Unknown')}:{finding.get('line', 'N/A')}")
            print(f"   Category: {finding.get('category', 'unknown')}")
            print(f"   Severity: {finding.get('severity', 'unknown')}")
            print(f"   Message: {finding.get('message', 'No message')}")
            
            if finding.get('remediation') or finding.get('suggestion'):
                suggestion = finding.get('remediation') or finding.get('suggestion')
                print(f"   💡 Suggestion: {suggestion}")
            
            if finding.get('code_snippet') and options.get("show_code"):
                print(f"   📄 Code:\n```\n{finding['code_snippet']}\n```")
    
    if total_findings > 20 and not options.get("verbose"):
        print(f"\n... and {total_findings - 20} more findings. Use --verbose to see all details.")
    
    print("\n" + "="*60)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Code Review Agent CLI - Advanced code analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s standalone /path/to/repo
  %(prog)s pipeline <job_id>
  %(prog)s standalone /path/to/repo --export-json report.json
  %(prog)s standalone /path/to/repo --verbose --show-code
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Standalone command
    standalone_parser = subparsers.add_parser(
        "standalone",
        help="Run standalone code review analysis on a local repository"
    )
    standalone_parser.add_argument(
        "repo_path",
        help="Path to the repository to analyze"
    )
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run enhanced pipeline analysis with code review (requires job ID)"
    )
    pipeline_parser.add_argument(
        "job_id",
        help="Job ID for pipeline analysis"
    )
    
    # Common options
    for subparser in [standalone_parser, pipeline_parser]:
        subparser.add_argument(
            "--export-json",
            metavar="FILE",
            help="Export results to JSON file"
        )
        subparser.add_argument(
            "--export-markdown",
            metavar="FILE",
            help="Export results to Markdown file"
        )
        subparser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Show detailed output"
        )
        subparser.add_argument(
            "--show-code",
            action="store_true",
            help="Show code snippets in output"
        )
        subparser.add_argument(
            "--no-llm",
            action="store_true",
            help="Disable LLM enrichment (standalone mode only)"
        )
    
    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version="Code Review Agent CLI v1.0.0"
    )
    parser.add_argument(
        "--help-detailed",
        action="store_true",
        help="Show detailed help information"
    )
    
    return parser.parse_args()


async def main():
    """Main CLI entry point"""
    args = parse_arguments()
    
    # Show detailed help if requested
    if args.help_detailed:
        print_banner()
        print_help()
        return 0
    
    # Show banner
    print_banner()
    
    # Validate command
    if not args.command:
        print("❌ Error: No command specified")
        print("Use 'standalone <repo_path>' or 'pipeline <job_id>'")
        print("Use --help for more information")
        return 1
    
    # Prepare options
    options = {
        "verbose": args.verbose,
        "show_code": args.show_code,
        "export_json": args.export_json,
        "export_markdown": args.export_markdown,
        "no_llm": args.no_llm
    }
    
    try:
        if args.command == "standalone":
            return await run_standalone_analysis(args.repo_path, options)
        elif args.command == "pipeline":
            return await run_enhanced_pipeline(args.job_id)
        else:
            print(f"❌ Error: Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  CLI interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
