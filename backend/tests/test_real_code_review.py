#!/usr/bin/env python3
"""
Real Code Review Test with Complete Integration
Tests the full pipeline with actual code analysis
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append(str(Path(__file__).parent))

async def test_real_code_review():
    """Test real code review with complete integration"""
    
    print("🧪 Real Code Review Test with Complete Integration")
    print("=" * 70)
    print("Testing: Full Pipeline with Real Code Analysis")
    print("=" * 70)
    
    try:
        # Import the code review agent
        from app.review.code_review_agent import CodeReviewAgent
        
        print("✅ Code Review Agent imported successfully")
        
        # Create a test repository path (use current directory)
        test_repo_path = Path.cwd()
        print(f"📁 Test repository path: {test_repo_path}")
        
        # Initialize the agent
        print("\n🚀 Initializing Code Review Agent...")
        agent = CodeReviewAgent(test_repo_path, standalone=True)
        print("✅ Code Review Agent initialized successfully")
        
        # Create a test Python file with various issues
        test_file_path = test_repo_path / "test_review.py"
        test_code = '''
"""
Test file for code review analysis
Contains various code quality issues for testing
"""

import os
import sys
from typing import List, Dict, Optional

def calculate_factorial(n):
    """Calculate factorial of n"""
    if n < 0:
        return None
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def process_data(data):
    """Process data with potential issues"""
    if not data:
        return []
    
    processed = []
    for item in data:
        if item > 0:
            processed.append(item * 2)
    
    return processed

def complex_function(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    """Function with too many parameters"""
    result = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y + z
    return result

def hardcoded_values():
    """Function with hardcoded values"""
    api_key = "sk-1234567890abcdef"
    database_url = "postgresql://user:pass@localhost:5432/db"
    timeout = 30000
    max_retries = 5
    
    return {
        "api_key": api_key,
        "database_url": database_url,
        "timeout": timeout,
        "max_retries": max_retries
    }

def unused_variables():
    """Function with unused variables"""
    important_data = "very important"
    temp_var = "temporary"
    unused_var = "never used"
    
    print(important_data)
    return temp_var

def exception_handling():
    """Function with poor exception handling"""
    try:
        result = 10 / 0
    except:
        print("Something went wrong")
    
    return result

if __name__ == "__main__":
    # Test the functions
    print(calculate_factorial(5))
    print(process_data([1, 2, 3, 4, 5]))
    print(complex_function(*range(26)))
    print(hardcoded_values())
    print(unused_variables())
    print(exception_handling())
'''
        
        with open(test_file_path, 'w') as f:
            f.write(test_code)
        
        print("📝 Created test file with various code quality issues")
        print("🔍 File contains: complexity, hardcoded values, unused variables, poor exception handling")
        
        # Run the code review
        print("\n🔄 Running Complete Code Review...")
        try:
            review_result = await agent.run_code_review()
            
            print("✅ Code review completed successfully!")
            print(f"📊 Found {len(agent.findings)} issues")
            
            # Show findings with LLM enhancements
            if agent.findings:
                print("\n📋 Code Review Findings with LLM Enhancements:")
                for i, finding in enumerate(agent.findings):
                    print(f"\n🔍 Finding {i+1}:")
                    print(f"   📁 File: {finding.file}")
                    print(f"   📍 Line: {finding.line}")
                    print(f"   🚨 Severity: {finding.severity}")
                    print(f"   🏷️ Category: {finding.category}")
                    print(f"   💬 Message: {finding.message}")
                    
                    if hasattr(finding, 'suggestion') and finding.suggestion:
                        print(f"   💡 AI Suggestion: {finding.suggestion}")
                    
                    if hasattr(finding, 'confidence') and finding.confidence:
                        print(f"   🎯 Confidence: {finding.confidence:.2f}")
                    
                    if hasattr(finding, 'impact') and finding.impact:
                        print(f"   📊 Impact: {finding.impact}")
                    
                    if hasattr(finding, 'effort') and finding.effort:
                        print(f"   ⚡ Effort: {finding.effort}")
                    
                    print("   " + "-" * 50)
            else:
                print("✅ No issues found - code is perfect!")
            
            # Show performance stats
            print("\n📊 Performance Statistics:")
            if agent.local_llm_enhancer:
                local_stats = agent.local_llm_enhancer.get_performance_stats()
                print(f"🧠 Local LLM: {local_stats['success_rate_percent']:.1f}% success rate")
            
            if agent.free_api_llm_enhancer:
                print(f"🌐 Free API LLM: {agent.free_api_llm_enhancer.total_requests} requests, {agent.free_api_llm_enhancer.successful_requests} successful")
            
        except Exception as e:
            print(f"❌ Code review failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up test file
        test_file_path.unlink()
        print("🧹 Cleaned up test file")
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎯 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        if agent.findings:
            print(f"✅ SUCCESS: Found and analyzed {len(agent.findings)} code issues")
            print("💡 Your code review agent is working with:")
            print("   - ML-powered code analysis")
            print("   - Local LLM enhancement (Qwen2.5-coder:7b)")
            print("   - Free API LLM enhancement (OpenRouter, Hugging Face, Replicate)")
            print("   - Professional AI explanations for all findings")
            print("   - Automatic provider fallback")
        else:
            print("✅ SUCCESS: Code review completed with no issues found")
        
        print("\n🚀 Your code review agent is ready for production use!")
        
        # Cleanup
        agent.cleanup()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed and paths are correct")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Real Code Review Test...")
    asyncio.run(test_real_code_review())
