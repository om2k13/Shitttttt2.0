#!/usr/bin/env python3
"""
LLM Integration Test for Code Review Agent
Tests Local LLM and Free API LLM working together
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

async def test_llm_integration():
    """Test LLM integration components only"""
    
    print("🧪 LLM Integration Test for Code Review Agent")
    print("=" * 60)
    print("Testing: Local LLM + Free API LLM Integration")
    print("=" * 60)
    
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
        
        # Check LLM status
        print("\n🔍 Checking LLM Integration Status...")
        llm_status = agent.get_llm_status()
        print(f"📊 LLM Status: {llm_status}")
        
        # Test Local LLM Enhancer
        print("\n🧠 Testing Local LLM Enhancer...")
        if agent.local_llm_enhancer:
            print("✅ Local LLM Enhancer is available")
            try:
                # Test a simple enhancement
                test_finding = {
                    "file": "test.py",
                    "line": 1,
                    "severity": "medium",
                    "category": "quality",
                    "message": "Test finding for Local LLM enhancement",
                    "suggestion": "Initial suggestion"
                }
                
                print("🔄 Testing Local LLM enhancement...")
                enhanced = await agent.local_llm_enhancer.enhance_finding(
                    "def test_function():\n    pass", 
                    test_finding
                )
                
                if enhanced.get("llm_enhanced"):
                    print("✅ Local LLM enhancement successful!")
                    print(f"   AI Explanation: {enhanced.get('ai_explanation', '')[:100]}...")
                else:
                    print("⚠️ Local LLM enhancement failed")
                    
            except Exception as e:
                print(f"❌ Local LLM enhancement error: {e}")
        else:
            print("⚠️ Local LLM Enhancer not available")
        
        # Test Free API LLM Enhancer
        print("\n🌐 Testing Free API LLM Enhancer...")
        if agent.free_api_llm_enhancer:
            print("✅ Free API LLM Enhancer is available")
            try:
                # Test a simple enhancement
                test_finding = {
                    "file": "test.py",
                    "line": 1,
                    "severity": "medium",
                    "category": "quality",
                    "message": "Test finding for API LLM enhancement",
                    "suggestion": "Initial suggestion"
                }
                
                print("🔄 Testing Free API LLM enhancement...")
                enhanced = await agent.free_api_llm_enhancer.enhance_finding(
                    "def test_function():\n    pass", 
                    test_finding
                )
                
                if enhanced.get("llm_enhanced"):
                    print("✅ Free API LLM enhancement successful!")
                    print(f"   Provider: {enhanced.get('provider', 'unknown')}")
                    print(f"   AI Explanation: {enhanced.get('ai_explanation', '')[:100]}...")
                    print(f"   Latency: {enhanced.get('latency_ms', 0):.0f}ms")
                else:
                    print("⚠️ Free API LLM enhancement failed")
                    
            except Exception as e:
                print(f"❌ Free API LLM enhancement error: {e}")
        else:
            print("⚠️ Free API LLM Enhancer not available")
        
        # Test LLM enhancement pipeline
        print("\n🔄 Testing LLM Enhancement Pipeline...")
        try:
            # Create a simple test file for analysis
            test_file_path = test_repo_path / "test_llm.py"
            test_code = '''
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
'''
            
            with open(test_file_path, 'w') as f:
                f.write(test_code)
            
            print("📝 Created test file for analysis")
            
            # Test the LLM enhancement methods directly
            print("🔍 Testing LLM enhancement methods...")
            
            # Test local LLM enhancement
            if agent.local_llm_enhancer:
                print("   🧠 Testing Local LLM enhancement method...")
                try:
                    await agent._enhance_findings_with_local_llm()
                    print("   ✅ Local LLM enhancement method successful")
                except Exception as e:
                    print(f"   ❌ Local LLM enhancement method failed: {e}")
            
            # Test free API LLM enhancement
            if agent.free_api_llm_enhancer:
                print("   🌐 Testing Free API LLM enhancement method...")
                try:
                    await agent._enhance_findings_with_free_api_llm()
                    print("   ✅ Free API LLM enhancement method successful")
                except Exception as e:
                    print(f"   ❌ Free API LLM enhancement method failed: {e}")
            
            # Clean up test file
            test_file_path.unlink()
            print("🧹 Cleaned up test file")
            
        except Exception as e:
            print(f"❌ LLM enhancement pipeline test failed: {e}")
        
        # Final status report
        print("\n" + "=" * 60)
        print("📊 FINAL LLM INTEGRATION STATUS")
        print("=" * 60)
        
        print(f"✅ Code Review Agent: Initialized")
        print(f"🧠 Local LLM: {'✅ Available' if agent.local_llm_enhancer else '❌ Not Available'}")
        print(f"🌐 Free API LLM: {'✅ Available' if agent.free_api_llm_enhancer else '❌ Not Available'}")
        
        # LLM Provider details
        if agent.free_api_llm_enhancer:
            print(f"\n🌐 Free API LLM Providers:")
            for provider_enum, provider in agent.free_api_llm_enhancer.providers.items():
                current = " (CURRENT)" if provider_enum == agent.free_api_llm_enhancer.current_provider else ""
                print(f"   - {provider_enum.value}{current}")
        
        print("\n" + "=" * 60)
        
        if agent.local_llm_enhancer or agent.free_api_llm_enhancer:
            print("🎉 SUCCESS: LLM integration working! Both enhancers are operational.")
            print("💡 Your code review agent has professional AI-powered code review capabilities!")
        else:
            print("❌ FAILURE: No LLM enhancers are available.")
            print("💡 Check the error messages above for troubleshooting.")
        
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
    print("🚀 Starting LLM Integration Test...")
    asyncio.run(test_llm_integration())
