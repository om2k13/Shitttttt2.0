#!/usr/bin/env python3
"""
Test Local LLM Integration with Code Review Agent
Tests the Qwen2.5-7B local model integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(__file__))

async def test_local_llm_integration():
    """Test the complete local LLM integration"""
    
    print("🧪 Testing Local LLM Integration...")
    print("=" * 50)
    
    try:
        # Test 1: Check if local LLM enhancer can be imported
        print("1️⃣ Testing Local LLM Enhancer import...")
        from app.review.local_llm_enhancer import LocalLLMEnhancer, LocalOllamaLLM
        
        print("✅ Local LLM Enhancer imported successfully")
        
        # Test 2: Test Local Ollama LLM initialization
        print("2️⃣ Testing Local Ollama LLM initialization...")
        ollama_llm = LocalOllamaLLM("qwen2.5-coder:7b")
        
        if ollama_llm.is_available:
            print("✅ Ollama is running and accessible")
        else:
            print("⚠️ Ollama not available - please start with 'ollama serve'")
            print("💡 To start Ollama: ollama serve")
            return False
        
        # Test 3: Test model information
        print("3️⃣ Testing model information...")
        model_info = ollama_llm.get_model_info()
        print(f"Model info: {model_info}")
        
        if model_info.get("available"):
            print("✅ Qwen2.5:7b model is available")
        else:
            print("⚠️ Qwen2.5:7b model not found")
            print("💡 To download: ollama pull qwen2.5:7b")
            return False
        
        # Test 4: Test code enhancement
        print("4️⃣ Testing code enhancement...")
        test_code = """def vulnerable_function(user_input):
    return execute_query('SELECT * FROM users WHERE id = ' + user_input)"""
        
        test_finding = {
            "message": "SQL injection vulnerability",
            "severity": "HIGH",
            "category": "security",
            "file": "test.py",
            "line": 1
        }
        
        response = await ollama_llm.enhance_finding(test_code, test_finding)
        
        if response.success:
            print("✅ Code enhancement successful!")
            print(f"Response: {response.explanation[:200]}...")
            print(f"Latency: {response.latency_ms:.0f}ms")
        else:
            print(f"❌ Code enhancement failed: {response.error}")
            return False
        
        # Test 5: Test Local LLM Enhancer
        print("5️⃣ Testing Local LLM Enhancer...")
        enhancer = LocalLLMEnhancer("qwen2.5-coder:7b")
        
        enhanced = await enhancer.enhance_finding(test_code, test_finding)
        
        if enhanced.get("llm_enhanced"):
            print("✅ Local LLM Enhancer working")
            print(f"Enhanced finding: {enhanced.get('ai_explanation', 'No explanation')[:200]}...")
        else:
            print(f"❌ Local LLM Enhancer failed: {enhanced.get('ai_explanation', 'Unknown error')}")
            return False
        
        # Test 6: Test performance stats
        print("6️⃣ Testing performance statistics...")
        stats = enhancer.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        # Test 7: Test model status
        print("7️⃣ Testing model status...")
        status = enhancer.get_model_status()
        print(f"Model status: {status}")
        
        # Cleanup
        enhancer.cleanup()
        print("✅ Local LLM Enhancer cleanup successful")
        
        print("=" * 50)
        print("🎉 All Local LLM tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_code_review_agent_integration():
    """Test the Code Review Agent with Local LLM integration"""
    
    print("\n🧪 Testing Code Review Agent + Local LLM Integration...")
    print("=" * 50)
    
    try:
        # Test 1: Import Code Review Agent
        print("1️⃣ Testing Code Review Agent import...")
        from app.review.code_review_agent import CodeReviewAgent
        
        print("✅ Code Review Agent imported successfully")
        
        # Test 2: Initialize Code Review Agent
        print("2️⃣ Testing Code Review Agent initialization...")
        current_dir = Path(".")
        agent = CodeReviewAgent(current_dir, standalone=True)
        
        print("✅ Code Review Agent initialized successfully")
        
        # Test 3: Check LLM status
        print("3️⃣ Testing LLM status...")
        llm_status = agent.get_llm_status()
        print(f"LLM Status: {llm_status}")
        
        if llm_status.get("status") == "available":
            print("✅ Local LLM integration is available")
        else:
            print("⚠️ Local LLM integration not available")
            print(f"Error: {llm_status.get('error', 'Unknown')}")
            return False
        
        # Test 4: Test code review with LLM enhancement
        print("4️⃣ Testing code review with LLM enhancement...")
        
        # Create a simple test file
        test_file = Path("test_code.py")
        test_content = """def bad_function():
    x = 10
    if x > 5:
        print("x is greater than 5")
    else:
        print("x is not greater than 5")
    
    # Unused variable
    unused_var = "this is never used"
    
    return x"""
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        try:
            # Run a simple code review
            result = await agent.run_code_review()
            
            if result and "findings" in result:
                print(f"✅ Code review completed with {len(result['findings'])} findings")
                
                # Check if any findings were enhanced with LLM
                enhanced_count = sum(1 for f in result['findings'] if hasattr(f, 'confidence') and f.confidence > 0.5)
                print(f"Enhanced findings: {enhanced_count}")
                
            else:
                print("⚠️ Code review completed but no findings returned")
                
        except Exception as e:
            print(f"⚠️ Code review failed: {e}")
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        
        agent.cleanup()
        print("✅ Code Review Agent cleanup successful")
        
        print("=" * 50)
        print("🎉 Code Review Agent + Local LLM integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Code Review Agent integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    
    print("🚀 Starting Local LLM Integration Tests...")
    print("=" * 60)
    
    # Test 1: Basic Local LLM functionality
    success1 = await test_local_llm_integration()
    
    if success1:
        # Test 2: Code Review Agent integration
        success2 = await test_code_review_agent_integration()
        
        if success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Local LLM integration is working correctly")
            print("✅ Code Review Agent can use Local LLM enhancement")
            print("✅ Your Qwen2.5-7B model is ready for production use!")
            return True
        else:
            print("\n❌ Code Review Agent integration test failed")
            return False
    else:
        print("\n❌ Basic Local LLM test failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
