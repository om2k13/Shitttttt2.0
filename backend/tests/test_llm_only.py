#!/usr/bin/env python3
"""
Simple LLM Test - Tests only the Local LLM functionality
No code review analysis, just LLM enhancement
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(__file__))

async def test_llm_only():
    """Test only the Local LLM functionality"""
    
    print("🧪 Testing Local LLM Only...")
    print("=" * 40)
    
    try:
        # Test 1: Import Local LLM Enhancer
        print("1️⃣ Testing Local LLM Enhancer import...")
        from app.review.local_llm_enhancer import LocalLLMEnhancer
        
        print("✅ Local LLM Enhancer imported successfully")
        
        # Test 2: Initialize Local LLM Enhancer
        print("2️⃣ Testing Local LLM Enhancer initialization...")
        enhancer = LocalLLMEnhancer("qwen2.5-coder:7b")
        
        print("✅ Local LLM Enhancer initialized successfully")
        
        # Test 3: Test simple code enhancement
        print("3️⃣ Testing simple code enhancement...")
        
        test_code = """def vulnerable_function(user_input):
    return execute_query('SELECT * FROM users WHERE id = ' + user_input)"""
        
        test_finding = {
            "message": "SQL injection vulnerability",
            "severity": "HIGH",
            "category": "security",
            "file": "test.py",
            "line": 1
        }
        
        print("   Sending request to Qwen2.5-coder:7b...")
        enhanced = await enhancer.enhance_finding(test_code, test_finding)
        
        if enhanced.get("llm_enhanced"):
            print("✅ Code enhancement successful!")
            print(f"AI Explanation: {enhanced.get('ai_explanation', 'No explanation')[:300]}...")
            print(f"Latency: {enhanced.get('latency_ms', 0):.0f}ms")
            print(f"Source: {enhanced.get('source', 'Unknown')}")
        else:
            print(f"❌ Code enhancement failed: {enhanced.get('ai_explanation', 'Unknown error')}")
            return False
        
        # Test 4: Test performance stats
        print("4️⃣ Testing performance statistics...")
        stats = enhancer.get_performance_stats()
        print(f"Performance: {stats}")
        
        # Test 5: Test model status
        print("5️⃣ Testing model status...")
        status = enhancer.get_model_status()
        print(f"Model Status: {status}")
        
        # Cleanup
        enhancer.cleanup()
        print("✅ Local LLM Enhancer cleanup successful")
        
        print("=" * 40)
        print("🎉 Local LLM test completed successfully!")
        print("✅ Your Qwen2.5-coder:7b model is working!")
        print("✅ Ready for integration with code review agent!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_only())
    sys.exit(0 if success else 1)
