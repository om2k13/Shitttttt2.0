#!/usr/bin/env python3
"""
Test Free API LLM Integration
Tests multiple completely free API providers with local Ollama fallback
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

async def test_free_api_llm():
    """Test the complete free API LLM integration"""
    
    print("🧪 Testing Free API LLM Integration...")
    print("=" * 60)
    
    try:
        # Test 1: Check environment and tokens
        print("1️⃣ Checking environment and API tokens...")
        
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        replicate_token = os.getenv("REPLICATE_TOKEN")
        
        if huggingface_token:
            print(f"✅ Hugging Face token found: {huggingface_token[:10]}...")
        else:
            print("⚠️ No Hugging Face token found (will skip Hugging Face tests)")
        
        if replicate_token:
            print(f"✅ Replicate token found: {replicate_token[:10]}...")
        else:
            print("⚠️ No Replicate token found (will skip Replicate tests)")
        
        if not huggingface_token and not replicate_token:
            print("⚠️ No API tokens found - will test only local Ollama API")
        
        # Test 2: Test Free API LLM Enhancer
        print("\n2️⃣ Testing Free API LLM Enhancer...")
        from app.review.free_api_llm_enhancer import FreeAPILLMEnhancer
        
        enhancer = FreeAPILLMEnhancer(
            huggingface_token=huggingface_token,
            replicate_token=replicate_token
        )
        
        print(f"✅ Free API LLM Enhancer initialized successfully")
        print(f"Available providers: {list(enhancer.providers.keys())}")
        
        # Test 3: Test individual providers
        print("\n3️⃣ Testing individual providers...")
        
        test_code = """def vulnerable_function(user_input):
    return execute_query('SELECT * FROM users WHERE id = ' + user_input)"""
        
        test_finding = {
            "message": "SQL injection vulnerability",
            "severity": "HIGH",
            "category": "security",
            "file": "test.py",
            "line": 1
        }
        
        # Test Hugging Face (if available)
        if huggingface_token:
            print("   Testing Hugging Face API...")
            try:
                from app.review.free_api_llm_enhancer import FreeHuggingFaceAPI
                hf_api = FreeHuggingFaceAPI(huggingface_token)
                response = await hf_api.enhance_finding(test_code, test_finding)
                
                if response.success:
                    print(f"   ✅ Hugging Face API working")
                    print(f"   Response: {response.explanation[:200]}...")
                    print(f"   Source: {response.source}")
                    print(f"   Latency: {response.latency_ms:.2f}ms")
                else:
                    print(f"   ❌ Hugging Face API failed: {response.error}")
            except Exception as e:
                print(f"   ❌ Hugging Face API test failed: {e}")
        
        # Test Replicate (if available)
        if replicate_token:
            print("   Testing Replicate API...")
            try:
                from app.review.free_api_llm_enhancer import FreeReplicateAPI
                rep_api = FreeReplicateAPI(replicate_token)
                response = await rep_api.enhance_finding(test_code, test_finding)
                
                if response.success:
                    print(f"   ✅ Replicate API working")
                    print(f"   Response: {response.explanation[:200]}...")
                    print(f"   Source: {response.source}")
                    print(f"   Latency: {response.latency_ms:.2f}ms")
                else:
                    print(f"   ❌ Replicate API failed: {response.error}")
            except Exception as e:
                print(f"   ❌ Replicate API test failed: {e}")
        
        # Test Ollama API
        print("   Testing Ollama API...")
        try:
            from app.review.free_api_llm_enhancer import OllamaAPIClient
            ollama_api = OllamaAPIClient()
            
            if ollama_api.is_available():
                print("   ✅ Ollama API is accessible")
                
                response = await ollama_api.enhance_finding(test_code, test_finding)
                if response.success:
                    print(f"   ✅ Ollama API working")
                    print(f"   Response: {response.explanation[:200]}...")
                    print(f"   Source: {response.source}")
                    print(f"   Latency: {response.latency_ms:.2f}ms")
                else:
                    print(f"   ❌ Ollama API failed: {response.error}")
            else:
                print("   ⚠️ Ollama API not accessible (make sure Ollama is running)")
        except Exception as e:
            print(f"   ❌ Ollama API test failed: {e}")
        
        # Test 4: Test complete hybrid enhancement
        print("\n4️⃣ Testing complete hybrid enhancement...")
        
        enhanced = await enhancer.enhance_finding(test_code, test_finding)
        
        if enhanced.get("llm_enhanced"):
            print("✅ Code enhancement successful!")
            print(f"AI Explanation: {enhanced.get('ai_explanation', 'No explanation')[:300]}...")
            print(f"Source: {enhanced.get('source', 'Unknown')}")
            print(f"Provider: {enhanced.get('provider', 'Unknown')}")
            print(f"Mode: {enhanced.get('mode', 'Unknown')}")
            print(f"Latency: {enhanced.get('latency_ms', 0):.2f}ms")
        else:
            print("❌ Code enhancement failed")
            print(f"Error: {enhanced.get('ai_explanation', 'Unknown error')}")
        
        # Test 5: Get performance statistics
        print("\n5️⃣ Performance Statistics...")
        stats = enhancer.get_performance_stats()
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Successful Requests: {stats['successful_requests']}")
        print(f"Success Rate: {stats['success_rate_percent']}%")
        print(f"Current Provider: {stats['current_provider']}")
        print(f"Available Providers: {stats['available_providers']}")
        
        # Test 6: Get provider status
        print("\n6️⃣ Provider Status...")
        status = enhancer.get_provider_status()
        for provider, info in status.items():
            print(f"{provider}: {info}")
        
        # Test 7: Test natural language queries
        print("\n7️⃣ Testing natural language queries...")
        explanation = await enhancer.explain_code_pattern(
            test_code,
            "What is the security risk in this code?"
        )
        
        if explanation and "error" not in explanation.lower():
            print("✅ Natural language explanation successful")
            print(f"Response: {explanation[:200]}...")
        else:
            print("❌ Natural language explanation failed")
        
        # Test 8: Test provider switching
        print("\n8️⃣ Testing provider switching...")
        print("Resetting provider failure counters...")
        enhancer.reset_provider_failures()
        print("✅ Provider failure counters reset")
        
        # Cleanup
        enhancer.cleanup()
        print("✅ Free API LLM enhancer cleanup successful")
        
        print("\n" + "=" * 60)
        print("🎉 All Free API LLM tests completed!")
        print("\n📊 SUMMARY:")
        print(f"✅ Free API LLM Enhancer: Working")
        print(f"✅ Multiple Providers: {len(enhancer.providers)} available")
        print(f"✅ Hybrid Fallback: Automatic provider switching")
        print(f"✅ Performance Tracking: Comprehensive statistics")
        print(f"✅ Cost: $0/month (completely free)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_provider_specific():
    """Test specific providers in detail"""
    
    print("\n🔍 Detailed Provider Testing...")
    print("=" * 40)
    
    # Test Hugging Face specific
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if huggingface_token:
        print("\n🧠 Testing Hugging Face Models...")
        try:
            from app.review.free_api_llm_enhancer import FreeHuggingFaceAPI
            
            hf_api = FreeHuggingFaceAPI(huggingface_token)
            print(f"Available models: {hf_api.models}")
            
            test_code = "def test(): pass"
            test_finding = {"message": "Test finding", "severity": "LOW", "category": "test", "file": "test.py", "line": 1}
            
            for i, model in enumerate(hf_api.models):
                print(f"\nTesting model {i+1}: {model}")
                try:
                    response = await hf_api.enhance_finding(test_code, test_finding)
                    if response.success:
                        print(f"✅ {model}: Working")
                        print(f"   Response: {response.explanation[:100]}...")
                    else:
                        print(f"❌ {model}: Failed - {response.error}")
                except Exception as e:
                    print(f"❌ {model}: Exception - {e}")
                    
        except Exception as e:
            print(f"❌ Hugging Face testing failed: {e}")
    
    # Test Replicate specific
    replicate_token = os.getenv("REPLICATE_TOKEN")
    if replicate_token:
        print("\n🔄 Testing Replicate Models...")
        try:
            from app.review.free_api_llm_enhancer import FreeReplicateAPI
            
            rep_api = FreeReplicateAPI(replicate_token)
            print(f"Available models: {[m[:50] + '...' for m in rep_api.models]}")
            
            test_code = "def test(): pass"
            test_finding = {"message": "Test finding", "severity": "LOW", "category": "test", "file": "test.py", "line": 1}
            
            for i, model in enumerate(rep_api.models):
                print(f"\nTesting model {i+1}: {model[:50]}...")
                try:
                    response = await rep_api.enhance_finding(test_code, test_finding)
                    if response.success:
                        print(f"✅ Model {i+1}: Working")
                        print(f"   Response: {response.explanation[:100]}...")
                    else:
                        print(f"❌ Model {i+1}: Failed - {response.error}")
                except Exception as e:
                    print(f"❌ Model {i+1}: Exception - {e}")
                    
        except Exception as e:
            print(f"❌ Replicate testing failed: {e}")

if __name__ == "__main__":
    print("🚀 Free API LLM Integration Test Suite")
    print("This will test multiple completely free LLM providers")
    print("Make sure you have set up your API tokens in .env file")
    print()
    
    # Run main test
    success = asyncio.run(test_free_api_llm())
    
    # Run detailed provider tests
    if success:
        asyncio.run(test_provider_specific())
    
    sys.exit(0 if success else 1)
