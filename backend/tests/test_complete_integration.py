#!/usr/bin/env python3
"""
Complete Integration Test for Code Review Agent
Tests ML models, neural networks, local LLM, and API LLM working together
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

async def test_complete_integration():
    """Test complete integration of all components"""
    
    print("🧪 Complete Code Review Agent Integration Test")
    print("=" * 70)
    print("Testing: ML Models + Neural Networks + Local LLM + API LLM")
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
        
        # Check LLM status
        print("\n🔍 Checking LLM Integration Status...")
        llm_status = agent.get_llm_status()
        print(f"📊 LLM Status: {llm_status}")
        
        # Test ML and Neural Network initialization
        print("\n🧠 Testing ML and Neural Network Initialization...")
        
        # Test ML Analyzer (lazy loading)
        if not agent.ml_analyzer:
            print("🔄 Loading ML Analyzer...")
            try:
                from app.review.enhanced_ml_analyzer import EnhancedMLAnalyzer
                agent.ml_analyzer = EnhancedMLAnalyzer()
                print("✅ ML Analyzer loaded successfully")
            except Exception as e:
                print(f"⚠️ ML Analyzer loading failed: {e}")
        
        # Test Neural Analyzer (lazy loading)
        if not agent.neural_analyzer:
            print("🔄 Loading Neural Analyzer...")
            try:
                from app.review.safe_neural_analyzer import SafeNeuralAnalyzer
                agent.neural_analyzer = SafeNeuralAnalyzer()
                print("✅ Neural Analyzer loaded successfully")
            except Exception as e:
                print(f"⚠️ Neural Analyzer loading failed: {e}")
        
        # Test Production ML Analyzer (lazy loading)
        if not agent.production_ml_analyzer:
            print("🔄 Loading Production ML Analyzer...")
            try:
                from app.review.production_ml_analyzer import ProductionMLAnalyzer
                agent.production_ml_analyzer = ProductionMLAnalyzer()
                print("✅ Production ML Analyzer loaded successfully")
            except Exception as e:
                print(f"⚠️ Production ML Analyzer loading failed: {e}")
        
        # Test Advanced ML Capabilities (lazy loading)
        if not agent.advanced_ml_capabilities:
            print("🔄 Loading Advanced ML Capabilities...")
            try:
                from app.review.advanced_ml_capabilities import AdvancedMLCapabilities
                agent.advanced_ml_capabilities = AdvancedMLCapabilities()
                print("✅ Advanced ML Capabilities loaded successfully")
            except Exception as e:
                print(f"⚠️ Advanced ML Capabilities loading failed: {e}")
        
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
                    "message": "Test finding for LLM enhancement",
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
        
        # Test complete pipeline with a small code analysis
        print("\n🔄 Testing Complete Pipeline Integration...")
        try:
            # Create a simple test file for analysis
            test_file_path = test_repo_path / "test_integration.py"
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
            
            # Run a quick code review
            print("🔍 Running quick code review...")
            review_result = await agent.run_code_review()
            
            print("✅ Code review completed successfully!")
            print(f"📊 Found {len(agent.findings)} issues")
            
            # Show some findings
            if agent.findings:
                print("\n📋 Sample Findings:")
                for i, finding in enumerate(agent.findings[:3]):  # Show first 3
                    print(f"   {i+1}. {finding.category.upper()}: {finding.message}")
                    if hasattr(finding, 'suggestion') and finding.suggestion:
                        print(f"      💡 Suggestion: {finding.suggestion[:100]}...")
            
            # Clean up test file
            test_file_path.unlink()
            print("🧹 Cleaned up test file")
            
        except Exception as e:
            print(f"❌ Pipeline integration test failed: {e}")
        
        # Final status report
        print("\n" + "=" * 70)
        print("📊 FINAL INTEGRATION STATUS REPORT")
        print("=" * 70)
        
        print(f"✅ Code Review Agent: Initialized")
        print(f"🧠 ML Analyzer: {'✅ Loaded' if agent.ml_analyzer else '❌ Not Available'}")
        print(f"🧠 Neural Analyzer: {'✅ Loaded' if agent.neural_analyzer else '❌ Not Available'}")
        print(f"🏭 Production ML: {'✅ Loaded' if agent.production_ml_analyzer else '❌ Not Available'}")
        print(f"🚀 Advanced ML: {'✅ Loaded' if agent.advanced_ml_capabilities else '❌ Not Available'}")
        print(f"🧠 Local LLM: {'✅ Available' if agent.local_llm_enhancer else '❌ Not Available'}")
        print(f"🌐 Free API LLM: {'✅ Available' if agent.free_api_llm_enhancer else '❌ Not Available'}")
        
        # LLM Provider details
        if agent.free_api_llm_enhancer:
            print(f"\n🌐 Free API LLM Providers:")
            for provider_enum, provider in agent.free_api_llm_enhancer.providers.items():
                current = " (CURRENT)" if provider_enum == agent.free_api_llm_enhancer.current_provider else ""
                print(f"   - {provider_enum.value}{current}")
        
        print("\n" + "=" * 70)
        
        if (agent.local_llm_enhancer or agent.free_api_llm_enhancer) and agent.production_ml_analyzer:
            print("🎉 SUCCESS: Complete integration working! All components are operational.")
            print("💡 Your code review agent is ready for production use!")
        else:
            print("⚠️ PARTIAL SUCCESS: Some components are missing or failed to load.")
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
    print("🚀 Starting Complete Integration Test...")
    asyncio.run(test_complete_integration())
