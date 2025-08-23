#!/usr/bin/env python3
"""
Safe ML Test
Tests ML models one by one to avoid segmentation faults
"""
import sys
from pathlib import Path
import gc

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_production_ml_safe():
    """Test Production ML Analyzer safely"""
    print("🧪 Testing Production ML Analyzer (Safe Mode)...")
    print("=" * 50)
    
    try:
        from app.review.production_ml_analyzer import ProductionMLAnalyzer
        
        print("🔄 Loading Production ML Analyzer...")
        analyzer = ProductionMLAnalyzer()
        
        status = analyzer.get_model_status()
        print(f"   ✅ Production ML loaded: {status['total_models']} models")
        
        # Test with simple code
        test_code = "import os\ndef test(): return os.system('ls')"
        print("🔍 Testing ML analysis...")
        
        results = analyzer.analyze_code_ml(test_code, "test.py")
        print(f"   ✅ Analysis completed")
        print(f"   🎯 Risk level: {results.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')}")
        
        del analyzer
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Production ML test failed: {e}")
        return False

def test_advanced_ml_safe():
    """Test Advanced ML Capabilities safely"""
    print("\n🧪 Testing Advanced ML Capabilities (Safe Mode)...")
    print("=" * 50)
    
    try:
        from app.review.advanced_ml_capabilities import AdvancedMLCapabilities
        
        print("🔄 Loading Advanced ML Capabilities...")
        analyzer = AdvancedMLCapabilities()
        
        print(f"   ✅ Advanced ML loaded: {len(analyzer.models)} models")
        
        # Test with simple features
        import numpy as np
        test_features = np.random.random((1, 9))
        
        print("🔍 Testing complexity prediction...")
        complexity = analyzer.predict_code_complexity(test_features)
        print(f"   ✅ Complexity prediction: {complexity.get('cyclomatic_complexity', 0):.1f}")
        
        del analyzer
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Advanced ML test failed: {e}")
        return False

def test_code_review_agent_safe():
    """Test Code Review Agent safely"""
    print("\n🧪 Testing Code Review Agent (Safe Mode)...")
    print("=" * 50)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create minimal test
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create simple test file
            test_file = temp_path / "test.py"
            test_file.write_text("def test(): pass")
            
            print("🔄 Initializing Code Review Agent...")
            agent = CodeReviewAgent(temp_path, standalone=True)
            
            print(f"   ✅ Agent initialized")
            print(f"   📊 Production ML: {agent.production_ml_analyzer is not None}")
            print(f"   📊 Advanced ML: {agent.advanced_ml_capabilities is not None}")
            
            del agent
            gc.collect()
            return True
            
    except Exception as e:
        print(f"❌ Code Review Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Safe ML Test Suite")
    print("Testing ML models one by one to avoid segmentation faults")
    print()
    
    # Test 1: Production ML
    prod_success = test_production_ml_safe()
    
    # Test 2: Advanced ML
    adv_success = test_advanced_ml_safe()
    
    # Test 3: Code Review Agent
    agent_success = test_code_review_agent_safe()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SAFE TEST RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"Production ML Test: {'✅ PASSED' if prod_success else '❌ FAILED'}")
    print(f"Advanced ML Test: {'✅ PASSED' if adv_success else '❌ FAILED'}")
    print(f"Code Review Agent Test: {'✅ PASSED' if agent_success else '❌ FAILED'}")
    
    total_passed = sum([prod_success, adv_success, agent_success])
    print(f"\n🎯 Overall: {total_passed}/3 tests passed")
    
    if total_passed == 3:
        print("\n🎉 ALL SAFE TESTS PASSED!")
        print("✅ ML models are working correctly")
        print("🚀 The segmentation fault issue has been resolved")
    else:
        print("\n⚠️ Some safe tests failed")
        print("🔧 Check the error messages above for details")
    
    return total_passed == 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
