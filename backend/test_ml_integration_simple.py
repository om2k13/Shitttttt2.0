#!/usr/bin/env python3
"""
Simple ML Integration Test
Tests if ML models are loading and working correctly
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ml_loading():
    """Test if ML models can be loaded"""
    print("🧪 Testing ML Model Loading...")
    print("=" * 50)
    
    try:
        # Test Production ML Analyzer
        print("1️⃣ Testing Production ML Analyzer...")
        from app.review.production_ml_analyzer import ProductionMLAnalyzer
        
        prod_analyzer = ProductionMLAnalyzer()
        status = prod_analyzer.get_model_status()
        print(f"   ✅ Production ML loaded: {status['total_models']} models")
        
        # Test Advanced ML Capabilities
        print("2️⃣ Testing Advanced ML Capabilities...")
        from app.review.advanced_ml_capabilities import AdvancedMLCapabilities
        
        adv_analyzer = AdvancedMLCapabilities()
        print(f"   ✅ Advanced ML loaded: {len(adv_analyzer.models)} models")
        
        # Test Code Review Agent
        print("3️⃣ Testing Code Review Agent...")
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test file
            test_file = temp_path / "test.py"
            test_file.write_text('''
def dangerous_function(user_input):
    import os
    return os.system(user_input)
''')
            
            agent = CodeReviewAgent(temp_path, standalone=True)
            print(f"   ✅ Code Review Agent initialized")
            print(f"   📊 Production ML available: {agent.production_ml_analyzer is not None}")
            print(f"   📊 Advanced ML available: {agent.advanced_ml_capabilities is not None}")
            
            if agent.production_ml_analyzer:
                print("   🧠 Production ML models loaded successfully")
            else:
                print("   ❌ Production ML models failed to load")
                
            if agent.advanced_ml_capabilities:
                print("   🧠 Advanced ML models loaded successfully")
            else:
                print("   ❌ Advanced ML models failed to load")
        
        print("\n🎉 ML Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ML Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_analysis():
    """Test if ML analysis works on sample code"""
    print("\n🧪 Testing ML Analysis...")
    print("=" * 50)
    
    try:
        from app.review.production_ml_analyzer import ProductionMLAnalyzer
        
        analyzer = ProductionMLAnalyzer()
        
        # Test code
        test_code = '''
import os
def dangerous_function(user_input):
    return os.system(user_input)
'''
        
        print("🔍 Analyzing test code with ML...")
        results = analyzer.analyze_code_ml(test_code, "test.py")
        
        print(f"   ✅ Analysis completed")
        print(f"   🎯 Risk level: {results.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')}")
        print(f"   🔢 Risk score: {results.get('risk_assessment', {}).get('risk_score', 0.0):.2f}")
        
        if results.get('ensemble_prediction'):
            ensemble = results['ensemble_prediction']
            print(f"   🎯 Ensemble vulnerable: {ensemble.get('is_vulnerable', False)}")
        
        print("\n🎉 ML Analysis Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ML Analysis Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 ML Integration Test Suite")
    print("Testing if ML models are properly loaded and working")
    print()
    
    # Test 1: ML Loading
    loading_success = test_ml_loading()
    
    # Test 2: ML Analysis
    analysis_success = test_ml_analysis()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"ML Loading Test: {'✅ PASSED' if loading_success else '❌ FAILED'}")
    print(f"ML Analysis Test: {'✅ PASSED' if analysis_success else '❌ FAILED'}")
    
    if loading_success and analysis_success:
        print("\n🎉 ALL TESTS PASSED! ML integration is working correctly.")
        print("🚀 Your code review agent should now use ML analysis.")
    else:
        print("\n⚠️ Some tests failed. ML integration may not work properly.")
        print("🔧 Check the error messages above for details.")
    
    return loading_success and analysis_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
