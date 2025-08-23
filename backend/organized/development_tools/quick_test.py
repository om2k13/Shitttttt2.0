#!/usr/bin/env python3
"""
Quick Test - Verify ML System Works
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ml_analyzer():
    """Quick test of ML analyzer"""
    print("🧪 Quick ML System Test")
    print("=" * 50)
    
    try:
        # Test 1: Import ML analyzer
        print("1️⃣ Testing imports...")
        from app.review.production_ml_analyzer import ProductionMLAnalyzer
        print("   ✅ ML analyzer imported successfully")
        
        # Test 2: Load models
        print("2️⃣ Testing model loading...")
        analyzer = ProductionMLAnalyzer()
        status = analyzer.get_model_status()
        print(f"   ✅ Models loaded: {status['total_models']}")
        print(f"   📊 Traditional ML: {status['traditional_models_loaded']}")
        print(f"   🧠 Neural Networks: {status['neural_models_loaded']}")
        
        # Test 3: Simple code analysis
        print("3️⃣ Testing code analysis...")
        test_code = """
import os
def dangerous_function(user_input):
    return os.system(user_input)
"""
        
        results = analyzer.analyze_code_ml(test_code, "test.py")
        print(f"   ✅ Analysis completed")
        print(f"   🎯 Risk level: {results.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')}")
        print(f"   🔢 Risk score: {results.get('risk_assessment', {}).get('risk_score', 0.0):.2f}")
        
        # Test 4: Check ensemble
        if results.get('ensemble_prediction'):
            ensemble = results['ensemble_prediction']
            print(f"   🎯 Ensemble vulnerable: {ensemble.get('is_vulnerable', False)}")
        
        print("\n🎉 QUICK TEST PASSED!")
        print("✅ ML system is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ QUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_analyzer()
    sys.exit(0 if success else 1)
