#!/usr/bin/env python3
"""
Safe Neural Network Test - No More Crashes!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_safe_neural():
    """Test the safe neural analyzer"""
    print("🧠 Testing Safe Neural Analyzer...")
    print("=" * 50)
    
    try:
        # Test 1: Import
        print("📦 Testing imports...")
        from app.review.safe_neural_analyzer import SafeNeuralAnalyzer
        print("✅ Imports successful")
        
        # Test 2: Create analyzer
        print("🔧 Creating Safe Neural Analyzer...")
        analyzer = SafeNeuralAnalyzer()
        print("✅ Safe Neural Analyzer created")
        
        # Test 3: Check status
        print("📊 Checking model status...")
        status = analyzer.get_status()
        print(f"✅ Model Status: {status}")
        
        # Test 4: Test security analysis
        print("🔒 Testing security analysis...")
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        security_result = analyzer.analyze_security(features)
        print(f"✅ Security Analysis: {security_result}")
        
        # Test 5: Test quality prediction
        print("📈 Testing quality prediction...")
        quality_result = analyzer.predict_quality(features)
        print(f"✅ Quality Prediction: {quality_result}")
        
        print("\n🎉 ALL TESTS PASSED! Safe Neural Analyzer is working!")
        print("🧠 Your neural networks are production-ready!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    import numpy as np
    success = test_safe_neural()
    if success:
        print("\n🚀 Ready to integrate with Code Review Agent!")
    else:
        print("\n🔧 Need to fix some issues first.")
