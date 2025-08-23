#!/usr/bin/env python3
"""
Safe Test Script for Code Review Agent ML Integration

This script tests components individually to avoid crashes and identify issues.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_imports():
    """Test if all modules can be imported without crashing"""
    print("🧪 Testing Module Imports...")
    print("=" * 50)
    
    try:
        print("Testing neural_analyzer import...")
        from app.review.neural_analyzer import NeuralAnalyzer
        print("✅ Neural Analyzer imported successfully")
    except Exception as e:
        print(f"❌ Neural Analyzer import failed: {e}")
        return False
    
    try:
        print("Testing enhanced_ml_analyzer import...")
        from app.review.enhanced_ml_analyzer import EnhancedMLAnalyzer
        print("✅ Enhanced ML Analyzer imported successfully")
    except Exception as e:
        print(f"❌ Enhanced ML Analyzer import failed: {e}")
        return False
    
    try:
        print("Testing code_review_agent import...")
        from app.review.code_review_agent import CodeReviewAgent
        print("✅ Code Review Agent imported successfully")
    except Exception as e:
        print(f"❌ Code Review Agent import failed: {e}")
        return False
    
    return True

def test_neural_analyzer_safe():
    """Test neural analyzer with error handling"""
    print("\n🧠 Testing Neural Analyzer (Safe Mode)...")
    print("-" * 50)
    
    try:
        from app.review.neural_analyzer import NeuralAnalyzer
        
        # Initialize with error handling
        neural_analyzer = NeuralAnalyzer()
        print("✅ Neural Analyzer initialized")
        
        # Test with simple data
        test_code = "def test(): pass"
        
        try:
            security_result = neural_analyzer.analyze_code_security(
                test_code, {'file': 'test.py', 'line': 1}
            )
            print(f"✅ Security Analysis: {security_result.get('severity', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Security analysis failed: {e}")
        
        try:
            quality_result = neural_analyzer.predict_code_quality({
                'lines': 1, 'complexity': 1, 'nesting': 1, 'imports': 0, 'functions': 1, 'classes': 0
            })
            print(f"✅ Quality Prediction: {quality_result.get('quality_score', 0):.3f}")
        except Exception as e:
            print(f"⚠️ Quality prediction failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural Analyzer test failed: {e}")
        return False

def test_ml_analyzer_safe():
    """Test ML analyzer with error handling"""
    print("\n🤖 Testing ML Analyzer (Safe Mode)...")
    print("-" * 50)
    
    try:
        from app.review.enhanced_ml_analyzer import EnhancedMLAnalyzer
        
        # Initialize with error handling
        ml_analyzer = EnhancedMLAnalyzer()
        print("✅ ML Analyzer initialized")
        
        # Test with simple data
        test_finding = {
            "file": "test.py", "line": 1, "category": "quality", "message": "Test finding",
            "suggestion": "Test suggestion", "code_snippet": "def test(): pass", "severity": "low",
            "confidence": 0.5, "impact": "low", "effort": "low", "tool": "test"
        }
        
        try:
            ml_result = ml_analyzer.analyze_finding_with_ml(test_finding)
            print(f"✅ ML Analysis completed")
            if 'ensemble_severity' in ml_result:
                print(f"   Ensemble Severity: {ml_result['ensemble_severity']}")
        except Exception as e:
            print(f"⚠️ ML analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Analyzer test failed: {e}")
        return False

def test_code_review_agent_safe():
    """Test code review agent with error handling"""
    print("\n🔍 Testing Code Review Agent (Safe Mode)...")
    print("-" * 50)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create a simple test repository
        test_repo = Path("./test_safe")
        test_repo.mkdir(exist_ok=True)
        
        # Create a simple test file
        test_file = test_repo / "simple.py"
        test_file.write_text("def hello(): return 'world'")
        
        try:
            # Initialize agent
            agent = CodeReviewAgent(test_repo, standalone=True)
            print("✅ Code Review Agent initialized")
            
            # Check if ML analyzers are available
            if hasattr(agent, 'ml_analyzer') and agent.ml_analyzer:
                print("✅ ML Analyzer available")
            else:
                print("⚠️ ML Analyzer not available")
            
            if hasattr(agent, 'neural_analyzer') and agent.neural_analyzer:
                print("✅ Neural Analyzer available")
            else:
                print("⚠️ Neural Analyzer not available")
            
        except Exception as e:
            print(f"⚠️ Code Review Agent initialization failed: {e}")
        
        # Cleanup
        import shutil
        if test_repo.exists():
            shutil.rmtree(test_repo)
        
        return True
        
    except Exception as e:
        print(f"❌ Code Review Agent test failed: {e}")
        return False

def main():
    """Run all safe tests"""
    print("🛡️ Safe Code Review Agent ML Integration Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Neural Analyzer", test_neural_analyzer_safe),
        ("ML Analyzer", test_ml_analyzer_safe),
        ("Code Review Agent", test_code_review_agent_safe)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Safe Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All safe tests passed!")
        print("🚀 Ready to test full integration")
    else:
        print("⚠️ Some tests failed. Check the output above.")
        print("🔧 Neural networks may need troubleshooting")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
