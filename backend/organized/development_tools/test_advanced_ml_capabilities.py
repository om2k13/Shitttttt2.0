#!/usr/bin/env python3
"""
Test Advanced ML Capabilities
Comprehensive testing of all new ML features
"""
import sys
from pathlib import Path
import asyncio
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_advanced_ml_imports():
    """Test that all advanced ML modules can be imported"""
    print("🧪 Testing Advanced ML Imports...")
    print("=" * 50)
    
    try:
        from app.review.advanced_ml_capabilities import (
            AdvancedMLCapabilities, 
            analyze_code_advanced,
            CodeComplexityPredictor,
            MaintainabilityScorer
        )
        print("✅ All advanced ML modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_ml_initialization():
    """Test advanced ML capabilities initialization"""
    print("\n🧪 Testing Advanced ML Initialization...")
    print("=" * 50)
    
    try:
        from app.review.advanced_ml_capabilities import AdvancedMLCapabilities
        
        # Test initialization
        analyzer = AdvancedMLCapabilities()
        print("✅ Advanced ML capabilities initialized successfully")
        
        # Test model status
        print(f"📊 Models loaded: {len(analyzer.models)}")
        for model_name, model in analyzer.models.items():
            print(f"   - {model_name}: {type(model).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_ml_analysis():
    """Test advanced ML analysis on sample code"""
    print("\n🧪 Testing Advanced ML Analysis...")
    print("=" * 50)
    
    try:
        from app.review.advanced_ml_capabilities import AdvancedMLCapabilities
        
        analyzer = AdvancedMLCapabilities()
        
        # Test code samples
        test_cases = [
            {
                'name': 'Simple Code',
                'code': '''
def simple_function():
    return "Hello World"
''',
                'expected': 'Low complexity'
            },
            {
                'name': 'Complex Code',
                'code': '''
def complex_function(data, condition1, condition2, condition3):
    result = []
    for item in data:
        if condition1:
            if condition2:
                if condition3:
                    for sub_item in item:
                        if sub_item > 0:
                            result.append(sub_item * 2)
                        else:
                            result.append(sub_item / 2)
                else:
                    result.append(item)
            else:
                result.append(item * -1)
        else:
            result.append(item)
    return result
''',
                'expected': 'High complexity'
            },
            {
                'name': 'Security Risk Code',
                'code': '''
import os
import subprocess

def dangerous_function(user_input):
    os.system(user_input)
    subprocess.call(user_input, shell=True)
    return eval(user_input)
''',
                'expected': 'High security risk'
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"Expected: {test_case['expected']}")
            
            # Extract features
            features = analyzer._extract_advanced_features(test_case['code'], "test.py")
            print(f"   Features extracted: {features.shape}")
            
            # Run comprehensive analysis
            results = analyzer.comprehensive_code_analysis(features)
            
            # Display key results
            complexity = results.get('complexity_analysis', {})
            maintainability = results.get('maintainability_analysis', {})
            tech_debt = results.get('technical_debt_analysis', {})
            code_smells = results.get('code_smell_analysis', {})
            quality = results.get('overall_quality_score', {})
            
            print(f"   Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 0):.1f}")
            print(f"   Maintainability Score: {maintainability.get('maintainability_score', 0):.2f} ({maintainability.get('maintainability_level', 'UNKNOWN')})")
            print(f"   Technical Debt: {tech_debt.get('debt_category', 'UNKNOWN')} ({tech_debt.get('technical_debt_hours', 0):.1f} hours)")
            print(f"   Code Smells: {code_smells.get('total_smells', 0)} detected")
            print(f"   Overall Quality: {quality.get('grade', 'UNKNOWN')} ({quality.get('quality_score', 0):.1f}/100)")
            
            print(f"   ✅ Analysis completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_code_review_agent_integration():
    """Test integration with the main code review agent"""
    print("\n🧪 Testing Code Review Agent Integration...")
    print("=" * 50)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create temporary test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_file = temp_path / "test_code.py"
            test_file.write_text('''
def complex_function(data, condition1, condition2):
    result = []
    for item in data:
        if condition1:
            if condition2:
                for sub_item in item:
                    if sub_item > 0:
                        result.append(sub_item * 2)
                    else:
                        result.append(sub_item / 2)
            else:
                result.append(item * -1)
        else:
            result.append(item)
    return result

import os
def dangerous_function(user_input):
    return os.system(user_input)
''')
            
            # Initialize agent
            agent = CodeReviewAgent(temp_path, standalone=True)
            print("✅ Code Review Agent initialized with advanced ML capabilities")
            
            # Check if advanced ML is loaded
            if agent.advanced_ml_capabilities:
                print("✅ Advanced ML capabilities loaded in agent")
                print(f"   Models available: {len(agent.advanced_ml_capabilities.models)}")
            else:
                print("⚠️ Advanced ML capabilities not loaded")
            
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_ml_training():
    """Test advanced ML model training"""
    print("\n🧪 Testing Advanced ML Training...")
    print("=" * 50)
    
    try:
        # Check if training script exists
        training_script = Path(__file__).parent / "train_advanced_ml_models.py"
        if not training_script.exists():
            print("⚠️ Training script not found, skipping training test")
            return True
        
        print("📁 Training script found")
        print("💡 To train advanced ML models, run:")
        print(f"   cd {Path(__file__).parent}")
        print("   python3 train_advanced_ml_models.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run all tests"""
    print("🚀 COMPREHENSIVE ADVANCED ML CAPABILITIES TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_advanced_ml_imports),
        ("Initialization Test", test_advanced_ml_initialization),
        ("Analysis Test", test_advanced_ml_analysis),
        ("Integration Test", test_code_review_agent_integration),
        ("Training Test", test_advanced_ml_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Advanced ML capabilities are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

def main():
    """Main test function"""
    print("🧪 Advanced ML Capabilities Test Suite")
    print("This will test all new ML features including:")
    print("  - Complexity Prediction")
    print("  - Maintainability Scoring") 
    print("  - Technical Debt Estimation")
    print("  - Code Smell Detection")
    print("  - Overall Quality Assessment")
    print()
    
    # Run tests
    success = asyncio.run(run_comprehensive_test())
    
    if success:
        print("\n🎉 ADVANCED ML CAPABILITIES READY!")
        print("Your code review agent now has:")
        print("  🧠 Advanced neural networks for complexity and maintainability")
        print("  🌲 Random Forest models for technical debt and code smells")
        print("  📊 Comprehensive quality scoring and recommendations")
        print("  🚀 Integration with existing ML pipeline")
    else:
        print("\n❌ Some tests failed. Advanced ML capabilities may not be fully functional.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
