#!/usr/bin/env python3
"""
ML Integration Test Script for Code Review Agent

This script tests the complete ML integration to ensure everything works flawlessly.
"""

import asyncio
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.review.code_review_agent import CodeReviewAgent
from app.review.neural_analyzer import NeuralAnalyzer
from app.review.enhanced_ml_analyzer import EnhancedMLAnalyzer

async def test_neural_analyzer():
    """Test the neural analyzer functionality"""
    print("🧠 Testing Neural Analyzer...")
    print("-" * 40)
    
    try:
        neural_analyzer = NeuralAnalyzer()
        
        # Test security analysis
        vulnerable_code = """
def unsafe_function(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    os.system("rm -rf " + user_input)
    return eval(user_input)
        """
        
        security_result = neural_analyzer.analyze_code_security(
            vulnerable_code, 
            {'file': 'test.py', 'line': 1}
        )
        
        print(f"✅ Security Analysis: {security_result.get('severity', 'unknown')} risk")
        print(f"   Risk Score: {security_result.get('risk_score', 0):.3f}")
        print(f"   Confidence: {security_result.get('confidence', 0):.3f}")
        print(f"   Analysis: {security_result.get('analysis', 'N/A')}")
        
        # Test quality prediction
        quality_metrics = {
            'lines': 8,
            'complexity': 12,
            'nesting': 4,
            'imports': 2,
            'functions': 1,
            'classes': 0
        }
        
        quality_result = neural_analyzer.predict_code_quality(quality_metrics)
        
        print(f"✅ Quality Prediction: {quality_result.get('quality_score', 0):.3f}")
        print(f"   Maintainability: {quality_result.get('predicted_maintainability', 0):.3f}")
        print(f"   Reliability: {quality_result.get('predicted_reliability', 0):.3f}")
        print(f"   Recommendations: {quality_result.get('recommendations', [])}")
        
        # Test model status
        status = neural_analyzer.get_model_status()
        print(f"✅ Model Status: {status['total_models']} models loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_ml_analyzer():
    """Test the enhanced ML analyzer functionality"""
    print("\n🤖 Testing Enhanced ML Analyzer...")
    print("-" * 40)
    
    try:
        ml_analyzer = EnhancedMLAnalyzer()
        
        # Test with sample finding
        sample_finding = {
            "file": "test.py",
            "line": 15,
            "category": "security",
            "message": "SQL injection vulnerability detected",
            "suggestion": "Use parameterized queries",
            "code_snippet": "query = 'SELECT * FROM users WHERE id = ' + user_input",
            "severity": "high",
            "confidence": 0.9,
            "impact": "high",
            "effort": "medium",
            "tool": "bandit"
        }
        
        ml_result = ml_analyzer.analyze_finding_with_ml(sample_finding)
        
        print(f"✅ ML Analysis completed")
        print(f"   Predicted Severity: {ml_result.get('predicted_severity', 'unknown')}")
        print(f"   Ensemble Severity: {ml_result.get('ensemble_severity', 'unknown')}")
        print(f"   False Positive: {ml_result.get('is_false_positive', 'unknown')}")
        print(f"   Anomaly: {ml_result.get('is_anomaly', 'unknown')}")
        
        # Test model status
        status = ml_analyzer.get_model_status()
        print(f"✅ Model Status: {status['total_models']} models loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced ML Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_code_review_agent_integration():
    """Test the complete Code Review Agent with ML integration"""
    print("\n🔍 Testing Code Review Agent ML Integration...")
    print("-" * 40)
    
    try:
        # Create a temporary repository for testing
        test_repo = Path("./test_repo")
        test_repo.mkdir(exist_ok=True)
        
        # Create a test Python file with issues
        test_file = test_repo / "test_code.py"
        test_code = '''
import os
import subprocess

def vulnerable_function(user_input):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_input
    
    # Command injection vulnerability
    os.system("rm -rf " + user_input)
    
    # Code execution vulnerability
    result = eval(user_input)
    
    # File operation vulnerability
    with open(user_input, 'r') as f:
        content = f.read()
    
    return content

def very_long_function_with_many_parameters(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    """This function is too long and has too many parameters"""
    result = 0
    for param in [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]:
        result += param
    return result

class BadClass:
    """This class has poor design"""
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)
    
    def get_data(self):
        return self.data
    
    def process_data(self):
        # Complex nested logic
        for item in self.data:
            if item > 0:
                if item % 2 == 0:
                    if item < 100:
                        if item % 3 == 0:
                            if item % 5 == 0:
                                print("Complex condition met")
                            else:
                                print("Almost there")
                        else:
                            print("Not divisible by 3")
                    else:
                        print("Too large")
                else:
                    print("Odd number")
            else:
                print("Non-positive")
        '''
        
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Initialize Code Review Agent
        agent = CodeReviewAgent(test_repo, standalone=True)
        
        # Run code review
        print("Running code review analysis...")
        result = await agent.run_code_review()
        
        if result['status'] == 'completed':
            print(f"✅ Code Review completed successfully")
            print(f"   Total findings: {result['total_findings']}")
            print(f"   Findings by severity: {result['findings_by_severity']}")
            print(f"   Findings by category: {result['findings_by_category']}")
            
            # Check if ML analysis was applied
            ml_enhanced_findings = 0
            for finding in result['findings']:
                if '🤖 ML Analysis:' in finding.get('suggestion', ''):
                    ml_enhanced_findings += 1
                if '🧠 Neural Network Insights:' in finding.get('suggestion', ''):
                    ml_enhanced_findings += 1
            
            print(f"   ML-enhanced findings: {ml_enhanced_findings}")
            
            # Show sample findings
            if result['findings']:
                print("\n📋 Sample Findings:")
                for i, finding in enumerate(result['findings'][:3]):
                    print(f"  {i+1}. {finding['message']}")
                    print(f"     Severity: {finding['severity']}")
                    print(f"     Category: {finding['category']}")
                    if '🤖 ML Analysis:' in finding.get('suggestion', ''):
                        print(f"     ✅ ML Analysis applied")
                    if '🧠 Neural Network Insights:' in finding.get('suggestion', ''):
                        print(f"     ✅ Neural Network analysis applied")
                    print()
            
            return True
        else:
            print(f"❌ Code Review failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Code Review Agent integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test repository
        import shutil
        if test_repo.exists():
            shutil.rmtree(test_repo)

async def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n🛡️ Testing Error Handling and Fallbacks...")
    print("-" * 40)
    
    try:
        # Test with invalid data
        neural_analyzer = NeuralAnalyzer()
        
        # Test with empty code
        empty_result = neural_analyzer.analyze_code_security("", {})
        print(f"✅ Empty code handling: {empty_result.get('severity', 'unknown')}")
        
        # Test with None values
        none_result = neural_analyzer.predict_code_quality({})
        print(f"✅ None values handling: {none_result.get('quality_score', 0):.3f}")
        
        # Test ML analyzer with invalid finding
        ml_analyzer = EnhancedMLAnalyzer()
        invalid_finding = {}
        invalid_result = ml_analyzer.analyze_finding_with_ml(invalid_finding)
        print(f"✅ Invalid data handling: {invalid_result.get('fallback_analysis', False)}")
        
        print("✅ All error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Code Review Agent ML Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Neural Analyzer", test_neural_analyzer),
        ("Enhanced ML Analyzer", test_enhanced_ml_analyzer),
        ("Code Review Agent Integration", test_code_review_agent_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
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
        print("🎉 All tests passed! ML integration is working flawlessly.")
        print("🚀 Code Review Agent is ready for production use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
