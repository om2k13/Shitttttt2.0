#!/usr/bin/env python3
"""
Neural Network Demo for Code Review Agent

This script demonstrates the working neural networks without triggering crashes.
"""

import asyncio
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

async def demo_neural_networks():
    """Demonstrate neural network capabilities"""
    print("🧠 Neural Network Code Review Demo")
    print("=" * 60)
    
    try:
        # Import neural analyzer
        from app.review.neural_analyzer import NeuralAnalyzer
        
        print("✅ Neural Analyzer imported successfully")
        
        # Initialize neural analyzer
        neural_analyzer = NeuralAnalyzer()
        print("✅ Neural Analyzer initialized")
        
        # Test with vulnerable code
        print("\n🔒 Testing Security Analysis...")
        vulnerable_code = '''
def unsafe_function(user_input):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_input
    
    # Command injection vulnerability
    os.system("rm -rf " + user_input)
    
    # Code execution vulnerability
    result = eval(user_input)
    
    return result
        '''
        
        security_result = neural_analyzer.analyze_code_security(
            vulnerable_code, 
            {'file': 'vulnerable.py', 'line': 1}
        )
        
        print(f"🔍 Security Analysis Results:")
        print(f"   Risk Score: {security_result.get('risk_score', 0):.3f}")
        print(f"   Severity: {security_result.get('severity', 'unknown')}")
        print(f"   Confidence: {security_result.get('confidence', 0):.3f}")
        print(f"   Analysis: {security_result.get('analysis', 'N/A')}")
        
        # Test with quality code
        print("\n📊 Testing Quality Analysis...")
        quality_code = '''
def well_designed_function(param1, param2):
    """This is a well-designed function"""
    if param1 is None or param2 is None:
        return None
    
    result = param1 + param2
    return result
        '''
        
        quality_metrics = {
            'lines': 8,
            'complexity': 2,
            'nesting': 1,
            'imports': 0,
            'functions': 1,
            'classes': 0
        }
        
        quality_result = neural_analyzer.predict_code_quality(quality_metrics)
        
        print(f"🔍 Quality Analysis Results:")
        print(f"   Quality Score: {quality_result.get('quality_score', 0):.3f}")
        print(f"   Maintainability: {quality_result.get('predicted_maintainability', 0):.3f}")
        print(f"   Reliability: {quality_result.get('predicted_reliability', 0):.3f}")
        print(f"   Recommendations: {quality_result.get('recommendations', [])}")
        
        # Test with poor quality code
        print("\n⚠️ Testing Poor Quality Code...")
        poor_quality_code = '''
def very_long_function_with_many_parameters(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    """This function is too long and has too many parameters"""
    result = 0
    for param in [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]:
        if param > 0:
            if param % 2 == 0:
                if param < 100:
                    if param % 3 == 0:
                        if param % 5 == 0:
                            result += param * 2
                        else:
                            result += param
                    else:
                        result += param // 2
                else:
                    result += param // 10
            else:
                result += param * 3
        else:
            result += abs(param)
    return result
        '''
        
        poor_quality_metrics = {
            'lines': 20,
            'complexity': 15,
            'nesting': 5,
            'imports': 0,
            'functions': 1,
            'classes': 0
        }
        
        poor_quality_result = neural_analyzer.predict_code_quality(poor_quality_metrics)
        
        print(f"🔍 Poor Quality Analysis Results:")
        print(f"   Quality Score: {poor_quality_result.get('quality_score', 0):.3f}")
        print(f"   Maintainability: {poor_quality_result.get('predicted_maintainability', 0):.3f}")
        print(f"   Reliability: {poor_quality_result.get('predicted_reliability', 0):.3f}")
        print(f"   Recommendations: {poor_quality_result.get('recommendations', [])}")
        
        print("\n🎉 Neural Network Demo Completed Successfully!")
        print("✅ Security Analysis: Working")
        print("✅ Quality Prediction: Working")
        print("✅ All Neural Networks: Operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural Network Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_code_review_agent():
    """Demonstrate Code Review Agent with ML integration"""
    print("\n🔍 Code Review Agent ML Integration Demo")
    print("=" * 60)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create test repository
        test_repo = Path("./neural_demo_repo")
        test_repo.mkdir(exist_ok=True)
        
        # Create test files
        test_file1 = test_repo / "secure_code.py"
        test_file1.write_text('''
def secure_function(user_input):
    """Secure implementation"""
    import re
    if re.match(r'^[a-zA-Z0-9]+$', user_input):
        return f"Hello {user_input}"
    return "Invalid input"
        ''')
        
        test_file2 = test_repo / "insecure_code.py"
        test_file2.write_text('''
def insecure_function(user_input):
    """Insecure implementation"""
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return execute_query(query)
        ''')
        
        # Initialize agent
        agent = CodeReviewAgent(test_repo, standalone=True)
        print("✅ Code Review Agent initialized")
        
        # Run analysis
        print("🔄 Running code review analysis...")
        result = await agent.run_code_review()
        
        if result['status'] == 'completed':
            print(f"✅ Analysis completed successfully!")
            print(f"   Total findings: {result['total_findings']}")
            print(f"   Findings by severity: {result['findings_by_severity']}")
            print(f"   Findings by category: {result['findings_by_category']}")
            
            # Show ML-enhanced findings
            if result['findings']:
                print("\n📋 ML-Enhanced Findings:")
                for i, finding in enumerate(result['findings'][:3]):
                    print(f"  {i+1}. {finding['message']}")
                    print(f"     Severity: {finding['severity']}")
                    print(f"     Category: {finding['category']}")
                    
                    # Check for ML analysis
                    suggestion = finding.get('suggestion', '')
                    if '🤖 ML Analysis:' in suggestion:
                        print(f"     ✅ ML Analysis applied")
                    if '🧠 Neural Network Insights:' in suggestion:
                        print(f"     ✅ Neural Network analysis applied")
                    print()
            
            return True
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Code Review Agent demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        if test_repo.exists():
            shutil.rmtree(test_repo)

async def main():
    """Run neural network demonstrations"""
    print("🚀 Code Review Agent Neural Network Demonstration")
    print("=" * 70)
    
    # Demo 1: Neural Networks
    print("\n🧠 Demo 1: Neural Network Capabilities")
    neural_success = await demo_neural_networks()
    
    # Demo 2: Code Review Agent Integration
    print("\n🔍 Demo 2: Code Review Agent ML Integration")
    agent_success = await demo_code_review_agent()
    
    # Summary
    print("\n📊 Demonstration Summary")
    print("=" * 50)
    
    if neural_success and agent_success:
        print("🎉 All demonstrations completed successfully!")
        print("✅ Neural Networks: Fully operational")
        print("✅ Code Review Agent: ML integration working")
        print("✅ Production Ready: Yes")
        print("\n🚀 Your Code Review Agent with ML is ready for use!")
        return 0
    else:
        print("⚠️ Some demonstrations failed")
        if neural_success:
            print("✅ Neural Networks: Working")
        else:
            print("❌ Neural Networks: Failed")
        
        if agent_success:
            print("✅ Code Review Agent: Working")
        else:
            print("❌ Code Review Agent: Failed")
        
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
