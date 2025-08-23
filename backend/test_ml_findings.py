#!/usr/bin/env python3
"""
Test ML Findings Generation
Verifies that ML analysis generates findings with proper categorization
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_ml_findings_generation():
    """Test that ML analysis generates findings with ml_ categories"""
    print("🧪 Testing ML Findings Generation...")
    print("=" * 50)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create test repository with problematic code
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create problematic Python files that should trigger ML findings
            test_file1 = temp_path / "dangerous.py"
            test_file1.write_text('''
import os
import subprocess
import pickle

def dangerous_function(user_input):
    # This should trigger security risk detection
    os.system(user_input)
    subprocess.call(user_input, shell=True)
    return eval(user_input)

def complex_function(data, condition1, condition2, condition3):
    # This should trigger complexity analysis
    result = []
    for item in data:
        if condition1:
            if condition2:
                if condition3:
                    for sub_item in item:
                        if sub_item > 0:
                            if sub_item < 100:
                                if sub_item % 2 == 0:
                                    result.append(sub_item * 2)
                                else:
                                    result.append(sub_item * 3)
                            else:
                                result.append(sub_item / 2)
                        else:
                            result.append(sub_item / 3)
                else:
                    result.append(item)
            else:
                result.append(item * -1)
        else:
            result.append(item)
    return result

def long_function_with_many_parameters(param1, param2, param3, param4, param5, param6, param7, param8, param9, param10):
    # This should trigger maintainability analysis
    result = param1 + param2 + param3 + param4 + param5 + param6 + param7 + param8 + param9 + param10
    if result > 100:
        if result < 200:
            if result % 2 == 0:
                if result % 3 == 0:
                    if result % 5 == 0:
                        return result * 2
                    else:
                        return result * 3
                else:
                    return result * 4
            else:
                return result * 5
        else:
            return result * 6
    else:
        return result * 7
''')
            
            test_file2 = temp_path / "utils.py"
            test_file2.write_text('''
import yaml
import xml.etree.ElementTree as ET

def load_user_data(data):
    # This should trigger security risk detection
    return yaml.load(data)

def parse_xml(xml_string):
    # This should trigger security risk detection
    return ET.fromstring(xml_string)

def process_input(user_input):
    # This should trigger security risk detection
    return exec(user_input)
''')
            
            print(f"📁 Created test repository at: {temp_path}")
            print(f"📄 Test files created with problematic code patterns")
            
            # Initialize code review agent
            print("\n🚀 Initializing Code Review Agent...")
            agent = CodeReviewAgent(temp_path, standalone=True)
            
            print(f"📊 Agent initialized successfully")
            print(f"   Production ML available: {agent.production_ml_analyzer is not None}")
            print(f"   Advanced ML available: {agent.advanced_ml_capabilities is not None}")
            
            # Run ML analysis manually to see what happens
            print("\n🔍 Running ML Analysis Manually...")
            
            if agent.production_ml_analyzer:
                print("   Testing Production ML Analysis...")
                for file_path in [test_file1, test_file2]:
                    try:
                        with open(file_path, 'r') as f:
                            code_content = f.read()
                        
                        relative_path = file_path.relative_to(temp_path)
                        print(f"     Analyzing {relative_path}...")
                        
                        ml_results = agent.production_ml_analyzer.analyze_code_ml(
                            code_content, str(file_path)
                        )
                        
                        print(f"       Risk level: {ml_results.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')}")
                        print(f"       Risk score: {ml_results.get('risk_assessment', {}).get('risk_score', 0.0):.2f}")
                        
                        # Process ML results
                        await agent._process_ml_results(str(relative_path), ml_results)
                        
                    except Exception as e:
                        print(f"       Error analyzing {relative_path}: {e}")
            
            if agent.advanced_ml_capabilities:
                print("   Testing Advanced ML Analysis...")
                for file_path in [test_file1, test_file2]:
                    try:
                        with open(file_path, 'r') as f:
                            code_content = f.read()
                        
                        relative_path = file_path.relative_to(temp_path)
                        print(f"     Analyzing {relative_path} with advanced ML...")
                        
                        from .advanced_ml_capabilities import _extract_advanced_features
                        features = _extract_advanced_features(code_content, str(file_path))
                        advanced_results = agent.advanced_ml_capabilities.comprehensive_code_analysis(features)
                        
                        print(f"       Complexity: {advanced_results.get('complexity_analysis', {}).get('cyclomatic_complexity', 0):.1f}")
                        print(f"       Maintainability: {advanced_results.get('maintainability_analysis', {}).get('maintainability_level', 'UNKNOWN')}")
                        
                        # Process advanced ML results
                        await agent._process_advanced_ml_results(str(relative_path), advanced_results)
                        
                    except Exception as e:
                        print(f"       Error analyzing {relative_path} with advanced ML: {e}")
            
            # Check findings
            print(f"\n📊 Findings Generated:")
            print(f"   Total findings: {len(agent.findings)}")
            
            # Categorize findings
            ml_findings = [f for f in agent.findings if f.category.startswith('ml_')]
            security_findings = [f for f in agent.findings if 'security' in f.category.lower()]
            complexity_findings = [f for f in agent.findings if 'complexity' in f.category.lower()]
            
            print(f"   ML findings: {len(ml_findings)}")
            print(f"   Security findings: {len(security_findings)}")
            print(f"   Complexity findings: {len(complexity_findings)}")
            
            # Show sample findings
            if agent.findings:
                print(f"\n🔍 Sample Findings:")
                for i, finding in enumerate(agent.findings[:10]):
                    print(f"   {i+1}. {finding.category}: {finding.message[:80]}...")
            
            # Check if ML analysis generated findings
            if len(ml_findings) > 0:
                print("\n✅ ML Analysis Successfully Generated Findings!")
                print(f"   Found {len(ml_findings)} ML-categorized findings")
                return True
            else:
                print("\n⚠️ ML Analysis Did Not Generate ML-Categorized Findings")
                print("   This suggests the ML analysis may not be working as expected")
                return False
                
    except Exception as e:
        print(f"\n❌ ML Findings Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 ML Findings Generation Test")
    print("Testing if ML analysis generates properly categorized findings")
    print()
    
    success = await test_ml_findings_generation()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    if success:
        print("🎉 ML FINDINGS TEST PASSED!")
        print("✅ ML analysis is generating properly categorized findings")
        print("🚀 Your code review agent should now show ML findings")
    else:
        print("⚠️ ML FINDINGS TEST FAILED")
        print("❌ ML analysis may not be generating findings correctly")
        print("🔧 Check the error messages above for details")
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
