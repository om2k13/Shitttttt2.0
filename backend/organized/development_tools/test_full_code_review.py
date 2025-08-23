#!/usr/bin/env python3
"""
Full Code Review Test
Simulates the complete code review process to test ML integration
"""
import sys
import asyncio
from pathlib import Path
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_full_code_review():
    """Test the complete code review process"""
    print("🧪 Testing Full Code Review Process...")
    print("=" * 60)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create temporary repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test repository structure
            backend_dir = temp_path / "backend"
            backend_dir.mkdir()
            
            # Create test Python files
            test_file1 = backend_dir / "app.py"
            test_file1.write_text('''
from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def index():
    user_input = request.args.get("input")
    return os.system(user_input)  # Security vulnerability

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
''')
            
            test_file2 = backend_dir / "utils.py"
            test_file2.write_text('''
import subprocess

def run_command(command):
    return subprocess.call(command, shell=True)

def dangerous_eval(code):
    return eval(code)
''')
            
            print(f"📁 Created test repository at: {temp_path}")
            print(f"📄 Test files: {list(backend_dir.glob('*.py'))}")
            
            # Initialize code review agent
            print("\n🚀 Initializing Code Review Agent...")
            agent = CodeReviewAgent(temp_path, standalone=True)
            
            print(f"📊 Agent initialized successfully")
            print(f"   Production ML available: {agent.production_ml_analyzer is not None}")
            print(f"   Advanced ML available: {agent.advanced_ml_capabilities is not None}")
            
            # Run code review
            print("\n🔍 Running Code Review...")
            report = await agent.run_code_review()
            
            print(f"\n📊 Code Review Results:")
            print(f"   Status: {report.get('status', 'unknown')}")
            print(f"   Total findings: {len(agent.findings)}")
            
            # Check for ML-generated findings
            ml_findings = [f for f in agent.findings if f.category.startswith('ml_')]
            production_ml_findings = [f for f in agent.findings if 'production' in f.category.lower()]
            advanced_ml_findings = [f for f in agent.findings if 'advanced' in f.category.lower()]
            
            print(f"   ML findings: {len(ml_findings)}")
            print(f"   Production ML findings: {len(production_ml_findings)}")
            print(f"   Advanced ML findings: {len(advanced_ml_findings)}")
            
            # Show sample findings
            if agent.findings:
                print(f"\n🔍 Sample Findings:")
                for i, finding in enumerate(agent.findings[:5]):
                    print(f"   {i+1}. {finding.category}: {finding.message[:100]}...")
            
            # Check if ML analysis actually ran
            if len(ml_findings) > 0:
                print("\n✅ ML Analysis Successfully Generated Findings!")
                return True
            else:
                print("\n⚠️ ML Analysis Did Not Generate Findings")
                print("   This suggests ML analysis may not have run properly")
                return False
                
    except Exception as e:
        print(f"\n❌ Full Code Review Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Full Code Review Test Suite")
    print("Testing the complete code review process with ML integration")
    print()
    
    success = await test_full_code_review()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 FULL CODE REVIEW TEST PASSED!")
        print("✅ ML analysis is working correctly in the code review process")
        print("🚀 Your code review agent should now use ML analysis on real repositories")
    else:
        print("⚠️ FULL CODE REVIEW TEST FAILED")
        print("❌ ML analysis may not be working in the code review process")
        print("🔧 Check the error messages above for details")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
