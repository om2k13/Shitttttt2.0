#!/usr/bin/env python3
"""
Test Lazy-Loaded ML System
Tests the code review agent with lazy-loaded ML models
"""
import sys
import asyncio
from pathlib import Path
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_lazy_ml_code_review():
    """Test code review with lazy-loaded ML models"""
    print("🧪 Testing Lazy-Loaded ML Code Review...")
    print("=" * 60)
    
    try:
        from app.review.code_review_agent import CodeReviewAgent
        
        # Create test repository with problematic code
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create problematic Python file
            test_file = temp_path / "dangerous.py"
            test_file.write_text('''
import os
import subprocess

def dangerous_function(user_input):
    # This should trigger ML security analysis
    os.system(user_input)
    subprocess.call(user_input, shell=True)
    return eval(user_input)

def complex_function(data, condition1, condition2):
    # This should trigger ML complexity analysis
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
''')
            
            print(f"📁 Created test repository at: {temp_path}")
            print(f"📄 Test file: {test_file.name}")
            
            # Initialize code review agent (ML models not loaded yet)
            print("\n🚀 Initializing Code Review Agent...")
            agent = CodeReviewAgent(temp_path, standalone=True)
            
            print(f"📊 Agent initialized successfully")
            print(f"   Production ML loaded: {agent.production_ml_analyzer is not None}")
            print(f"   Advanced ML loaded: {agent.advanced_ml_capabilities is not None}")
            
            # Run code review (this should trigger lazy loading)
            print("\n🔍 Running Code Review with Lazy ML Loading...")
            report = await agent.run_code_review()
            
            print(f"\n📊 Code Review Results:")
            print(f"   Status: {report.get('status', 'unknown')}")
            print(f"   Total findings: {len(agent.findings)}")
            
            # Check ML findings
            ml_findings = [f for f in agent.findings if f.category.startswith('ml_')]
            production_ml_findings = [f for f in agent.findings if 'production' in f.category.lower()]
            advanced_ml_findings = [f for f in agent.findings if 'advanced' in f.category.lower()]
            
            print(f"   ML findings: {len(ml_findings)}")
            print(f"   Production ML findings: {len(production_ml_findings)}")
            print(f"   Advanced ML findings: {len(advanced_ml_findings)}")
            
            # Show sample findings
            if agent.findings:
                print(f"\n🔍 Sample Findings:")
                for i, finding in enumerate(agent.findings[:10]):
                    print(f"   {i+1}. {finding.category}: {finding.message[:80]}...")
            
            # Check if ML analysis worked
            if len(ml_findings) > 0:
                print("\n✅ ML Analysis Successfully Generated Findings!")
                print(f"   Found {len(ml_findings)} ML-categorized findings")
                return True
            else:
                print("\n⚠️ ML Analysis Did Not Generate ML-Categorized Findings")
                print("   This suggests the ML analysis may not have run properly")
                return False
                
    except Exception as e:
        print(f"\n❌ Lazy ML Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Lazy-Loaded ML Test Suite")
    print("Testing code review with lazy-loaded ML models")
    print()
    
    success = await test_lazy_ml_code_review()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 LAZY ML TEST PASSED!")
        print("✅ ML analysis is working with lazy loading")
        print("🚀 Your code review agent should now work without segmentation faults")
    else:
        print("⚠️ LAZY ML TEST FAILED")
        print("❌ ML analysis may not be working properly")
        print("🔧 Check the error messages above for details")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
