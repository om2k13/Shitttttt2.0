#!/usr/bin/env python3
"""
Test Enhanced Pipeline
Tests if the enhanced pipeline is working with ML analysis
"""
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_enhanced_pipeline():
    """Test the enhanced pipeline"""
    print("🧪 Testing Enhanced Pipeline...")
    print("=" * 50)
    
    try:
        from app.review.enhanced_pipeline import run_enhanced_review
        
        # Create a test job ID
        test_job_id = "test_enhanced_pipeline"
        
        print(f"🚀 Testing enhanced pipeline with job ID: {test_job_id}")
        print("   This should include ML analysis...")
        
        # Test the enhanced pipeline
        results = await run_enhanced_review(test_job_id, include_code_review=True)
        
        print(f"\n📊 Enhanced Pipeline Results:")
        print(f"   Status: {results.get('status', 'unknown')}")
        print(f"   Total findings: {results.get('total_findings', 0)}")
        print(f"   Pipeline stages: {results.get('pipeline_stages', [])}")
        
        if results.get('status') == 'completed':
            print("\n✅ Enhanced Pipeline Test PASSED!")
            print("   The enhanced pipeline is working correctly")
            return True
        else:
            print(f"\n⚠️ Enhanced Pipeline Test FAILED")
            print(f"   Status: {results.get('status')}")
            print(f"   Error: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Enhanced Pipeline Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Enhanced Pipeline Test Suite")
    print("Testing if the enhanced pipeline is working with ML analysis")
    print()
    
    success = await test_enhanced_pipeline()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    if success:
        print("🎉 ENHANCED PIPELINE TEST PASSED!")
        print("✅ The enhanced pipeline is working correctly")
        print("🚀 Your code review agent should now run ML analysis automatically")
    else:
        print("⚠️ ENHANCED PIPELINE TEST FAILED")
        print("❌ The enhanced pipeline may not be working properly")
        print("🔧 Check the error messages above for details")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
