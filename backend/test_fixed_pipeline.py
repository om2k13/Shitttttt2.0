#!/usr/bin/env python3
"""
Test Fixed Pipeline
Tests if the fixed pipeline is working with ML analysis
"""
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_fixed_pipeline():
    """Test the fixed pipeline with ML analysis"""
    print("🧪 Testing Fixed Pipeline with ML Analysis...")
    print("=" * 60)
    
    try:
        from app.review.pipeline import run_review
        
        # Create a test job ID
        test_job_id = "test_fixed_pipeline"
        
        print(f"🚀 Testing fixed pipeline with job ID: {test_job_id}")
        print("   This should now include ML analysis...")
        
        # Test the fixed pipeline
        results = await run_review(test_job_id)
        
        print(f"\n📊 Fixed Pipeline Results:")
        print(f"   Results: {results}")
        
        if results:
            print("\n✅ Fixed Pipeline Test PASSED!")
            print("   The fixed pipeline is working correctly")
            return True
        else:
            print(f"\n⚠️ Fixed Pipeline Test FAILED")
            print(f"   No results returned")
            return False
            
    except Exception as e:
        print(f"\n❌ Fixed Pipeline Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Fixed Pipeline Test Suite")
    print("Testing if the fixed pipeline is working with ML analysis")
    print()
    
    success = await test_fixed_pipeline()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 FIXED PIPELINE TEST PASSED!")
        print("✅ The fixed pipeline is working correctly")
        print("🚀 Your code review agent should now run ML analysis automatically")
    else:
        print("⚠️ FIXED PIPELINE TEST FAILED")
        print("❌ The fixed pipeline may not be working properly")
        print("🔧 Check the error messages above for details")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
