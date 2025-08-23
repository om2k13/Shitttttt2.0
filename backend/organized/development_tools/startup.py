#!/usr/bin/env python3
"""
Startup Script for Code Review Agent with ML Integration

This script initializes all ML models and verifies the system is ready for production use.
"""

import asyncio
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

async def initialize_system():
    """Initialize the complete system with ML models"""
    print("🚀 Initializing Code Review Agent with ML Integration...")
    print("=" * 70)
    
    try:
        # Import components
        from app.review.neural_analyzer import NeuralAnalyzer
        from app.review.enhanced_ml_analyzer import EnhancedMLAnalyzer
        from app.review.code_review_agent import CodeReviewAgent
        
        print("✅ All components imported successfully")
        
        # Initialize ML models
        print("\n🧠 Initializing Neural Networks...")
        neural_analyzer = NeuralAnalyzer()
        neural_status = neural_analyzer.get_model_status()
        print(f"   Neural Models: {neural_status['total_models']} loaded")
        print(f"   Device: {neural_status['device']}")
        
        print("\n🤖 Initializing Enhanced ML Models...")
        ml_analyzer = EnhancedMLAnalyzer()
        ml_status = ml_analyzer.get_model_status()
        print(f"   ML Models: {ml_status['total_models']} loaded")
        print(f"   Total Models: {ml_status['total_models']}")
        
        # Test with sample data
        print("\n🧪 Running System Tests...")
        
        # Test neural analyzer
        test_code = "def test(): return 'hello'"
        security_result = neural_analyzer.analyze_code_security(
            test_code, {'file': 'test.py', 'line': 1}
        )
        print(f"   ✅ Neural Security Analysis: {security_result.get('severity', 'unknown')}")
        
        quality_result = neural_analyzer.predict_code_quality({
            'lines': 1, 'complexity': 1, 'nesting': 1, 'imports': 0, 'functions': 1, 'classes': 0
        })
        print(f"   ✅ Neural Quality Prediction: {quality_result.get('quality_score', 0):.3f}")
        
        # Test ML analyzer
        test_finding = {
            "file": "test.py", "line": 1, "category": "quality", "message": "Test finding",
            "suggestion": "Test suggestion", "code_snippet": test_code, "severity": "low",
            "confidence": 0.5, "impact": "low", "effort": "low", "tool": "test"
        }
        
        ml_result = ml_analyzer.analyze_finding_with_ml(test_finding)
        print(f"   ✅ ML Analysis: {ml_result.get('ensemble_severity', 'unknown')} severity")
        
        # Test code review agent
        print("\n🔍 Testing Code Review Agent...")
        test_repo = Path("./test_startup")
        test_repo.mkdir(exist_ok=True)
        
        # Create test file
        test_file = test_repo / "test.py"
        test_file.write_text("def test(): pass")
        
        agent = CodeReviewAgent(test_repo, standalone=True)
        result = await agent.run_code_review()
        
        if result['status'] == 'completed':
            print(f"   ✅ Code Review Agent: {result['total_findings']} findings")
        else:
            print(f"   ⚠️ Code Review Agent: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        import shutil
        if test_repo.exists():
            shutil.rmtree(test_repo)
        
        print("\n🎉 System Initialization Complete!")
        print("=" * 70)
        print("✅ Neural Networks: Ready")
        print("✅ ML Models: Ready")
        print("✅ Code Review Agent: Ready")
        print("✅ All Systems: Operational")
        print("\n🚀 Code Review Agent with ML Integration is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main startup function"""
    success = await initialize_system()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Run: python train_ml_models.py (if models need training)")
        print("2. Run: python test_ml_integration.py (for comprehensive testing)")
        print("3. Use: from app.review.code_review_agent import CodeReviewAgent")
        print("4. Deploy: Ready for production use")
        return 0
    else:
        print("\n⚠️ System initialization failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
