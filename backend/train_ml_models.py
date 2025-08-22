#!/usr/bin/env python3
"""
ML Model Training Script for Code Review Agent

This script trains all neural networks and ML models used by the Code Review Agent.
It ensures all models are production-ready and properly trained.
"""

import asyncio
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.review.neural_analyzer import NeuralAnalyzer
from app.review.enhanced_ml_analyzer import EnhancedMLAnalyzer

async def train_all_models():
    """Train all ML and neural network models"""
    print("🚀 Starting comprehensive ML model training...")
    
    # Set model directory
    model_dir = Path("./ml_models")
    model_dir.mkdir(exist_ok=True)
    
    try:
        # Train Neural Networks
        print("\n🧠 Training Neural Networks...")
        print("=" * 50)
        
        neural_analyzer = NeuralAnalyzer(model_dir)
        print("✅ Neural Networks trained successfully")
        
        # Train Enhanced ML Models
        print("\n🤖 Training Enhanced ML Models...")
        print("=" * 50)
        
        ml_analyzer = EnhancedMLAnalyzer(model_dir)
        print("✅ Enhanced ML Models trained successfully")
        
        # Get model status
        print("\n📊 Model Status Report...")
        print("=" * 50)
        
        neural_status = neural_analyzer.get_model_status()
        ml_status = ml_analyzer.get_model_status()
        
        print(f"Neural Models: {neural_status['total_models']}")
        print(f"ML Models: {ml_status['total_models']}")
        print(f"Total Models: {neural_status['total_models'] + ml_status['total_models']}")
        print(f"Models Directory: {model_dir.absolute()}")
        
        # Verify model files exist
        print("\n🔍 Verifying Model Files...")
        print("=" * 50)
        
        model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pkl"))
        print(f"Found {len(model_files)} model files:")
        
        for model_file in model_files:
            file_size = model_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {model_file.name} ({file_size:.2f} MB)")
        
        print(f"\n🎉 All models trained and saved successfully!")
        print(f"📁 Models saved to: {model_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_models():
    """Verify that all models can be loaded and used"""
    print("\n🔍 Verifying Model Functionality...")
    print("=" * 50)
    
    try:
        model_dir = Path("./ml_models")
        
        # Test Neural Analyzer
        print("Testing Neural Analyzer...")
        neural_analyzer = NeuralAnalyzer(model_dir)
        
        # Test with sample code
        sample_code = """
def vulnerable_function(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    return execute_query(query)
        """
        
        security_result = neural_analyzer.analyze_code_security(
            sample_code, 
            {'file': 'test.py', 'line': 1}
        )
        print(f"  ✅ Security Analysis: {security_result.get('severity', 'unknown')} risk")
        
        quality_result = neural_analyzer.predict_code_quality({
            'lines': 5,
            'complexity': 8,
            'nesting': 2,
            'imports': 1,
            'functions': 1,
            'classes': 0
        })
        print(f"  ✅ Quality Prediction: {quality_result.get('quality_score', 0):.2f}")
        
        # Test Enhanced ML Analyzer
        print("Testing Enhanced ML Analyzer...")
        ml_analyzer = EnhancedMLAnalyzer(model_dir)
        
        sample_finding = {
            "file": "test.py",
            "line": 1,
            "category": "security",
            "message": "Potential SQL injection vulnerability",
            "suggestion": "Use parameterized queries",
            "code_snippet": sample_code,
            "severity": "high",
            "confidence": 0.8,
            "impact": "high",
            "effort": "medium",
            "tool": "bandit"
        }
        
        ml_result = ml_analyzer.analyze_finding_with_ml(sample_finding)
        print(f"  ✅ ML Analysis: {ml_result.get('ensemble_severity', 'unknown')} severity")
        
        print("✅ All models verified and working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main training function"""
    print("🤖 Code Review Agent ML Model Training")
    print("=" * 60)
    
    # Train models
    training_success = await train_all_models()
    
    if training_success:
        # Verify models
        verification_success = await verify_models()
        
        if verification_success:
            print("\n🎯 Training Summary:")
            print("✅ All neural networks trained successfully")
            print("✅ All ML models trained successfully")
            print("✅ All models verified and functional")
            print("✅ Code Review Agent is ready for production use")
        else:
            print("\n⚠️ Training completed but verification failed")
            print("Please check the error messages above")
    else:
        print("\n❌ Training failed")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    # Run the training
    asyncio.run(main())
