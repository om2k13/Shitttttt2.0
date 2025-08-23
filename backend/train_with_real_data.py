"""
Train ML Models with Real Code Data

This script trains our ML models using the real training data collected
from actual code analysis instead of synthetic data.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib

class RealDataTrainer:
    """Trains ML models with real code analysis data"""
    
    def __init__(self, data_dir: str = "real_code_analysis"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.training_results = {}
        
    def load_real_data(self):
        """Load the real training data we collected"""
        print("📊 Loading real training data...")
        
        # Load features and labels
        self.X = np.load(self.data_dir / "features.npy")
        self.y_quality = np.load(self.data_dir / "quality_labels.npy")
        self.y_security = np.load(self.data_dir / "security_labels.npy")
        
        # Load metadata
        with open(self.data_dir / "dataset_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        print(f"📊 Quality distribution: {self.metadata['quality_distribution']}")
        print(f"📊 Security distribution: {self.metadata['security_distribution']}")
        
        # Split data for training
        self.X_train_quality, self.X_test_quality, self.y_train_quality, self.y_test_quality = train_test_split(
            self.X, self.y_quality, test_size=0.2, random_state=42, stratify=self.y_quality
        )
        
        self.X_train_security, self.X_test_security, self.y_train_security, self.y_test_security = train_test_split(
            self.X, self.y_security, test_size=0.2, random_state=42, stratify=self.y_security
        )
        
        print(f"🔧 Training set: {len(self.X_train_quality)} samples")
        print(f"🧪 Test set: {len(self.X_test_quality)} samples")
    
    def train_quality_models(self):
        """Train models for code quality prediction"""
        print("\n🎯 Training Code Quality Models...")
        
        # Random Forest
        print("🌲 Training Random Forest...")
        rf_quality = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_random_forest'] = rf_quality
        
        # Gradient Boosting
        print("📈 Training Gradient Boosting...")
        gb_quality = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_gradient_boosting'] = gb_quality
        
        # XGBoost
        print("🚀 Training XGBoost...")
        xgb_quality = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_xgboost'] = xgb_quality
        
        # LightGBM
        print("💡 Training LightGBM...")
        lgb_quality = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgb_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_lightgbm'] = lgb_quality
        
        # SVM
        print("🔧 Training SVM...")
        svm_quality = SVC(kernel='rbf', random_state=42)
        svm_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_svm'] = svm_quality
        
        # Neural Network
        print("🧠 Training Neural Network...")
        mlp_quality = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_mlp'] = mlp_quality
        
        print("✅ Quality models trained successfully!")
    
    def train_security_models(self):
        """Train models for security vulnerability prediction"""
        print("\n🔒 Training Security Models...")
        
        # Random Forest
        print("🌲 Training Random Forest...")
        rf_security = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_random_forest'] = rf_security
        
        # Gradient Boosting
        print("📈 Training Gradient Boosting...")
        gb_security = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_gradient_boosting'] = gb_security
        
        # XGBoost
        print("🚀 Training XGBoost...")
        xgb_security = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_xgboost'] = xgb_security
        
        # LightGBM
        print("💡 Training LightGBM...")
        lgb_security = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgb_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_lightgbm'] = lgb_security
        
        # Logistic Regression
        print("📊 Training Logistic Regression...")
        lr_security = LogisticRegression(random_state=42, max_iter=1000)
        lr_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_logistic_regression'] = lr_security
        
        print("✅ Security models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n📊 Evaluating Models...")
        
        self.training_results = {}
        
        # Evaluate quality models
        print("\n🎯 Code Quality Models:")
        for name, model in self.models.items():
            if 'quality' in name:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test_quality)
                    accuracy = accuracy_score(self.y_test_quality, y_pred)
                    self.training_results[name] = {
                        'accuracy': accuracy,
                        'test_predictions': y_pred.tolist()
                    }
                    print(f"  {name}: {accuracy:.3f}")
        
        # Evaluate security models
        print("\n🔒 Security Models:")
        for name, model in self.models.items():
            if 'security' in name:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test_security)
                    accuracy = accuracy_score(self.y_test_security, y_pred)
                    self.training_results[name] = {
                        'accuracy': accuracy,
                        'test_predictions': y_pred.tolist()
                    }
                    print(f"  {name}: {accuracy:.3f}")
    
    def save_models(self, output_dir: str = "trained_models"):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving models to {output_path}...")
        
        # Save individual models
        for name, model in self.models.items():
            model_path = output_path / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  ✅ Saved {name}")
        
        # Save training results
        results_path = output_path / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save feature names
        feature_names = ['lines', 'complexity', 'nesting', 'imports', 'functions', 'classes']
        features_path = output_path / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        
        # Save metadata
        metadata_path = output_path / "model_metadata.json"
        model_metadata = {
            "total_models": len(self.models),
            "quality_models": [name for name in self.models.keys() if 'quality' in name],
            "security_models": [name for name in self.models.keys() if 'security' in name],
            "features": self.X.shape[1],
            "training_samples": len(self.X_train_quality),
            "test_samples": len(self.X_test_quality),
            "data_source": "real_code_analysis"
        }
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ All models saved to {output_path}")
        return output_path
    
    def create_ensemble_model(self):
        """Create an ensemble model that combines predictions"""
        print("\n🤝 Creating Ensemble Model...")
        
        # For quality prediction
        quality_predictions = []
        for name, model in self.models.items():
            if 'quality' in name and hasattr(model, 'predict_proba'):
                try:
                    pred = model.predict_proba(self.X_test_quality)
                    quality_predictions.append(pred)
                except:
                    pass
        
        if quality_predictions:
            # Average the probabilities
            ensemble_quality = np.mean(quality_predictions, axis=0)
            ensemble_quality_pred = np.argmax(ensemble_quality, axis=1)
            ensemble_quality_acc = accuracy_score(self.y_test_quality, ensemble_quality_pred)
            print(f"  Ensemble Quality Accuracy: {ensemble_quality_acc:.3f}")
        
        # For security prediction
        security_predictions = []
        for name, model in self.models.items():
            if 'security' in name and hasattr(model, 'predict_proba'):
                try:
                    pred = model.predict_proba(self.X_test_security)
                    security_predictions.append(pred)
                except:
                    pass
        
        if security_predictions:
            # Average the probabilities
            ensemble_security = np.mean(security_predictions, axis=0)
            ensemble_security_pred = np.argmax(ensemble_security, axis=1)
            ensemble_security_acc = accuracy_score(self.y_test_security, ensemble_security_pred)
            print(f"  Ensemble Security Accuracy: {ensemble_security_acc:.3f}")
    
    def train_all(self):
        """Train all models with real data"""
        print("🚀 Starting ML Model Training with Real Data...")
        
        # Load data
        self.load_real_data()
        
        # Train models
        self.train_quality_models()
        self.train_security_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Create ensemble
        self.create_ensemble_model()
        
        # Save models
        output_dir = self.save_models()
        
        print(f"\n🎉 Training Complete! Models saved to: {output_dir}")
        return output_dir

def main():
    """Main function to train models with real data"""
    trainer = RealDataTrainer()
    output_dir = trainer.train_all()
    print(f"🎯 Your ML models are now trained with REAL data in: {output_dir}")

if __name__ == "__main__":
    main()
